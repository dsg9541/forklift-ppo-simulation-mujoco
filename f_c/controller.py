import numpy as np
import mujoco


class ForkliftController:
    """叉车底层控制器：负责速度PID、转向角度限制及货叉高度控制"""
    
    def __init__(self, model, data, max_steer=1.0636):
        self.model = model
        self.data = data
        
        # 控制参数（与训练代码完全一致）
        self.KP_SPEED = 1000.0
        self.KI_SPEED = 50.0
        self.KD_SPEED = 20.0
        self.INTEGRAL_MAX = 1000.0
        self.MAX_TORQUE = 2000.0
        self.FEEDFORWARD_GAIN = 200.0      # ← 新增：与训练代码一致
        self.WHEEL_RADIUS = 0.2
        self.L_AXLE = 1.8
        self.W_TRACK = 1.0
        self.MAX_STEER = max_steer
        self.MAX_STEER_RATE = 4.0  

        # 货叉参数
        self.forks_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "forks")
        self.lift_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lift_joint")
        self.lift_joint_range = [-0.23, 1.45]

        # 后轮转向关节（用来做角度伺服，见 _apply_steering）
        self._steer_joints = []
        for jname in ("steer_left_joint", "steer_right_joint"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            self._steer_joints.append(
                (int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])))
        self.STEER_KP = 20.0     # 舵角伺服 P 增益（控制量上限 1.315N·m）
        self.STEER_KD = 1.0      # 舵角伺服 D 增益
        self.steer_servo = True  # False = 退回"把目标角度当力矩"的老写法
        
        self.use_raw_target = True
        self.raw_target = -0.23
        self.target_height = 0.0

        # PID 状态
        self.integral_left = 0.0
        self.integral_right = 0.0
        self.last_error_left = 0.0
        self.last_error_right = 0.0
        self.current_steer = 0.0
        self._v_prev = 0.0            # 【30 号】creep 指令速度斜坡用的上一个值

    def reset(self):
        """重置控制器内部状态"""
        self.integral_left = 0.0
        self.integral_right = 0.0
        self.last_error_left = 0.0
        self.last_error_right = 0.0
        self.current_steer = 0.0
        self._v_prev = 0.0            # 【30 号】
        self.use_raw_target = True
        self.raw_target = -0.23

    def set_fork_target_height(self, target_height):
        """设置货叉下表面的目标全局高度"""
        self.use_raw_target = False
        self.target_height = target_height

    def set_fork_raw_target(self, raw_val):
        """直接下发 ctrl[4] 数值"""
        self.use_raw_target = True
        self.raw_target = raw_val

    # ------------------------------------------------------------------
    # 货叉"到位锁止"【28 号】
    # ------------------------------------------------------------------
    # 为什么：升降伺服（lift_act kp=5000/kv=2000 + lift_joint damping=150）带
    # 71kg 载荷时阻尼不足，货叉到位后会自激上下摆（实测峰峰 ~0.30m，lift_act
    # 波幅 ~0.2），重心跟着晃、模型对准被带偏。调增益又会把"能不能升起来"搞坏
    # （kv 太低直接升不动，用户实测）。
    # 真车的做法是：升到位后液压锁住。这里等价实现——**到位后把升降轴的阻尼
    # 直接拉大**（相当于给升降轴加一个大刹车），轴被"锁"在当前高度：
    #   · 位置伺服继续提供托举力（高度不掉）；
    #   · 阻尼足够大 → 振荡被吃掉；
    #   · 只在"模型段/保持段"锁，需要升降货叉时解锁，不影响升降动作本身；
    #   · 只动 lift_joint 一个自由度，轮子/转向/悬挂/驱动一律不碰。
    FORK_LOCK_DAMPING = 40000.0

    def set_fork_lock(self, locked):
        """True = 锁住货叉（升降轴大阻尼），False = 恢复正常升降。"""
        if not hasattr(self, "_lift_dof"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                    "lift_joint")
            self._lift_dof = int(self.model.jnt_dofadr[jid])
            self._lift_damping0 = float(self.model.dof_damping[self._lift_dof])
        self._fork_locked = bool(locked)
        self.model.dof_damping[self._lift_dof] = (
            self.FORK_LOCK_DAMPING if locked else self._lift_damping0)

    def fork_at_target(self, tol=0.02):
        """货叉（升降轴）是不是已经到目标高度了。

        锁止**必须**以此做门控：只有到位了才锁。否则会出现"货叉还在上升途中
        被锁住 → 永远升不到位 → 低着叉齿去插货、撞到货堆"（用户实测）。
        """
        if not hasattr(self, "_lift_qadr"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                    "lift_joint")
            self._lift_qadr = int(self.model.jnt_qposadr[jid])
        return abs(float(self.data.qpos[self._lift_qadr])
                   - float(self.data.ctrl[4])) < tol

    def _update_fork_height(self):
        """内部方法：根据目标高度计算并下发 ctrl[4]"""
        if self.use_raw_target:
            self.data.ctrl[4] = np.clip(self.raw_target, self.lift_joint_range[0], self.lift_joint_range[1])
            return

        current_height = self.data.body(self.forks_body_id).xpos[2]
        error = self.target_height - current_height
        
        if abs(error) < 0.005:
            return
            
        if abs(error) > 0.2:
            speed = 0.25
        else:
            speed = 0.125
            
        dt = self.model.opt.timestep
        max_delta = speed * dt
        
        delta_ctrl = np.sign(error) * max_delta
            
        new_ctrl = self.data.ctrl[4] + delta_ctrl
        self.data.ctrl[4] = np.clip(new_ctrl, self.lift_joint_range[0], self.lift_joint_range[1])

    def apply_control(self, v_cmd, steer_cmd, max_lin_speed, has_cargo):
        """执行底层控制指令（与训练代码 _apply_control 完全一致）"""
        self._v_prev = 0.0      # 【30 号】模型段介入后，creep 的速度斜坡重新起算
        # --- 转向与速度控制 ---
        err = steer_cmd - self.current_steer
        max_chg = self.MAX_STEER_RATE * self.model.opt.timestep
        self.current_steer += np.clip(err, -max_chg, max_chg)

        self._apply_steering(-self.current_steer)

        v_des = v_cmd * max_lin_speed
        delta_rear = -self.current_steer
        tan_d = np.tan(delta_rear)
        factor = (self.W_TRACK/2) * tan_d / self.L_AXLE
        left_lin  = v_des * (1.0 - factor)
        right_lin = v_des * (1.0 + factor)

        target_wl = left_lin / self.WHEEL_RADIUS
        target_wr = right_lin / self.WHEEL_RADIUS

        wl = self.data.joint("spin_front_left").qvel[0]
        wr = self.data.joint("spin_front_right").qvel[0]

        el = target_wl - wl
        er = target_wr - wr
        if abs(v_des) < 0.001:
            self.integral_left = 0.0
            self.integral_right = 0.0

        dt = self.model.opt.timestep
        self.integral_left  += el * dt
        self.integral_right += er * dt
        self.integral_left  = np.clip(self.integral_left, -self.INTEGRAL_MAX, self.INTEGRAL_MAX)
        self.integral_right = np.clip(self.integral_right, -self.INTEGRAL_MAX, self.INTEGRAL_MAX)

        dl = (el - self.last_error_left) / dt if dt > 0 else 0.0
        dr = (er - self.last_error_right) / dt if dt > 0 else 0.0

        # ===== 前馈项（与训练代码一致） =====
        ff_left  = self.FEEDFORWARD_GAIN * target_wl
        ff_right = self.FEEDFORWARD_GAIN * target_wr

        # ===== 分别计算两轮 PID 力矩（含前馈） =====
        tl_raw = ff_left  + self.KP_SPEED * el + self.KI_SPEED * self.integral_left  + self.KD_SPEED * dl
        tr_raw = ff_right + self.KP_SPEED * er + self.KI_SPEED * self.integral_right + self.KD_SPEED * dr

        # ===== 两轮力矩取平均，保证一致 =====
        avg_torque = (tl_raw + tr_raw) / 2.0
        tl = np.clip(avg_torque, -self.MAX_TORQUE, self.MAX_TORQUE)
        tr = np.clip(avg_torque, -self.MAX_TORQUE, self.MAX_TORQUE)

        self.last_error_left = el
        self.last_error_right = er
        self.data.ctrl[0] = tl
        self.data.ctrl[1] = tr

        # --- 货叉高度闭环控制 ---
        self._update_fork_height()

        # --- 制动控制 ---
        lin_vel = (wl + wr) / 2.0 * self.WHEEL_RADIUS
        if abs(v_des) < 0.01:
            brake_torque = -np.sign(lin_vel) * min(800, abs(lin_vel) * 2000)
        else:
            brake_torque = 0.0
            
        self.data.ctrl[5] = brake_torque
        self.data.ctrl[6] = brake_torque

    # ==================================================================
    # 后轮转向：角度伺服
    # ------------------------------------------------------------------
    # 模型里 steer_left_act / steer_right_act 是 **motor** 型（gear=1、
    # ctrlrange=±1.315），即"控制量=力矩"，最大只有 1.315N·m，而转向关节
    # damping=2。按工程原来的写法（把目标阿克曼角当力矩直接下发），实测：
    #   1s 后后轮只到 -17°，6s 才到 -33°，**轨迹半径 4.4m**。
    # 改成对关节角做 P(+D) 伺服后：
    #   1s 就到 -53°，6s 车身转 48°，**轨迹半径 1.9m**（tools/steer_servo.py）。
    # 这跟 27 号设计文档里"1.0636rad 最大转向角"的意图一致——那个 ctrlrange
    # 本来就是按"目标角度"写的，只是执行器类型选错了。
    # ==================================================================
    def _apply_steering(self, delta_rear):
        """delta_rear = 后轮目标转角(rad)，内部按阿克曼关系分配左右轮。"""
        d = float(delta_rear)
        if abs(d) < 1e-6:
            targets = (0.0, 0.0)
        elif d > 0:
            R = self.L_AXLE / np.tan(d)
            targets = (np.arctan(self.L_AXLE / (R + self.W_TRACK / 2)),
                       np.arctan(self.L_AXLE / (R - self.W_TRACK / 2)))
        else:
            dp = -d
            R = self.L_AXLE / np.tan(dp)
            targets = (-np.arctan(self.L_AXLE / (R - self.W_TRACK / 2)),
                       -np.arctan(self.L_AXLE / (R + self.W_TRACK / 2)))
        for idx, ((qadr, dadr), target) in enumerate(
                zip(self._steer_joints, targets)):
            act = 2 + idx                     # ctrl[2]=左  ctrl[3]=右
            if not self.steer_servo:
                self.data.ctrl[act] = float(target)
                continue
            q = float(self.data.qpos[qadr])
            qd = float(self.data.qvel[dadr])
            tq = self.STEER_KP * (target - q) - self.STEER_KD * qd
            self.data.ctrl[act] = float(np.clip(tq, -1.315, 1.315))

    # ==================================================================
    # 低速精确控制（给"程序控制"用，训练模型不走这条路）
    # ------------------------------------------------------------------
    # 上面的 apply_control 与训练代码逐行一致，但它的速度环带 KD 项：
    # 在 dt=0.001 下把轮速噪声放大到力矩饱和（实测 v_des=0.15m/s 时车几乎
    # 不动、还会倒着走），程序对准需要的恰恰是 0.05~0.3m/s 的可控蠕行。
    # 这里用"只有 P 项 + 前馈"的轮速环，实测稳态速度线性跟随指令
    # （v_des=0.05→0.049、0.15→0.146、0.30→0.292 m/s，见 tools/creep_id.py）。
    # 转向与货叉高度的映射、制动逻辑与 apply_control 一致。
    # ==================================================================
    CREEP_KP = 200.0
    CREEP_MAX_TORQUE = 900.0
    # 载货时阻力矩大得多（实测带 115kg 载货时 P 环有 60% 静差、车只能爬到
    # 指令速度的四成），所以载货段把轮速环增益乘上去，见 creep_gain。
    creep_gain = 1.0

    def apply_creep(self, v_des, steer_cmd, accel_limit=None, fork_hold=False):
        """低速精确控制：v_des 单位 m/s，steer_cmd 单位 rad（后轮转角指令）。

        fork_hold=True：**不动货叉**（保持 ctrl[4] 原样）。
        为什么要这个开关：模型段 `_drive()` 每步直接写训练值 ctrl[4]=0.4（载货）
        /-0.23（空载），而 `apply_creep` 会按"爬升环"改写 ctrl[4]；两条路一起跑
        时 ctrl[4] 就在 0.4↔0.6 之间来回跳（用户实测 lift_act 波幅 ~0.2），
        货叉上下摆、重心跟着晃。末段的"停住/直线后退"属于模型段，必须保持货叉
        不动，所以给它们用 fork_hold=True。
        """
        self._fork_hold = bool(fork_hold)
        # 【30 号】把签名里一直没用上的 accel_limit 实现出来：按 m/s² 的上限把
        # 指令速度斜坡化。为什么需要它：程序段的直线移动原来都是"阶跃"给速度，
        # 托盘/货物在起步那一下就吃一个冲击 —— 实测放货误差 4~12cm 主要就是
        # 托盘在货叉上被甩出的位移。有了斜坡，就可以把直线速度整体提上去
        # （原来退 3m 用 0.15m/s 要 20s），同时**冲击比原来还小**。
        if accel_limit is not None and float(accel_limit) > 0.0:
            _lim = float(accel_limit) * float(self.model.opt.timestep)
            _pv = float(getattr(self, "_v_prev", 0.0))
            v_des = _pv + float(np.clip(float(v_des) - _pv, -_lim, _lim))
        self._v_prev = float(v_des)
        err = steer_cmd - self.current_steer
        max_chg = self.MAX_STEER_RATE * self.model.opt.timestep
        self.current_steer += float(np.clip(err, -max_chg, max_chg))

        delta_rear = -self.current_steer
        tan_d = np.tan(delta_rear)
        factor = (self.W_TRACK / 2.0) * tan_d / self.L_AXLE
        target_wl = v_des * (1.0 - factor) / self.WHEEL_RADIUS
        target_wr = v_des * (1.0 + factor) / self.WHEEL_RADIUS

        wl = self.data.joint("spin_front_left").qvel[0]
        wr = self.data.joint("spin_front_right").qvel[0]
        # 只有 P 项（tools/creep_id.py 标定过的形式，实测线性跟随）。
        # 试过"按训练脚本加前馈 ff=200×目标轮速"和"把 P 增益乘 4"：
        # 前者让带载低速段反而更慢（力矩常驻 300N·m 把轮子憋住），
        # 后者把力矩顶到上限 900 且低速发抖（实测放货"前移送入"倒退了 0.8m）。
        kp = self.CREEP_KP * float(getattr(self, "creep_gain", 1.0))
        self.data.ctrl[0] = np.clip(
            kp * (target_wl - wl), -self.CREEP_MAX_TORQUE,
            self.CREEP_MAX_TORQUE)
        self.data.ctrl[1] = np.clip(
            kp * (target_wr - wr), -self.CREEP_MAX_TORQUE,
            self.CREEP_MAX_TORQUE)

        self._apply_steering(delta_rear)

        if not getattr(self, "_fork_hold", False):
            self._update_fork_height()

        if abs(v_des) < 0.005:
            lin_vel = (wl + wr) / 2.0 * self.WHEEL_RADIUS
            brake = -np.sign(lin_vel) * min(800.0, abs(lin_vel) * 2000.0)
        else:
            brake = 0.0
        self.data.ctrl[5] = brake
        self.data.ctrl[6] = brake
