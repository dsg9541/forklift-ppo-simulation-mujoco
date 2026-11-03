import numpy as np

class ForkliftController:
    """叉车底层控制器：负责速度PID、转向角度限制及货叉高度控制"""
    
    def __init__(self, model, data, max_steer=1.0):
        self.model = model
        self.data = data
        
        # 控制参数
        self.KP_SPEED = 1000.0
        self.KI_SPEED = 50.0
        self.KD_SPEED = 20.0
        self.INTEGRAL_MAX = 1000.0
        self.MAX_TORQUE = 2000.0
        self.WHEEL_RADIUS = 0.2
        self.L_AXLE = 1.8
        self.W_TRACK = 1.0
        self.MAX_STEER = max_steer
        self.MAX_STEER_RATE = 4.0  

        # PID 状态
        self.integral_left = 0.0
        self.integral_right = 0.0
        self.last_error_left = 0.0
        self.last_error_right = 0.0
        self.current_steer = 0.0

    def reset(self):
        """重置控制器内部状态"""
        self.integral_left = 0.0
        self.integral_right = 0.0
        self.last_error_left = 0.0
        self.last_error_right = 0.0
        self.current_steer = 0.0

    def apply_control(self, v_cmd, steer_cmd, max_lin_speed, has_cargo):
        """执行底层控制指令"""
        err = steer_cmd - self.current_steer
        max_chg = self.MAX_STEER_RATE * self.model.opt.timestep
        self.current_steer += np.clip(err, -max_chg, max_chg)

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

        tl = self.KP_SPEED * el + self.KI_SPEED * self.integral_left + self.KD_SPEED * dl
        tr = self.KP_SPEED * er + self.KI_SPEED * self.integral_right + self.KD_SPEED * dr
        tl = np.clip(tl, -self.MAX_TORQUE, self.MAX_TORQUE)
        tr = np.clip(tr, -self.MAX_TORQUE, self.MAX_TORQUE)

        self.last_error_left = el
        self.last_error_right = er
        self.data.ctrl[0] = tl
        self.data.ctrl[1] = tr

        delta = delta_rear
        if abs(delta) < 1e-6:
            ls, rs = 0.0, 0.0
        else:
            if delta > 0:
                R = self.L_AXLE / np.tan(delta)
                ls = np.arctan(self.L_AXLE / (R - self.W_TRACK/2))
                rs = np.arctan(self.L_AXLE / (R + self.W_TRACK/2))
            else:
                dp = -delta
                R = self.L_AXLE / np.tan(dp)
                ls = -np.arctan(self.L_AXLE / (R + self.W_TRACK/2))
                rs = -np.arctan(self.L_AXLE / (R - self.W_TRACK/2))
        
        self.data.ctrl[2] = np.clip(ls, -1.0, 1.0)
        self.data.ctrl[3] = np.clip(rs, -1.0, 1.0)

        # 货叉高度控制
        if has_cargo:
            self.data.ctrl[4] = 0.4
        else:
            self.data.ctrl[4] = -0.23

        # 制动控制
        lin_vel = (wl + wr) / 2.0 * self.WHEEL_RADIUS
        if abs(v_des) < 0.01:
            brake_torque = -np.sign(lin_vel) * min(800, abs(lin_vel) * 2000)
        else:
            brake_torque = 0.0
            
        self.data.ctrl[5] = brake_torque
        self.data.ctrl[6] = brake_torque