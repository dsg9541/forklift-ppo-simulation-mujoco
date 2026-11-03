"""平衡重叉车三轮取放作业：P1 第 3/2/1 层 → A/B 排货堆顶面，全程无人工干预。

底层 MuJoCo 3.12 + f_c/ 控制库；远距离用 ppo_2 导航，末段用 ppo_3 视觉对准，
配合程序兜底（对准复核、重进站、插货保护）。取放高度来自 model/height_table.json。

    python main.py                                   # 弹窗选三轮放货目标
    python main.py --no-target-ui                    # 用默认三轮（A2/B2/A3，seed 2200）
    python main.py --demo-targets A2,B2,A3 --seed 0  # 逐字复现冠军那一组
    python main.py --no-render                       # 无窗口复跑
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

# 策略网络定义（原 train_*.py）在 f_c/train/ 下，加入模块搜索路径后按原模块名导入。
_TRAIN_DIR = os.path.join(ROOT, "f_c", "train")
if os.path.isdir(_TRAIN_DIR) and _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)

import mujoco  # noqa: E402

from f_c import (ForkliftController, FineAlignController, PathPlanner,  # noqa: E402
                 merge_warehouse_and_forklift, GuidanceModel, Judge,
                 build_obs, build_lidar_scan, body_frame_goal)
from f_c.align_vision import (DOCK_STANDOFF_M, DockAligner, face_fit,  # noqa: E402
                              stack_lateral_error)
from f_c.vision_mask import MaskedCamera  # noqa: E402
from f_c.clearance import (clearance as foot_clearance,  # noqa: E402
                           boxes_from_info, path_clearance)

# ---- 末段蹭进 + 插货保护 ----
CREEP_IN_ENABLE = True
CREEP_IN_MAX_TICKS = 60              # 蹭进最多持续多少 tick
CREEP_IN_SPEED = 0.10                # 蹭进前进速度上限 (m/s)
CREEP_IN_LAT_TOL = 0.05              # 图像横向归一化容差
CREEP_IN_POS_TOL = 0.03              # 雷达站位距离容差 (m)
INSERT_GUARD_TICKS = 25000
# 插货保护：最多前进距离 (m)。
INSERT_MAX_FORWARD_M = 1.35
INSERT_SPEED = 0.14

# ---- 视野裁剪面 + 预停点阶段保护 ----
CAM_NEAR_ABS_M = 0.1535
CAM_FAR_ABS_M = 51.17

# ---- 程序控制对准 ----
ALG_ALIGN_ENABLE = True
ALG_ALIGN_LAT_TOL = 0.10        # 图像横向归一化容差（0=正中）
ALG_ALIGN_SYM_TOL = 0.08        # 门架雷达左右镜像差容差 (m)
ALG_ALIGN_NEAR_M = 3.40         # 距货堆正面最大距离 (m)：超过就不判为对准
ALG_ALIGN_MAX_TICKS = 900       # 对准闭环最长 tick
ALG_ALIGN_V = 0.12              # 对准闭环的前进/后退速度 (m/s)
ALG_ALIGN_RETRY_MAX = 3         # 未对准时"回正+后退+重对"的最大次数
DOCK_STALL_TICKS = 40
ALIGN_CHECK_ENABLE = True
HANDOFF_YAW_OK_DEG = 70.0
HANDOFF_TURN_MAX_TICKS = 6000
HANDOFF_TURN_V = 0.22
HANDOFF_TURN_K = 1.30           # 程序转弯的航向比例增益（舵角 rad / 偏差 rad）
PROG_LOAD_LEN_M = 0.80       # 载货（托盘+货物）沿进站轴长度 (m)
PROG_FEED_GAP_TARGET_M = 0.046   # 闭环目标 (货堆面−载货后表面) m
PROG_FEED_GAP_MAX_EXTRA = 0.12  # 闭环额外前移上限 m
ALIGN_BACKOFF_M = 1.50          # 复核不过时后退多少米
REDOCK2_MAX = 3          # 每轮最多重进站几次
REDOCK2_NEAR_M = 1.5      # 距目标多近才允许硬插收尾
ALIGN_CHECK_RETRY_MAX = 8       # 最多复核重试这么多次（防死循环）
ALIGN_CHECK_LAT_TOL = 0.05      # 图像横向归一化容差（0 = 货物正中）
ALIGN_CHECK_SYM_TOL = 0.03
ALIGN_CHECK_LONG_TOL = 0.1

# 程序控制末段对准（27 号最终方案，替代 ppo_3）
DOCK_HANDOFF_M = 7.00
DOCK_GATE_DEG = 60.0
DOCK_MAX_TICKS = 14000
DOCK_RETRY_MAX = 2
DOCK_MIN_START_GAP = 1.20
# 预对准点：锚点正后方多远（沿进场方向的反方向）。最后一段折线就是
PRE_ALIGN_BACK = 3.20
# 末段对准的两级：
PPO3_ENGAGE_M = 1.50         # 距引导锚点多远交给 ppo_3（用户指定 1.5m）
PPO3_DECISIONS_FIRST = 500   # 第 1 轮对准的决策上限
PPO3_DECISIONS_RETRY = 500   # 后续轮次（含"后退重来"）的决策上限
PPO3_MAX_DECISIONS = PPO3_DECISIONS_FIRST   # 兼容旧引用
PPO3_MIN_KEEP = 0.004        # 处理后图里目标堆像素占比低于这个值就算"看不到"
# 原地摆头（多轮前进/后退把车头转到进场方向）：
TURN_ENGAGE_M = 5.00         # 距引导锚点多远开始转向
TURN_TRIGGER_DEG = 25.0      # 车头偏差超过多少度才走"弧线转向"（小偏差交给模型）
# 这一步（车头偏差 >25° 时的粗对准）**交给 ppo_2 做**：
TURN_COARSE_ENABLE = False
TURN_DONE_DEG = 10.0         # 车头偏差进到多少度以内就交给对准模型（用户指定 10°）
TURN_V = 0.35                # 弧线转向速度 (m/s)
TURN_STEER_MAX = 0.95        # 弧线转向的舵量上限（打满才够 2m 半径）
TURN_STEER_MAX = 0.45        # 摆头时的转角上限
TURN_MAX_TICKS = 9000        # 摆头最长时间
TURN_HALF_TICKS = 150        # 摆头：一相（前进/后退）持续多少个 tick
DOCK_STEPS = 10
PROG_ACCEL_LIMIT = 0.35
PROG_BACKOUT_V = 0.45
PROG_BACKOUT_V_LADEN = 0.22
PROG_FORWARD_V = 0.30
# ---- 进给末段减速用的最低速度 ----
FEED_STOP_V = 0.08
PRE_DOCK_NEAR_SPEED = 0.6            # 预停点接近段限速 (m/s)
PRE_DOCK_MAX_DECISIONS = 400         # 超过就转入蹭进兜底（训练单回合上限 500）
PRE_DOCK_PROGRESS_EVERY = 25         # 每多少次决策打印一次进度
WP_ARRIVE_M = 0.60                   # 逐路径点：位置到点判据 (m)
WP_YAW_TOL = 0.10                    # 逐路径点：航向到点判据 (rad，用户指定)
PRE_DOCK_TURN_RAD = 0.0
                                     # 交给 ppo_2 自己处理更好；需要时可调回 0.60 试）
PRE_DOCK_TURN_K = 1.0                # 转身转向增益
PRE_DOCK_TURN_V = 0.16               # 转身时的前进/后退速度 (m/s)
PRE_DOCK_TURN_MAX = 300              # 转身单独的上限 tick
# 引导点方位限幅：两个模型训练时目标都在车头 ±90° 内，超过就会"原地绕圈"。
NAV_BEARING_LIMIT = 0.60             # rad（≈34°）
NAV_AIM_DIST = 3.0                   # 引导点取正前方多远 (m)
# 兜底切换：偏离折线超过 1.2m（或车头偏差 >25°、前方净空不足）就把控制权
NAV_PROGRAM_FALLBACK = False
NAV_FALLBACK_DEV = 0.80
ROUTE_HEAD_MIN_M = 1.60
CLEAR_OUT_CLEAR_M = 0.60          # 脚下净空小于它就先插一个"让开"点
CLEAR_OUT_BACK_M = 2.50           # "让开"点离车多远
PIN_BACKOFF_TICKS = 200
ESCAPE_BACK_M = 0.9          # 脱困直线倒车距离 (m)
ESCAPE_BACK_V = 0.22
NAV_MODEL_ENTER_DEG = 25.0
                                  # 虽然到 90°，但仓库里空间小、2m/s 下模型
NAV_MODEL_BACK_DEG = 12.0
# 偏离折线超过 1.2m 就判定"模型走飞"，交给程序纯追踪拉回；拉回到
NAV_FALLBACK_BACK = 0.45

APPROACH_BACKOFF = 2.85          # 引导点距目标中心距离（与妙妙工具一致）
ROW_Y = {"A": 7.0, "B": -1.0}
PICK_STACK = "P1"
DROP_ZONE = "G1"

EMPTY_FORK_H = 0.03              # 空载行驶时货叉下表面高度
WITH_CARGO_FORK_H = 0.41
# 模型段的货叉和训练环境完全一致：训练脚本每一步都直接写
LIFT_RAW_CARGO = 0.4
LIFT_RAW_CARGO_HIGH = 0.60
LIFT_RAW_EMPTY = -0.23

ACTION_REPEAT = 50               # 每次模型决策的物理步数（与训练一致）
TICK_STEPS = 50
# 画面流畅度：窗口刷新插到物理步里，每多少个物理步刷一次
VIEW_SYNC_EVERY = 5
NAV_MAX_SPEED = 2.0              # 训练时 max_lin_speed
NAV_MODEL_SPEED = 1.50           # 部署侧给 ppo_2 的速度上限（训练是 2.0）
                                 # 训练场景是开阔地面，模型学会了"先加速、后转弯"
NAV_TURN_SLOW_DEG = 45.0
NAV_TURN_SLOW_SPEED = 0.45       # 大偏角段的速度上限 (m/s)
FINAL_MAX_SPEED = 2.0            # 末段必须与训练同量纲，否则闭环动态失配
LOOKAHEAD_M = 2.5
# 交接距离：模型在训练里是"从静止、目标 1~7m 在前方"开始学对准的，
NEAR_HANDOFF_M = 4.0

FINAL_POS_TOL = 0.02             # 目标点门限：2cm
FINAL_YAW_TOL = 0.020
FINAL_YAW_TOL_DROP = 0.023
FINAL_VEL_TOL = 0.1              # 0.1m/s
FINAL_MAX_DECISIONS = 600        # 训练时单回合上限 500 次决策，留一点余量

# ===== 交接协议（用户指定）=====
PRE_DOCK_BACK_M = 2.0            # 预停点距模糊目标的距离（1~3m 内取值）
PRE_DOCK_POS_TOL = 0.10          # 预停点位置容差
PRE_DOCK_YAW_TOL = 0.10          # 预停点航向容差（≈7°）
PRE_DOCK_VEL_TOL = 0.05          # 交接时的速度上限（要"停好"）
# 距预停点 4m 以外只用预瞄点做路径跟踪（避免 90° 直冲）；进入 4m 后改成
PRE_DOCK_STATIC_M = 4.0

# ===== 放货腿（27 号改动，按用户思路重做）=====
DROP_TURN_R = 2.40
                              # 半径≈2.5m**（tools 里量过：后轮舵角打到 1.06rad
DROP_TURN_STANDOFF = 0.90     # 圆弧结束点到锚点的**直线进站段**长度（留给
DROP_LANE_Y_MIN = -7.20       # 放货进站"转弯道"的最南位置（南围墙内表面
                              # -8.25 + 车半宽 0.63 + 余量）——B 排锚点在
DROP_LANE_Y_MAX = 6.00        # 同理的北侧上限（北围墙内表面 11.25）
DROP_LANE_RUN = 1.20          # 圆弧前的转弯道直线段 (m)
DROP_EDGE_CLEAR_M = 0.95      # 路径图里的边距货堆中心少于这个值就不走
DROP_DESCENT_RATIO = 1.80     # 下切段"水平/竖直"比（越大越平缓）：1.15 时
                              # 41° 的斜切在纯追踪里会冲过车道 1m，B 排那侧直接
DROP_DESCENT_MIN = 2.20       # 下切段最短水平距离 (m)
DROP_HANDOFF_HEAD_DEG = 20.0  # 交接给 ppo_3 时的车头偏差上限（训练用 20°）
DROP_CRUISE_SPEED = 0.45      # 载货巡航速度 (m/s)
DROP_APPROACH_SPEED = 0.30    # 进站段速度 (m/s)
DROP_LOOK = 1.60              # 折线跟踪前视距离 (m)
DROP_LOOK_SLOW = 1.00         # 进站段前视距离 (m)
SLIP_RATIO = 0.22
DROP_HANDOFF_M = 1.50         # 距锚点 1.5m 交给 ppo_3（用户指定）
DOCK_OVERSHOOT_M = 0.35
TIP_MIN_GAP_M = 0.02
TIP_SLOW_GAP_M = 0.50         # 进入这个净距就限速
TIP_SLOW_V = 0.25             # 限速值 (m/s)
MAST_TO_TIP_M = 0.9
                                  # （路径图里的"预交接点"在锚点后 1.75m，加上
# ---- 交接前必须把车摆进 ppo_3 的训练分布 ----
                                  # （A/B 排锚点在车道线上，车沿内车道(-2.3m)进站时
PREDOCK_BACK = 2.00
PREDOCK_POS_TOL = 0.12            # 到位判据（对模糊点）：位置 ≤12cm
PREDOCK_YAW_TOL_DEG = 3.44        # 车头偏差 ≤3.44°（0.06 rad，用户要求）
TURNIN_ENGAGE_M = 3.20        # 距引导锚点这么近就转"程序进站转弯"
TURNIN_ENGAGE_REM_M = 3.00    # 且折线剩余里程 ≤ 它（= 快走到折线末端了）：
                              # 取货腿从南侧车道绕过去时会从锚点旁边 2m 擦过，
TURNIN_DONE_DEG = 20.0
TURNIN_LAT_TOL = 0.45         # 且横向也要收进走廊线这么多米
TURNIN_V = 0.22               # 转弯速度 (m/s)
TURNIN_K = 1.30               # 航向比例增益（只在追瞄准点失效时兜底用）
TURNIN_AIM_AHEAD = 1.00       # 进站转弯的瞄准点 = 锚点再往前这么多（在走廊线上）
TURNIN_LOOK = 1.20            # 进站转弯纯追踪的前视距离 (m)
TURNIN_BACK_V = 0.18          # 退车速度 (m/s)
TURNIN_MAX_TICKS = 2500       # 转弯+退车最长 tick
BACK_BEFORE_PPO3_M = 1.00         # ppo_3 对不上时：先程序后退 1m 再交一次
DOCK_MIN_APPROACH_M = 1.15        # 放货腿：交给 ppo_3 的路径点离锚点的距离窗口
DOCK_MAX_APPROACH_M = 4.00
NAV_HANDOFF_SPEED = 0.12
NAV_HANDOFF_LOOK = 0.90
NAV_HANDOFF_SLOW_M = 2.60
NAV_MID_SPEED = 0.26              # 中段（3~6m）速度：给转弯留出响应时间
NAV_MID_LOOK = 1.30
NAV_MID_M = 6.00
                             # 进来：1.4m 处横向还差 0.89m、车头差 45°，交接
# ===== 取货腿交接门限（同一套逻辑）=====
PICK_HANDOFF_M = 1.50             # 交 ppo_3 的距离（取货锚点在空地上，好走）
# ===== 行驶安全阀 =====
NAV_SAFE_MARGIN = 0.10
                                  # （足迹本身已经比真实车体宽 5cm：车体实际
NAV_DEBUG = False                 # 开发期：打印慢速进站阶段的逐项状态
NAV_FOLLOW_SPEED = 0.55           # 折线跟踪巡航速度 (m/s)
NAV_FOLLOW_SLOW = 0.30            # 进站/近场速度 (m/s)
# 地面区（第 3 轮）：货叉先抬到取货堆顶之上再掉头，避免掉头时刮到 P1。
GROUND_TURN_FORK_H = 1.55     # 掉头前把货叉抬到取货堆顶上
# 下面两个只给"三点掉头"备用分支用（第六版已改用"直追锚点"完成 180° 掉头，
GROUND_TURN_DONE_DEG = 12.0   # 三点掉头：车头偏差进到多少度算转好
GROUND_TURN_PHASE_TICKS = 150 # 三点掉头：前进/后退每相持续 tick
GROUND_BACKUP_M = 1.50        # 地面区掉头前先退到"离开取货堆"的位置
INNER_LANE_Y = 1.5            # 路径图里的内侧捷径车道（不参与放货路径规划）
# 地面区（第 3 轮）：目标精确已知、没有货堆可看，走"绕场掉头 + 倒进锚点"

FORWARD_DROP_M = 1.15            # 放货前移量（小球后收前的值是 1.10）
BACKOUT_PICK_M = 3.0             # 取货后退量
PICK_BACKOUT_M = 2.30            # 返程起步直退量 (m)
PICK_BACKOUT_V = 0.45
PICK_BACKOUT_MAX_TICKS = 1200    # 退车最长 tick
NAV_DEPART_TRIGGER_DEG = 60.0    # 车头与折线方向差超过它才需要摆头
NAV_DEPART_DONE_DEG = 25.0       # 摆到差多少度以内交 ppo_2
NAV_DEPART_V = 0.22              # 摆头前进速度 (m/s)
NAV_DEPART_LOOK = 2.00           # 摆头时的前视距离 (m)
NAV_DEPART_MAX_TICKS = 1500      # 摆头最长 tick
BACKOUT_DROP_M = 3.0             # 放完货先程序后退 3m，之后整段导航（回货堆取
                                 # 下一层）全权交给 ppo_2。
LIFT_AFTER_INSERT_M = 0.15       # 插到位后抬升量

STATE_NAMES = {
    0: "导航至取货点", 1: "取货调整高度", 2: "前移插货", 3: "检测货物刹车",
    4: "抬升撤出高度", 5: "后退撤出货堆", 6: "设载货行驶高度", 7: "导航至放货点",
    8: "放货调整高度", 9: "前移送入", 10: "下落释放", 11: "倒车退出", 12: "完成",
}

EVAL_TASKS_FILE = "model/eval_groups.json"
EVAL_GROUP = None

DEMO_TARGETS = ["A2", "B2", "A3"]
TARGET_CHOICES = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4")

def choose_targets_ui(default=None):
    """启动前的简易界面：选三轮放货目标点。返回长度 3 的列表；取消返回 None"""
    import tkinter as tk
    from tkinter import messagebox, ttk

    d0 = list(default or DEMO_TARGETS)
    ui = {"targets": None}
    root = tk.Tk()
    root.title("平衡重叉车 · 放货目标选择")
    root.resizable(False, False)
    try:
        root.attributes("-topmost", True)
    except Exception:                                          # noqa: BLE001
        pass
    ttk.Label(root, text="三轮放货目标点",
              font=("Microsoft YaHei", 15, "bold")).grid(
        row=0, column=0, columnspan=2, padx=22, pady=(18, 4), sticky="w")
    ttk.Label(root, text="取货固定为 P1 第 3 / 2 / 1 层。",
              foreground="#555").grid(row=1, column=0, columnspan=2,
                                      padx=22, pady=(0, 12), sticky="w")
    vars_ = []
    for i in range(3):
        ttk.Label(root, text="第 %d 轮（P1 第 %d 层）" % (i + 1, 3 - i)).grid(
            row=2 + i, column=0, padx=(22, 10), pady=6, sticky="e")
        v = tk.StringVar(value=d0[i])
        ttk.Combobox(root, textvariable=v, values=list(TARGET_CHOICES),
                     state="readonly", width=10).grid(
            row=2 + i, column=1, padx=(0, 22), pady=6, sticky="w")
        vars_.append(v)

    def on_ok():
        sel = [v.get() for v in vars_]
        if len(set(sel)) != 3:
            messagebox.showwarning("目标重复", "三轮请选三个不同的货堆。")
            return
        ui["targets"] = sel
        root.destroy()

    def on_reset():
        for v, t in zip(vars_, d0):
            v.set(t)

    def on_cancel():
        ui["targets"] = None
        root.destroy()

    bar = ttk.Frame(root)
    bar.grid(row=6, column=0, columnspan=2, pady=(14, 18))
    ttk.Button(bar, text="开始作业", command=on_ok).pack(side="left", padx=6)
    ttk.Button(bar, text="恢复默认", command=on_reset).pack(side="left", padx=6)
    ttk.Button(bar, text="取消", command=on_cancel).pack(side="left", padx=6)
    root.mainloop()
    return ui["targets"]

def load_tasks():
    """三轮任务：取货高度与放货高度直接来自标定表"""
    with open("model/height_table.json", encoding="utf-8") as f:
        tbl = json.load(f)
    w = tbl["insert_windows"]
    d = tbl["dropoff"]
    if EVAL_GROUP is not None and os.path.exists(EVAL_TASKS_FILE):
        up = d["on2"]["carry_h"] - d["on2"]["T"]
        dn = d["on2"]["release_h"] - d["on2"]["T"]
        with open(EVAL_TASKS_FILE, encoding="utf-8") as f:
            group = json.load(f)["groups"][int(EVAL_GROUP)]
        tasks = []
        for r in group:
            top = tbl["stack_tops"][r["target"]]
            lay = int(r["pick_layer"])
            tasks.append(dict(rnd=r["rnd"], pick_layer=lay,
                              pick_h=round(w[f"L{lay}"]["mid"], 3),
                              target=r["target"], kind="stack",
                              drop_h=round(top + up, 3),
                              release_h=round(top + dn, 3)))
        return tasks
    return [
        dict(rnd=1, pick_layer=3, pick_h=round(w["L3"]["mid"], 3),
             target=DEMO_TARGETS[0], kind="stack",
             drop_h=d["on2"]["carry_h"], release_h=d["on2"]["release_h"]),
        dict(rnd=2, pick_layer=2, pick_h=round(w["L2"]["mid"], 3),
             target=DEMO_TARGETS[1], kind="stack",
             drop_h=d["on1"]["carry_h"], release_h=d["on1"]["release_h"]),
        dict(rnd=3, pick_layer=1, pick_h=round(w["L1"]["mid"], 3),
             # 暂时放弃地面放货区（那一腿的路径点数为 0，
             target=DEMO_TARGETS[2], kind="stack",
             drop_h=d["on1"]["carry_h"], release_h=d["on1"]["release_h"]),
    ]

class ForkliftTaskFSM:
    def __init__(self, use_model=True, render=False, seed=0, guidance=True,
                 verbose=True):
        self.verbose = verbose
        self.use_model = use_model
        self.use_guidance = guidance

        fl = open("model/forklift_train.xml", encoding="utf-8").read()
        wh = open("model/warehouse_env.xml", encoding="utf-8").read()
        self.model = mujoco.MjModel.from_xml_string(
            merge_warehouse_and_forklift(wh, fl))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.dt = self.model.opt.timestep
        # 相机近/远裁剪面：MuJoCo 存的是相对 stat.extent 的值，这里按训练场景的实际米数换算，
        # 保证 ppo_2 看到的图像与训练时同分布。
        self.model.vis.map.znear = CAM_NEAR_ABS_M / float(self.model.stat.extent)
        self.model.vis.map.zfar = CAM_FAR_ABS_M / float(self.model.stat.extent)

        B = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
        J = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.forklift_body_id = B("forklift")
        self.mast_body_id = B("mast")
        self.forks_body_id = B("forks")
        self.root_qpos_adr = int(self.model.jnt_qposadr[J("root_joint")])
        self.root_dof_adr = int(self.model.jnt_dofadr[J("root_joint")])
        self.lift_qpos_adr = int(self.model.jnt_qposadr[J("lift_joint")])
        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        self.controller = ForkliftController(self.model, self.data)
        self.fine_aligner = FineAlignController(self.model, self.data)
        self.judge = Judge(self.model, self.data, self.forklift_body_id)
        self.guidance = GuidanceModel(seed=seed)
        self.dock_aligner = DockAligner()

        with open("model/path_graph.json", encoding="utf-8") as f:
            graph = json.load(f)
        self.planner = PathPlanner(graph, self.model, self.data,
                                   self.forklift_body_id, self.root_qpos_adr)
        self.points = self.planner.points
        # 障碍矩形（货堆/围墙/木箱）：部署侧安全网用，见 f_c/clearance.py
        try:
            self.obstacle_boxes = boxes_from_info(
                os.path.join(ROOT, "model/warehouse_info.json"))
        except Exception as e:      # noqa: BLE001
            print(f"[警告] 障碍矩形读取失败（安全阀退化为只靠路径图）：{e}")
            self.obstacle_boxes = []

        self.renderer = None
        self.masked_cam = None
        if render:
            self.renderer = mujoco.Renderer(self.model, height=128, width=128)
            # ppo_3 的输入要做图像处理（只留目标货堆），需要一个带语义分割的相机
            self.masked_cam = MaskedCamera(self.model, 128, 128, "mast_cam")
            self.viewer = None

        self.nav_model = None
        self.align_model = None
        if use_model:
            self._load_models()

        self.tasks = load_tasks()
        self.total_rounds = len(self.tasks)
        self.state = 0
        self.round = 1
        self.route = []          # 本轮路径折线（精确点）
        self.route_idx = 0
        self.current_goal = None      # 真值目标
        self.guidance_goal = None     # 引导目标（带噪声）
        self.pick_bodies = []         # [(pallet_bid, cargo_bid)]
        self.has_cargo = False
        self._decisions = 0
        self._state_timer = 0
        self._release_done = False
        self.reset()

    def _load_models(self):
        from stable_baselines3 import PPO
        import train_input_104_3二版_131rad as tr_precise

        # 对准模型：train_mohu_12（图像处理后成功率 ~0.98 的那一版）。
        try:
            import train_mohu_12 as tr_align
            align_script = "train_mohu_12"
        except Exception:
            try:
                import train_mohu_10 as tr_align
                align_script = "train_mohu_10"
            except Exception:
                import train_mohu_2_90 as tr_align
                align_script = "train_mohu_2_90"

        def _load(mod, path):
            """显式给策略类与 policy_kwargs（ppo_2 用）"""
            return PPO.load(path, custom_objects={
                "policy": mod.VisionMinStdPolicy,
                "policy_kwargs": mod.vision_policy_kwargs(),
                "learning_rate": mod.linear_schedule()}, device="cpu")

        def _patch_policy(model, mod):
            """把 get_distribution 换成当前脚本里的那一版"""
            import types
            pol = model.policy
            pol.get_distribution = types.MethodType(
                mod.VisionMinStdPolicy.get_distribution, pol)
            pol.min_log_std = float(np.log(mod.VisionMinStdPolicy.MIN_STD_VALUE))
            return model

        def _load_saved(mod, path):
            """只替换策略类，policy_kwargs 用权重里存的那份"""
            return PPO.load(path, custom_objects={
                "policy": mod.VisionMinStdPolicy}, device="cpu")

        self.nav_model = _load(tr_precise, "robot/ppo_2.zip")
        align_path = None
        for cand in ("robot/ppo_3_v3.zip",
                     "robot/ppo_3_新3.zip",
                     "robot/ppo_3_新.zip", "robot/ppo_3_xin.zip",
                     "robot/ppo_3_新2.zip",
                     "robot/ppo_3.zip", "robot/ppo_3_v12.zip",
                     "train_model/train_mohu_12_latest.zip"):
            if os.path.exists(cand):
                align_path = cand
                break
        if align_path is None:
            raise FileNotFoundError("找不到 ppo_3 权重（robot/ppo_3_v12.zip）")
        try:
            self.align_model = _patch_policy(
                _load_saved(tr_align, align_path), tr_align)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] 直接加载 ppo_3 失败（{type(e).__name__}: {e}）"
                      f"，改用显式 policy_kwargs")
            self.align_model = _patch_policy(_load(tr_align, align_path),
                                             tr_align)
        if self.verbose:
            print(f"[INFO] 已加载 ppo_2(精确导航) + ppo_3(视觉对准, "
                  f"策略类来自 {align_script}，权重 {align_path}，"
                  f"输入图像={align_script} 的 CameraMasker 'uniform')")

    def _body_id(self, name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            # 报错时把"同类候选"和当前场景信息一起打出来：
            have = sorted(n for n in (mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                for i in range(self.model.nbody)) if n)
            key = name.split("_L")[0] if "_L" in name else name.split("_")[0]
            like = [n for n in have if n.startswith(key)][:12]
            raise ValueError(
                f"模型里没有 body {name}\n"
                f"  当前工作目录: {os.getcwd()}\n"
                f"  已加载的仓库场景: model/warehouse_env.xml"
                f"（该场景 body 共 {len(have)} 个）\n"
                f"  同类候选: {like}\n"
                f"  提示：27 号需要 stack_*/drop_zone_*/obstacle_* 这套命名的场景；"
                f"若这里列出来的是 cargo_*/shelf_*，说明加载到别的项目的场景了。")
        return bid

    def body_pos(self, name):
        return self.data.body(self._body_id(name)).xpos.copy()

    def body_yaw(self, name):
        R = self.data.xmat[self._body_id(name)].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def body_low_z(self, name):
        """body 所有 geom 的最低点（世界 z）"""
        bid = self._body_id(name)
        lo = np.inf
        adr = self.model.body_geomadr[bid]
        for gid in range(adr, adr + self.model.body_geomnum[bid]):
            R = self.data.geom_xmat[gid].reshape(3, 3)
            size = self.model.geom_size[gid]
            ext = (float(np.abs(R[2, :]).dot(size))
                   if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
                   else float(self.model.geom_rbound[gid]))
            lo = min(lo, float(self.data.geom_xpos[gid][2]) - ext)
        return float(lo)

    def pose(self):
        body = self.data.body(self.forklift_body_id)
        pos = body.xpos.copy()
        R = body.xmat.reshape(3, 3)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        return pos, R, yaw

    def fork_z(self):
        return float(self.data.body(self.forks_body_id).xpos[2])

    def speed(self):
        v = self.data.qvel[self.root_dof_adr:self.root_dof_adr + 3]
        x_axis = self.data.xmat[self.forklift_body_id].reshape(3, 3)[:, 0]
        return float(np.dot(v, x_axis))

    def _diag_place(self, tag):
        """诊断（只读，不改动作）：托盘相对靶心的纵向/横向误差 + 托盘相对货叉偏移"""
        try:
            _t = self._place_target_pose()
            _p = np.asarray(self.data.body(self._body_id(self.pick_pallet)).xpos,
                            dtype=float)
            _dx, _dy = _p[0] - _t[0], _p[1] - _t[1]
            _c, _s = float(np.cos(_t[2])), float(np.sin(_t[2]))
            self._log(f"[放置分解-{tag}] 纵向 {_dx * _c + _dy * _s:+.3f}m "
                      f"横向 {-_dx * _s + _dy * _c:+.3f}m")
        except Exception:                                     # noqa: BLE001
            pass
        _o = self._pallet_fork_offset()
        if _o:
            self._log(f"[滑移诊断] {tag}：托盘相对货叉 "
                      f"long={_o[0]:+.3f} lat={_o[1]:+.3f} up={_o[2]:+.3f}")

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def _status_log(self):
        if not self.verbose:
            return
        pos, _, yaw = self.pose()
        g = self.guidance_goal
        dg = ""
        if g is not None:
            eg = body_frame_goal(pos, yaw, g[:2], g[2])
            dg = f" 距引导锚点{float(np.hypot(eg[0], eg[1])):.2f}m"
        self._log(f"[状态] t={self._total_ticks} 第{self.round}轮 "
                  f"{STATE_NAMES.get(self.state, '?')} "
                  f"车({pos[0]:+.2f},{pos[1]:+.2f}) 航向{np.rad2deg(yaw):+.0f}°"
                  f" 货叉{self.fork_z():.2f} 载货={int(self.has_cargo)}{dg} "
                  f"模型={getattr(self, 'last_model', '-')}")

    def task(self):
        """当前轮次的任务"""
        return self.tasks[min(self.round, len(self.tasks)) - 1]

    def _pick_goal(self):
        """取货引导点：取货堆正北 2.85m，车头朝南"""
        p = self.body_pos(f"stack_{PICK_STACK}_L1_cargo")
        return np.array([p[0], p[1] + APPROACH_BACKOFF, -np.pi / 2.0])

    def _target_goal(self):
        """放货引导点：目标堆（以最下方货物为准）正南 2.85m，车头朝北"""
        t = self.task()
        if t["kind"] == "stack":
            name = f"stack_{t['target']}_L1_cargo"
        else:
            name = f"drop_zone_{t['target']}"
        p = self.body_pos(name)
        return np.array([p[0], p[1] - APPROACH_BACKOFF, np.pi / 2.0])

    def _place_target_pose(self):
        """被放上去那一层应该落在的目标位姿（真值，用于判定）"""
        t = self.task()
        if t["kind"] == "stack":
            name = f"stack_{t['target']}_L1_cargo"
        else:
            name = f"drop_zone_{t['target']}"
        p = self.body_pos(name)
        return np.array([p[0], p[1], self.body_yaw(name)])

    def _node_id(self, name, kind):
        """从路径图里按附属点 kind 与坐标找到节点号"""
        if kind == "pick":
            want = self._pick_goal()
        else:
            want = self._target_goal()
        best, best_d = -1, 1e9
        for pid, p in self.points.items():
            if not p.get("attached"):
                continue
            d = (p["x"] - want[0]) ** 2 + (p["y"] - want[1]) ** 2
            if d < best_d:
                best, best_d = pid, d
        return best

    def _nearest_node(self, x, y, lane_only=False):
        """离 (x,y) 最近的路径点"""
        best, best_d = -1, 1e9
        for pid in self.planner.connected_nodes:
            p = self.points[pid]
            if lane_only and p.get("attached"):
                continue
            d = (p["x"] - x) ** 2 + (p["y"] - y) ** 2
            if d < best_d:
                best, best_d = pid, d
        return best

    def _plan_route(self, goal, end_node):
        pos, _, _ = self.pose()
        start = self._nearest_node(pos[0], pos[1], lane_only=True)
        _approach = np.array([np.cos(goal[2]), np.sin(goal[2])])
        _pre = np.asarray(goal[:2], dtype=float) - PRE_ALIGN_BACK * _approach
        _end2 = self._nearest_node(_pre[0], _pre[1])
        if _end2 >= 0:
            end_node = _end2
        self.planner.plan(start, end_node)
        route = [np.array([wp["x"], wp["y"]], dtype=float)
                 for wp in self.planner.smoothed_path]
        # 去掉与终点重合的锚点节点，再补一个"预对准点"：
        g = np.array([goal[0], goal[1]], dtype=float)
        route = [p for p in route
                 if float(np.hypot(p[0] - g[0], p[1] - g[1])) > 0.05]
        approach = np.array([np.cos(goal[2]), np.sin(goal[2])])
        # 只有当最后一段"不是顺着进场方向进来"时才补预对准点：
        insert = True
        if route:
            tail = g - route[-1]
            n = float(np.hypot(tail[0], tail[1]))
            if n > 0.2 and abs(float(tail @ approach) / n) > 0.85:
                insert = False
        if insert:
            route.append(g - PRE_ALIGN_BACK * approach)
        route.append(g)
        if len(route) > 1 and np.hypot(route[0][0] - pos[0],
                                       route[0][1] - pos[1]) < 1.0:
            route.pop(0)
        return route

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset()
        self.fine_aligner.reset()
        self.judge = Judge(self.model, self.data, self.forklift_body_id)
        self.round = 1
        self.has_cargo = False
        self._begin_round()

    def _begin_round(self):
        self._redock_n = 0        # 本轮重进站次数
        t = self.task()
        self._pick_goal_true = self._pick_goal()
        self._drop_goal_true = self._target_goal()
        if self.use_guidance:
            self._guide_pick = self.guidance.sample(self._pick_goal_true)
            err = self.guidance.last_error()
            self._log(f"[引导] 取货点引导误差 {err[0] * 100:.1f}cm / "
                      f"{np.rad2deg(err[1]):.1f}°")
        else:
            self._guide_pick = self._pick_goal_true.copy()
        # 地面区没有"最下方货物"，按用户规则用精确坐标
        if t["kind"] == "stack" and self.use_guidance:
            self._guide_drop = self.guidance.sample(self._drop_goal_true)
            err = self.guidance.last_error()
            self._log(f"[引导] 放货点引导误差 {err[0] * 100:.1f}cm / "
                      f"{np.rad2deg(err[1]):.1f}°")
        else:
            self._guide_drop = self._drop_goal_true.copy()
            self._log("[引导] 地面放置区：使用精确坐标（无最下方货物）")

        self.pick_pallet = f"stack_{PICK_STACK}_L{t['pick_layer']}_pallet"
        self.pick_cargo = f"stack_{PICK_STACK}_L{t['pick_layer']}_cargo"
        self._dock_retry = 0
        self._target_T = self._current_target_top()
        self._log(f"\n===== 第 {self.round} 轮：取 {PICK_STACK} 第 "
                  f"{t['pick_layer']} 层 -> 放到 {t['target']}"
                  f"（承托面 {self._target_T:.3f}m）=====")
        self._start_nav(self._pick_goal_true, self._guide_pick,
                        self._node_id(PICK_STACK, "pick"), "pick")
        # 第 2 轮起，取货返程改为"先直退 2.5m 再交给 ppo_2"（用户方案）
        if self.round >= 2:
            self._pick_backout_left = PICK_BACKOUT_M
            self._pick_backout_start = None
        self._enter(0)
        # ---- 本轮判定事件基线 + 取货失败标记 ----
        self._ev_round_start = (len(self.judge.events["topple"]),
                                len(self.judge.events["vehicle_collision"]),
                                len(self.judge.events["cargo_collision"]))
        self._round_failed = False

    def _end_round(self, failed=False):
        """每轮收尾：打印机读结果行 → 进下一轮 / 收工"""
        pe = self.judge.report()["place_error"]
        n = (len(self.judge.events["topple"]),
             len(self.judge.events["vehicle_collision"]),
             len(self.judge.events["cargo_collision"]))
        n0 = getattr(self, "_ev_round_start", (0, 0, 0))
        dcol = sum(a - b for a, b in zip(n, n0))
        pos, yaw = pe.get("pos_m"), pe.get("yaw_deg")
        ok = int((not failed) and (not getattr(self, "_round_failed", False))
                 and dcol == 0 and bool(pe.get("ok")))
        self._log(
            f"[RESULT] round={self.round} ok={ok} "
            f"pos_err={'nan' if pos is None else round(float(pos), 4)} "
            f"yaw_err={'nan' if yaw is None else round(float(yaw), 3)} "
            f"coll={dcol} ticks={getattr(self, '_total_ticks', 0)} "
            f"phys={getattr(self, '_phys_steps', 0)}")
        self._ev_round_start = n
        self._round_failed = False
        self.round += 1
        if self.round <= self.total_rounds:
            self._log(f"→ 进入第 {self.round} 轮")
            self._begin_round()
            return
        self._log(f"[收尾] 3 轮作业结束（第 {self.total_rounds} 轮放货完成、"
                  f"已退到位并放下货叉）")
        self._enter(12)

    def _current_target_top(self):
        """目标堆当前顶面（放货前测量；地面区为 0）"""
        t = self.task()
        if t["kind"] != "stack":
            return 0.0
        top = 0.0
        for i in range(1, 4):
            name = f"stack_{t['target']}_L{i}_cargo"
            try:
                self._body_id(name)
            except ValueError:
                break
            _, hi = self.body_z_range(name)
            top = max(top, hi)
        return float(top)

    def _start_nav(self, true_goal, guide_goal, node_id, kind):
        self.current_goal = np.asarray(true_goal, dtype=float)
        self.guidance_goal = np.asarray(guide_goal, dtype=float)
        self.nav_kind = kind
        self._nav_node = node_id
        self._docking = False
        self._dock_mode = "ppo3"
        self._dock_ticks = 0
        self._dock_hit0 = None
        self._handoff_back = False
        self._handoff_retry = 0
        self._turnin_ticks = 0
        self._turnin_phase = "turn"
        self._fallback_active = False
        self._rev_ticks = 0
        self.dock_aligner.reset()
        # 图像处理的目标货堆：取货=取货堆，放货=目标堆，地面区=不做视觉
        if self.masked_cam is not None:
            t = self.task()
            if kind == "pick":
                self.masked_cam.set_target(f"stack_{PICK_STACK}_")
            elif kind == "stack":
                self.masked_cam.set_target(f"stack_{t['target']}_")
            else:
                self.masked_cam.set_target(None)
        self._route_i = 0
        self._ground_backin = False
        if kind in ("stack", "ground"):
            self._decisions = 0
            self.route, note = self._plan_drop_route(self.guidance_goal, kind)
            p0, _, yaw0 = self.pose()
            if kind == "ground":
                pass
            else:
                self._drop_phase = "follow"
            self._log(f"[导航] 放货腿折线 {len(self.route)} 点：{note}"
                      if self._drop_phase != "uturn" else
                      f"[导航] 放货腿：{note}（掉头 {len(self.route)} 点）")
            self._log("[导航] " + " ".join(
                f"({p[0]:+.2f},{p[1]:+.2f})" for p in self.route[:3]) +
                " ... " + " ".join(
                f"({p[0]:+.2f},{p[1]:+.2f})" for p in self.route[-3:]))
            return
        self._needs_align = False
        self._align_phase = not self._needs_align
        g = self.guidance_goal
        self.pre_dock = np.array([g[0] - PRE_DOCK_BACK_M * np.cos(g[2]),
                                  g[1] - PRE_DOCK_BACK_M * np.sin(g[2]),
                                  g[2]])
        dest = self.current_goal if self._align_phase else self.pre_dock
        self.route = self._plan_route(dest, node_id)
        self.route = self._trim_route_head(self.route)
        self.route = self._prepend_clear_waypoint(self.route)
        if self.route:
            self._log(f"[导航] 折线首点 ({self.route[0][0]:.2f},{self.route[0][1]:.2f}) 末点 ({self.route[-1][0]:.2f},{self.route[-1][1]:.2f}) 共 {len(self.route)} 点，目标 ({dest[0]:.2f},{dest[1]:.2f})")
        self._decisions = 0
        if self._needs_align:
            self._log(f"[导航] 真值目标 {np.round(self.current_goal[:2], 2)} "
                      f"引导目标 {np.round(g[:2], 2)} "
                      f"-> 预停点 ({self.pre_dock[0]:.2f},{self.pre_dock[1]:.2f}) "
                      f"节点#{node_id} 折线 {len(self.route)} 点"
                      f"（先精确导航+停稳，再交视觉对准）")
        else:
            self._log(f"[导航] 目标 {np.round(self.current_goal[:2], 2)} "
                      f"节点#{node_id} 折线 {len(self.route)} 点（地面区：全程精确坐标）")

    def _enter(self, state):
        self.state = state
        self._state_timer = 0
        self._release_done = False
        self._insert_armed = False      # 重新进入状态时重置插货保护基线
        self._drop_armed = False        # 放货期间的"正在操作"登记
        self._fork_home_done = False    # 状态 11 收尾：货叉是否已回到行驶高度
        self._fork_home_started = False
        pos, _, _ = self.pose()
        self._leg_start = pos.copy()

    def _moved(self):
        pos, _, _ = self.pose()
        return float(np.hypot(pos[0] - self._leg_start[0],
                              pos[1] - self._leg_start[1]))

    def _sync_view(self):
        """把窗口刷新插到物理步里，而不是每个决策才刷一次——决策一次要跑"""
        v = getattr(self, "viewer", None)
        if v is None:
            return
        try:
            if not v.is_running():
                return
            v.sync()
        except Exception:
            pass

    def _drive(self, v_cmd, steer_cmd, max_speed, steps=TICK_STEPS):
        self.last_action = (float(v_cmd), float(steer_cmd), float(max_speed))
        self.controller.set_fork_lock(self.controller.fork_at_target())
        # 模型段（ppo_2/ppo_3）必须用与训练一致的执行器特性：它们是在"把目标
        self.controller.steer_servo = False
        # 货叉也和训练脚本一样：每步直接写 ctrl[4]=0.4(载货)/-0.23(空载)。
        lift_raw = LIFT_RAW_CARGO if self.has_cargo else LIFT_RAW_EMPTY
        # ---- 放货腿的接近高度改成**本轮 drop_h** ----
        if self.has_cargo and self.nav_kind in ("stack", "ground"):
            try:
                lift_raw = float(self.task()["drop_h"])
            except Exception:                                     # noqa: BLE001
                lift_raw = LIFT_RAW_CARGO_HIGH
        for i in range(steps):
            self.controller.apply_control(v_cmd, steer_cmd, max_speed,
                                          self.has_cargo)
            self.data.ctrl[4] = lift_raw
            mujoco.mj_step(self.model, self.data)
            self._phys_steps = getattr(self, '_phys_steps', 0) + 1
            self.judge.step(self.dt)
            if VIEW_SYNC_EVERY and i % VIEW_SYNC_EVERY == VIEW_SYNC_EVERY - 1:
                self._sync_view()
        self._state_timer += steps
        vis = getattr(self, "vis", None)
        if vis is not None:
            vis.capture()

    def _hold(self, steps=TICK_STEPS, fork_hold=True):
        """保持转向、零速刹停"""
        self._drive_creep(0.0, self.controller.current_steer, steps,
                          fork_hold=fork_hold)

    def _set_fork(self, h):
        """设定货叉目标高度（要升降时必须先解锁，否则大阻尼会挡住升降）"""
        self.controller.set_fork_lock(False)
        self.controller.set_fork_target_height(h)

    def _fork_reached(self, h, tol=0.02):
        return abs(self.fork_z() - h) < tol

    def _fork_ray_up(self):
        R = self.data.xmat[self.forklift_body_id].reshape(3, 3)
        ray_dir = R @ np.array([0.0, 0.0, 1.0])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        geomid_out = np.zeros(1, dtype=np.int32)
        left_hit = right_hit = False
        for site_name in ("lidar_fork_left", "lidar_fork_right"):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                    site_name)
            if sid < 0:
                continue
            pnt = self.data.site_xpos[sid].copy()
            d = mujoco.mj_ray(self.model, self.data, pnt, ray_dir, None, False,
                              self.forklift_body_id, geomid_out)
            if 0.0 < d < 0.5:
                if "left" in site_name:
                    left_hit = True
                else:
                    right_hit = True
        if left_hit and right_hit:
            self._log("[插货] 叉齿上表面左右都打到托盘 → 停止前移")
            return True
        return False

    def _insert_diag(self):
        """插货中止时的诊断：接触对 + 车/叉齿到货堆的距离"""
        try:
            pairs = set()
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                b1 = int(self.model.geom_bodyid[c.geom1])
                b2 = int(self.model.geom_bodyid[c.geom2])
                n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b1) or "?"
                n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b2) or "?"
                pairs.add(f"{n1}<->{n2}")
            sp = self.body_pos(f"stack_{PICK_STACK}_L1_cargo")
            pos, _, _ = self.pose()
            fork = self.data.body(self.forks_body_id).xpos
            self._log(f"[插货诊断] 接触({len(pairs)}): " + "; ".join(sorted(pairs)[:6]))
            self._log(f"[插货诊断] 车距货堆中心 {float(np.hypot(pos[0]-sp[0], pos[1]-sp[1])):.2f}m，"
                      f"叉齿中心 ({fork[0]:.2f},{fork[1]:.2f},{fork[2]:.2f})，"
                      f"货堆中心 ({sp[0]:.2f},{sp[1]:.2f})")
        except Exception as e:
            self._log(f"[插货诊断] 失败: {e}")

    def _pick_stack_bodies(self):
        bodies = [self._body_id(self.pick_pallet),
                  self._body_id(self.pick_cargo)]
        for i in range(1, 4):
            for k in ("pallet", "cargo"):
                try:
                    bodies.append(self._body_id(f"stack_{PICK_STACK}_L{i}_{k}"))
                except ValueError:
                    pass
        return bodies

    def _target_stack_bodies(self):
        bodies = [self._body_id(self.pick_pallet),
                  self._body_id(self.pick_cargo)]
        t = self.task()
        if t.get("kind") == "stack":
            for i in range(1, 4):
                for k in ("pallet", "cargo"):
                    try:
                        bodies.append(
                            self._body_id(f"stack_{t['target']}_L{i}_{k}"))
                    except ValueError:
                        pass
        return bodies

    def _load_on_fork(self):
        """被取的托盘是否已经落在货叉上"""
        pal = self.body_pos(self.pick_pallet)
        fork = self.data.body(self.forks_body_id).xpos
        dxy = np.hypot(pal[0] - fork[0], pal[1] - fork[1])
        return dxy < 0.8 and abs(self.body_low_z(self.pick_pallet)
                                 - self.fork_z()) < 0.20

    def _pallet_fork_offset(self):
        """托盘相对**货叉**的偏移（货叉坐标系，m）：[0]=纵向(朝车头)"""
        try:
            pal = np.asarray(self.data.body(self._body_id(
                self.pick_pallet)).xpos, dtype=float)
            fk = np.asarray(self.data.body(self.forks_body_id).xpos, dtype=float)
            R = self.data.xmat[self.forks_body_id].reshape(3, 3)
            d = R.T @ (pal - fk)
            return float(d[0]), float(d[1]), float(d[2])
        except Exception:                                     # noqa: BLE001
            return None

    def body_z_range(self, name):
        bid = self._body_id(name)
        lo, hi = np.inf, -np.inf
        adr = self.model.body_geomadr[bid]
        for gid in range(adr, adr + self.model.body_geomnum[bid]):
            R = self.data.geom_xmat[gid].reshape(3, 3)
            size = self.model.geom_size[gid]
            ext = (float(np.abs(R[2, :]).dot(size))
                   if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
                   else float(self.model.geom_rbound[gid]))
            lo = min(lo, float(self.data.geom_xpos[gid][2]) - ext)
            hi = max(hi, float(self.data.geom_xpos[gid][2]) + ext)
        return float(lo), float(hi)

    def _lookahead(self, pos, dist=None):
        _d = float(LOOKAHEAD_M if dist is None else dist)
        r = self.route
        if not r:
            return None
        while len(r) > 1 and np.hypot(r[0][0] - pos[0], r[0][1] - pos[1]) < 0.8:
            r.pop(0)
        acc = 0.0
        prev = np.array([pos[0], pos[1]])
        for p in r:
            seg = float(np.hypot(p[0] - prev[0], p[1] - prev[1]))
            if acc + seg >= _d:
                k = (_d - acc) / max(seg, 1e-6)
                return prev + k * (p - prev)
            acc += seg
            prev = p
        return r[-1]

    def _route_deviation(self, pos):
        """车体到折线的最短距离（只用当前位置与规划折线，不读真值）"""
        if not self.route:
            return 0.0
        p = np.array([pos[0], pos[1]], dtype=float)
        best = 1e9
        prev = None
        for q in self.route:
            if prev is not None:
                seg = q - prev
                L2 = float(seg @ seg)
                if L2 > 1e-9:
                    t = float(np.clip((p - prev) @ seg / L2, 0.0, 1.0))
                    foot = prev + t * seg
                    best = min(best, float(np.hypot(*(p - foot))))
            prev = q
        return 0.0 if best > 1e8 else float(best)

    def _center_steer(self, ticks=90):
        """把后轮**真正**回正到 0（第 3 轮取货撞货的根因就在这里）"""
        self.controller.steer_servo = True
        for _ in range(int(ticks)):
            self.controller.apply_creep(0.0, 0.0)
            mujoco.mj_step(self.model, self.data)
            self._phys_steps = getattr(self, '_phys_steps', 0) + 1
            self.judge.step(self.dt)
        for _ in range(4000):
            q = max(abs(float(self.data.qpos[qadr]))
                    for qadr, _dadr in self.controller._steer_joints)
            # 回正判据从 0.02rad 放宽到 0.07rad(≈4°)：
            if abs(self.controller.current_steer) < 0.05 and q < 0.05:
                break
            self.controller.apply_creep(0.0, 0.0)
            mujoco.mj_step(self.model, self.data)
            self._phys_steps = getattr(self, '_phys_steps', 0) + 1
            self.judge.step(self.dt)
        q = max(abs(float(self.data.qpos[qadr]))
                for qadr, _dadr in self.controller._steer_joints)
        self._log(f"[转向] 回正完成：指令 {np.rad2deg(self.controller.current_steer):+.1f}°"
                  f" / 实际后轮 {np.rad2deg(q):+.1f}°")

    def _drive_creep(self, v_des, steer_cmd, steps=DOCK_STEPS, servo=True,
                     gain=None, fork_hold=False):
        """程序控制用的低速精确驱动（轮速 P 环，见 f_c.controller.apply_creep）"""
        self.last_action = (float(v_des), float(steer_cmd), 0.0)
        self.controller.steer_servo = bool(servo)
        # 载货时轮速环增益加大：带载的阻力矩让 P 环静差 60%，车只能爬到
        self.controller.creep_gain = (float(gain) if gain is not None
                                      else (1.0 if self.has_cargo else 1.0))
        for i in range(steps):
            # 带加速度上限：0.45m/s 的直线退出也能做到"起步不顿"
            self.controller.apply_creep(v_des, steer_cmd,
                                        accel_limit=PROG_ACCEL_LIMIT,
                                        fork_hold=fork_hold)
            mujoco.mj_step(self.model, self.data)
            self._phys_steps = getattr(self, '_phys_steps', 0) + 1
            self.judge.step(self.dt)
            if VIEW_SYNC_EVERY and i % VIEW_SYNC_EVERY == VIEW_SYNC_EVERY - 1:
                self._sync_view()
        self._state_timer += steps
        vis = getattr(self, "vis", None)
        if vis is not None:
            vis.capture()

    def _anchor_errors(self):
        """自车相对引导锚点的 (纵向, 横向) 距离；纵向 >0 = 冲过了锚点"""
        pos, _, yaw = self.pose()
        g = self.guidance_goal
        apsi = float(g[2])
        dx = float(pos[0] - g[0])
        dy = float(pos[1] - g[1])
        along = dx * float(np.cos(apsi)) + dy * float(np.sin(apsi))
        lat = -dx * float(np.sin(apsi)) + dy * float(np.cos(apsi))
        return float(along), float(lat)

    def _anchor_along(self):
        return self._anchor_errors()[0]

    def _guide_errors(self):
        """自车相对引导锚点的 (纵向, 横向, 车头偏差)：只用引导位姿 + 自车里程"""
        pos, _, yaw = self.pose()
        g = self.guidance_goal
        dx, dy = float(pos[0] - g[0]), float(pos[1] - g[1])
        ca, sa = float(np.cos(g[2])), float(np.sin(g[2]))
        head = float(np.arctan2(np.sin(g[2] - yaw), np.cos(g[2] - yaw)))
        return float(dx * ca + dy * sa), float(-dx * sa + dy * ca), head

    def _dock_tick(self):
        """末段对准总入口：**全程 ppo_3**（用户最高优先级要求：程序兜底全删）"""
        self._dock_ticks += 1
        self._dock_ppo3_tick()

    def _arrived_by_truth_gate(self):
        """按真值门限判断是否已经到位（判定与终止一律用真值）"""
        pos, _, yaw = self.pose()
        e_x, e_y, e_theta = body_frame_goal(pos, yaw, self.current_goal[:2],
                                            self.current_goal[2])
        dist = float(np.hypot(e_x, e_y))
        speed = abs(self.speed())
        _yaw_tol = (FINAL_YAW_TOL if self.nav_kind == "pick"
                    else FINAL_YAW_TOL_DROP)
        if dist <= FINAL_POS_TOL and abs(e_theta) <= _yaw_tol:
            self._log(f"[末段对准] ppo_3 到位（真值误差 {dist * 100:.1f}cm / "
                      f"{np.rad2deg(abs(e_theta)):.2f}° / {speed:.3f}m/s）")
            return True
        return False

    def _dock_ppo3_tick(self):
        """用 ppo_3 做末段视觉对准"""
        if self._arrived_by_truth_gate():
            self._finish_align()
            return
        _pos, _, _yaw = self.pose()
        _ex, _ey, _et = body_frame_goal(_pos, _yaw, self.current_goal[:2],
                                        self.current_goal[2])
        _metric = float(np.hypot(_ex, _ey)) + 0.35 * abs(float(_et))
        if self._dock_ticks <= 2:            # 每进入一轮新对准就重置检测器
            self._dock_best, self._dock_stall = 1e9, 0
        if getattr(self, "_dock_best", 1e9) - _metric > 0.002:
            self._dock_best = _metric
            self._dock_stall = 0
        else:
            self._dock_stall = getattr(self, "_dock_stall", 0) + 1
            if self._dock_stall >= DOCK_STALL_TICKS:
                self._dock_stall = 0
                self._dock_best = 1e9
                self._log(f"[交接] ppo_3 卡住（{DOCK_STALL_TICKS} 次无进展，"
                          f"当前 {_metric * 100:.1f}cm 等效）"
                          f"→ 直接进入下一次对准")
                self._finish_align()
                return
        # 决策预算：第 1 轮 500 步，之后每轮 300 步（用户指定）
        _cap = (PPO3_DECISIONS_FIRST
                if getattr(self, "_align_check_retry", 0) == 0
                else PPO3_DECISIONS_RETRY)
        if self._dock_ticks > _cap:
            self._log(f"[交接] ppo_3 决策超限（{self._dock_ticks} > {_cap}）"
                      f"→ 复核/后退重来")
            self._finish_align()
            return
        image = None
        if self.masked_cam is not None:
            image = self.masked_cam.render(self.data, masked=True)
        obs = build_obs(self.model, self.data, self.guidance_goal,
                        self.forklift_body_id, self.controller,
                        self.root_dof_adr, self.mast_body_id,
                        self.forks_body_id, mode=1, renderer=self.renderer,
                        image=image)
        act, _ = self.align_model.predict(obs, deterministic=True)
        self.last_obs, self.last_mode = obs, 1
        self.last_model = "ppo_3(视觉对准·图像已处理)"
        v_cmd = self._cap_forward(float(act[0]))
        if self._dock_ticks % 40 == 0:
            st = self.masked_cam.masker.last_stats if self.masked_cam else {}
            self._log(f"  [ppo3] t={self._dock_ticks} "
                      f"保留像素={st.get('keep', 0.0):.3f} "
                      f"动作=({float(act[0]):+.2f},{float(act[1]):+.2f})")
        self._drive(v_cmd, float(act[1]) * self.controller.MAX_STEER,
                    FINAL_MAX_SPEED)

    def _aligned_now(self):
        if not ALIGN_CHECK_ENABLE or self.renderer is None:
            return True, "关闭校验/无渲染器"
        try:
            self.renderer.update_scene(self.data, camera="mast_cam")
            img_raw = np.transpose(self.renderer.render(), (2, 0, 1))
            # ---- 横向复核改用**只留目标货堆**的掩码图 ----
            if (self.masked_cam is not None
                    and self.masked_cam.masker.keep_geoms):
                img = self.masked_cam.render(self.data, masked=True)
            else:
                img = img_raw
            lidar = build_lidar_scan(self.model, self.data,
                                     self.forklift_body_id, self.mast_body_id,
                                     self.forks_body_id)
            fwd = np.asarray(lidar[0:20], dtype=float) * 8.0
            psi, d_perp, _npts = face_fit(fwd)
            x_dist = 1.25 + (0.0 if d_perp is None
                             else d_perp - DOCK_STANDOFF_M)
            long_err = (None if d_perp is None
                        else float(d_perp) - float(DOCK_STANDOFF_M))
            lat, _npix, _cols, _clip = stack_lateral_error(img, x_dist)
            lat_raw, _n2, _c2, _cl2 = stack_lateral_error(img_raw, x_dist)
            _l = "None" if lat is None else f"{lat:+.3f}"
            _lr = "None" if lat_raw is None else f"{lat_raw:+.3f}"
            _p = "None" if psi is None else f"{np.rad2deg(psi):+.2f}°"
            self._log(f"  [复核诊断] 掩码图横向 {_l} / 原图横向 {_lr} / "
                      f"雷达法向 {_p}")
        except Exception as e:                                # noqa: BLE001
            return True, f"校验异常({e})"
        bad = []
        if lat is None:
            bad.append("相机看不到货物（横向不可测）")
        if psi is None:
            bad.append("雷达拟合不出货面（法向不可测）")
        if lat is not None and abs(lat) > ALIGN_CHECK_LAT_TOL:
            bad.append(f"图像横向 {lat:+.2f}（容差 {ALIGN_CHECK_LAT_TOL:.2f}）")
        if psi is not None and abs(psi) > ALIGN_CHECK_SYM_TOL:
            bad.append(f"雷达法向 {np.rad2deg(psi):+.1f}°"
                       f"（容差 {np.rad2deg(ALIGN_CHECK_SYM_TOL):.1f}°）")
        if (long_err is not None
                and abs(long_err) > ALIGN_CHECK_LONG_TOL):
            bad.append(f"纵向站位 {long_err:+.2f}m"
                       f"（容差 {ALIGN_CHECK_LONG_TOL:.2f}m）")
        if bad:
            return False, "；".join(bad)
        return True, (f"图像横向 {(0.0 if lat is None else lat):+.2f}、"
                      f"雷达法向 "
                      f"{(0.0 if psi is None else np.rad2deg(psi)):+.1f}°")

    def _radar_probe(self):
        """→ (门架→载货后表面, 门架→货堆正面)，测不到给 None"""
        try:
            _R = self.data.xmat[self.forklift_body_id].reshape(3, 3)
            _ax = np.asarray(_R[:, 0], dtype=float)
            _own = {int(self.forklift_body_id)}
            for _n in ("pick_pallet", "pick_cargo"):
                _b = getattr(self, _n, None)
                if _b:
                    try:
                        _own.add(int(self._body_id(_b)))
                    except Exception:                          # noqa: BLE001
                        pass
            _load, _stack = [], []
            for _z in ("z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8"):
                for _suf in ("", "_l", "_r"):
                    _nm = f"radar_load_{_z}{_suf}"
                    _sid = mujoco.mj_name2id(self.model,
                                             mujoco.mjtObj.mjOBJ_SITE, _nm)
                    if _sid < 0:
                        continue
                    _p = self.data.site_xpos[_sid].copy()
                    _out = np.zeros(1, dtype=np.int32)
                    _d = mujoco.mj_ray(self.model, self.data, _p, _ax, None,
                                       False, self.forklift_body_id, _out)
                    if _d is None or _d <= 0.005 or _d > 8.0:
                        continue
                    _hit = int(self.model.geom_bodyid[_out[0]])
                    if _hit in _own:
                        if _hit != int(self.forklift_body_id):
                            _load.append(float(_d))
                    else:
                        _stack.append(float(_d))
            return (float(np.median(_load)) if _load else None,
                    float(np.median(_stack)) if _stack else None)
        except Exception:                                     # noqa: BLE001
            return None, None

    def _load_rear_dist(self):
        return self._radar_probe()[0]

    def _stack_face_dist(self):
        return self._radar_probe()[1]

    def _back_off_straight(self, dist):
        """程序**直线后退**：轮胎保持回正、货叉保持不动（fork_hold）"""
        _start = self.pose()[0].copy()
        for _ in range(4000):
            self._drive_creep(-0.40, 0.0, fork_hold=True)
            if self._moved_from(_start) > dist:
                break
        self._center_steer()
        self._log(f"[对准校验] 已直线后退 {self._moved_from(_start):.2f}m"
                  f"（车 {self.pose()[0][0]:+.2f},{self.pose()[0][1]:+.2f}）")

    def _finish_align(self):
        """末段对准完成后的**统一收尾**（取货/放货共用）"""
        self._center_steer()
        ok, why = self._aligned_now()
        n = getattr(self, "_align_check_retry", 0)
        if not ok and n < ALIGN_CHECK_RETRY_MAX:
            self._align_check_retry = n + 1
            self._log(f"[对准校验] 第 {n + 1} 次复核：回正后没对正（{why}）"
                      f"→ 程序直线后退 {ALIGN_BACKOFF_M:.1f}m，再交 ppo_3")
            self._back_off_straight(ALIGN_BACKOFF_M)
            self._dock_ticks = 0
            self._back1_done = True     # 不让旧的"后退 1m"逻辑再插一脚
            return True
        if not ok and getattr(self, "_redock_n", 0) < REDOCK2_MAX:
            self._redock_round = int(self.round)
            self._redock_n = getattr(self, "_redock_n", 0) + 1
            self._align_check_retry = 0
            self._log("[重进站] 复核仍没对正 → 退回路线重新进站一次"
                      "（不再直接硬插）")
            try:
                self._start_nav(self.current_goal, self.guidance_goal,
                                getattr(self, "_nav_node", None), self.nav_kind)
            except Exception:                                 # noqa: BLE001
                pass
            return True
        if not ok:
            _p, _z, _y = self.pose()
            _d = float(np.hypot(float(self.current_goal[0]) - _p[0],
                                float(self.current_goal[1]) - _p[1]))
            if _d > REDOCK2_NEAR_M:
                self._log(f"[对准校验] 仍未对正且距目标 {_d:.2f}m"
                          f"（>{REDOCK2_NEAR_M:.2f}m）→ 不硬插，退回路线重进站")
                try:
                    self._start_nav(self.current_goal, self.guidance_goal,
                                    getattr(self, "_nav_node", None),
                                    self.nav_kind)
                except Exception:                     # noqa: BLE001
                    pass
                return True
        _tag = "已对正" if ok else "仍未对正"
        self._log(f"[对准校验] {_tag}（{why}）→ 进入插货/放货")
        self._hold(30)
        self._enter(1 if self.nav_kind == "pick" else 8)
        return False

    def _handoff_turn_tick(self, dist_g, head_g):
        if abs(head_g) <= np.deg2rad(HANDOFF_YAW_OK_DEG):
            self._log(f"[交接] 车头偏差已收到 {np.rad2deg(head_g):+.0f}°"
                      f"（≤{HANDOFF_YAW_OK_DEG:.0f}°）→ 交 ppo_3")
            self._enter_dock()
            return True
        self._handoff_turn = getattr(self, "_handoff_turn", 0) + 1
        if self._handoff_turn % 200 == 0:
            self._log(f"  [交接转向] t={self._handoff_turn} 车头偏差 "
                      f"{np.rad2deg(head_g):+.0f}° 距锚点 {dist_g:.2f}m "
                      f"车({self.pose()[0][0]:+.2f},{self.pose()[0][1]:+.2f})")
        if self._handoff_turn > HANDOFF_TURN_MAX_TICKS:
            self._log(f"[交接] 转向超时（{self._handoff_turn} tick）→ "
                      f"按当前姿态交 ppo_3")
            self._enter_dock()
            return True
        # 程序控制转弯：航向比例打舵 + 低速前进（舵角伺服，半径 ~1.5~1.9m）
        steer = float(np.clip(HANDOFF_TURN_K * head_g,
                              -self.controller.MAX_STEER,
                              self.controller.MAX_STEER))
        self.last_model = f"程序交接转弯(偏差{np.rad2deg(head_g):+.0f}°)"
        self._drive_creep(HANDOFF_TURN_V, steer)
        return False

    def _cap_forward(self, v_cmd):
        """末段过冲保护（相机 + 龙门架雷达实测几何）"""
        if v_cmd <= 0:
            return v_cmd
        along, _lat = self._anchor_errors()
        if along > DOCK_OVERSHOOT_M:
            if self._dock_ticks % 100 == 0:
                self._log(f"  [末段保护] 已冲过锚点 {along:+.2f}m "
                          f"（>{DOCK_OVERSHOOT_M:.2f}m）→ 禁止再前进")
            return 0.0
        gap = self._tip_gap_m()
        if gap is None:
            return v_cmd
        if gap <= TIP_MIN_GAP_M:
            if self._dock_ticks % 100 == 0:
                self._log(f"  [末段保护] 叉尖离货堆 {gap * 100:.1f}cm "
                          f"≤ {TIP_MIN_GAP_M * 100:.0f}cm → 禁止再前进")
            return 0.0
        if gap < TIP_SLOW_GAP_M:
            if v_cmd > TIP_SLOW_V:
                if self._dock_ticks % 100 == 0:
                    self._log(f"  [末段保护] 叉尖离货堆 {gap * 100:.1f}cm → "
                              f"限速 {TIP_SLOW_V:.2f}m/s")
                return TIP_SLOW_V
        return v_cmd

    def _tip_gap_m(self):
        """叉尖到货堆正面的净距 (m)；量不到返回 None"""
        lidar = build_lidar_scan(self.model, self.data, self.forklift_body_id,
                                 self.mast_body_id, self.forks_body_id)
        fwd = np.asarray(lidar[0:20], dtype=float) * 8.0
        psi, d_perp, _n = face_fit(fwd)
        self._tip_psi = psi
        if d_perp is None:
            d_perp = self._front_stack_dist()
        if d_perp is None:
            return None
        # 相机横向偏差（诊断：能看到目标却雷达没读数，说明雷达打到别处了）
        self._tip_cam_lat = None
        if self.renderer is not None:
            try:
                self.renderer.update_scene(self.data, camera="mast_cam")
                img = np.transpose(self.renderer.render(), (2, 0, 1))
                _lat, _npx, _c, _cl = stack_lateral_error(
                    img, max(0.5, float(d_perp)))
                self._tip_cam_lat = _lat
            except Exception:                                # noqa: BLE001
                pass
        return float(d_perp) - MAST_TO_TIP_M

    def _front_stack_dist(self, max_dist=8.0):
        """门架前向雷达打到**货堆正面**的距离（m）；看不到返回 None"""
        R = self.data.xmat[self.forklift_body_id].reshape(3, 3)
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                "lidar_mast")
        if sid < 0:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                    "sensor_mast")
        if sid < 0:
            return None
        origin0 = self.data.site_xpos[sid].copy()
        # 自家子树（叉车 + 门架 + 货叉 + 驮着的托盘/货物）的 body id 集合
        own = set()
        bid = self.forklift_body_id
        own.add(bid)
        for b in range(self.model.nbody):
            p = b
            while p > 0:
                p = int(self.model.body_parentid[p])
                if p == bid:
                    own.add(b)
                    break
        half_fov = np.deg2rad(45.0)
        rays = 20
        ds = []
        geomid = np.zeros(1, dtype=np.int32)
        for k in range(10 - 3, 10 + 3):
            ang = -half_fov + 2 * half_fov * (k / (rays - 1))
            vec = R @ np.array([np.cos(ang), np.sin(ang), 0.0])
            vec /= np.linalg.norm(vec)
            o = origin0.copy()
            for _ in range(5):                       # 跳过自家部件
                d = mujoco.mj_ray(self.model, self.data, o, vec, None,
                                  False, self.forklift_body_id, geomid)
                if d < 0 or d > max_dist:
                    break
                hit_body = int(self.model.geom_bodyid[geomid[0]])
                if hit_body not in own:
                    ds.append(float(np.linalg.norm(o - origin0)) + float(d))
                    break
                o = o + vec * (float(d) + 0.005)
            else:
                continue
        if not ds:
            return None
        return float(np.median(ds))

    def _pick_backout_tick(self):
        """取货返程第一步：程序直线后退 PICK_BACKOUT_M，再交给 ppo_2"""
        pos, _, yaw = self.pose()
        if getattr(self, "_pick_backout_start", None) is None:
            self._pick_backout_start = pos.copy()
            self._pick_backout_ticks = 0
            self._log(f"[返程] 先程序直线后退 {PICK_BACKOUT_M:.1f}m 离开货堆")
        self._pick_backout_ticks = getattr(self, "_pick_backout_ticks", 0) + 1
        moved = self._moved_from(self._pick_backout_start)
        _bx = pos[0] - 0.45 * float(np.cos(yaw))
        _by = pos[1] - 0.45 * float(np.sin(yaw))
        back_clear = self._foot_clear(_bx, _by, yaw)
        if moved >= PICK_BACKOUT_M or back_clear < 0.20 \
                or self._pick_backout_ticks > PICK_BACKOUT_MAX_TICKS:
            self._pick_backout_left = 0.0
            # 从当前位姿重新规划：`_plan_route` 取"路径图上离车最近的节点"起步
            self._replan_route("返程后退完成")
            self._log(f"[返程] 后退 {moved:.2f}m 完成（后方净空 "
                      f"{back_clear:.2f}m）→ 从最近路径点交给 ppo_2")
            return
        self.last_model = (f"返程后退({moved:.2f}/{PICK_BACKOUT_M:.1f}m "
                           f"净空{back_clear:.2f})")
        self._drive_creep(-PICK_BACKOUT_V, 0.0)

    def _nav_pick_tick(self):
        """取货腿导航：**ppo_2 沿折线前视驱动** → 进入交接窗口交 ppo_3"""
        pos, _, yaw = self.pose()
        g = self.guidance_goal
        e_x, e_y, _e_th = body_frame_goal(pos, yaw, g[:2], g[2])
        dist_g = float(np.hypot(e_x, e_y))
        # 返程起步（第 2 轮起）：先程序直线后退，再从最近路径点交给 ppo_2
        if getattr(self, "_pick_backout_left", 0.0) > 0.0 and not self._docking:
            self._pick_backout_tick()
            return
        # 看门狗只在"真的在走"的阶段生效：抬货叉（ground_lift）本来就是原地
        if not self._docking and getattr(self, "_drop_phase", "") \
                not in ("ground_lift",):
            self._stall_watchdog()
        if self._docking:
            self._dock_tick()
            return
        along, lat, head_g = self._guide_errors()
        look, rem, dev = self._path_follow(
            DROP_LOOK_SLOW if dist_g < 3.0 else DROP_LOOK)
        if look is None:
            self._hold(20)
            return
        # ===== 末段：走到折线末端附近就整段交给 ppo_3（含模糊目标点）=====
        if dist_g <= TURNIN_ENGAGE_M and (rem <= TURNIN_ENGAGE_REM_M
                                          or dist_g <= 1.60):
            # 到交接距离：车头偏差还大就用 ppo_2 + 模糊目标先转（不绕行），
            if self._handoff_turn_tick(dist_g, head_g):
                return
            return
        self._leg_speed(look, dist_g, rem)

    def _nav_tick(self):
        """导航总入口"""
        if self.nav_kind in ("stack", "ground"):
            self._nav_drop_tick()
            return
        if self.nav_kind == "pick":
            self._nav_pick_tick()
            return
        self._hold(20)

    def _dijkstra_nodes(self, start_id, end_id, banned=()):
        """路径图节点最短路（可禁用若干有向边，用来禁止开腿原地掉头）"""
        import heapq
        adj = {}

        def add(u, v):
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

        for e in self.planner.edges:
            u, v = int(e["from"]), int(e["to"])
            if (u, v) in banned or (v, u) in banned:
                continue
            if not self._edge_clear(u, v):
                continue
            add(u, v)
        # 妙妙工具生成的路径图漏掉了同一条车道上相邻点之间的连边
        ids = sorted(self.points)
        for i, u in enumerate(ids):
            for v in ids[i + 1:]:
                pu, pv = self.points[u], self.points[v]
                dx = abs(pu["x"] - pv["x"])
                dy = abs(pu["y"] - pv["y"])
                if (dx < 0.1 and 0.0 < dy <= 4.0) or \
                        (dy < 0.1 and 0.0 < dx <= 4.0):
                    if (u, v) in banned or (v, u) in banned:
                        continue
                    if not self._edge_clear(u, v):
                        continue
                    add(u, v)

        def seg(u, v):
            pu, pv = self.points[u], self.points[v]
            return float(np.hypot(pu["x"] - pv["x"], pu["y"] - pv["y"]))

        dist = {start_id: 0.0}
        prev = {}
        pq = [(0.0, start_id)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, 1e18):
                continue
            for v in adj.get(u, ()):
                if (u, v) in banned:
                    continue
                nd = d + seg(u, v)
                if nd < dist.get(v, 1e18):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if end_id != start_id and end_id not in prev:
            return []
        path = [end_id]
        while path[-1] != start_id:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def _is_inner_lane(self, pid):
        return abs(float(self.points[pid]["y"]) - INNER_LANE_Y) < 0.3

    def _stack_xy(self):
        """场景里所有货堆的中心（用于判断路径会不会蹭到货堆）"""
        if getattr(self, "_stack_xy_cache", None) is None:
            pts = []
            for i in range(self.model.nbody):
                nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if not nm or not nm.startswith("stack_") \
                        or not nm.endswith("_cargo"):
                    continue
                if "_L1_cargo" not in nm:
                    continue
                pts.append((float(self.data.xpos[i][0]),
                            float(self.data.xpos[i][1])))
            self._stack_xy_cache = pts
        return self._stack_xy_cache

    def _edge_clear(self, u, v):
        """这条边离货堆够远吗（本车宽 1.3m、托盘 0.8m，留 0.95m 净空）"""
        pu, pv = self.points[u], self.points[v]
        a = np.array([pu["x"], pu["y"]], float)
        b = np.array([pv["x"], pv["y"]], float)
        ab = b - a
        L2 = float(ab @ ab)
        for sx, sy in self._stack_xy():
            p = np.array([sx, sy], float)
            t = 0.0 if L2 < 1e-9 else float(np.clip((p - a) @ ab / L2, 0, 1))
            foot = a + t * ab
            if float(np.hypot(p[0] - foot[0], p[1] - foot[1])) \
                    < DROP_EDGE_CLEAR_M:
                return False
        return True

    def _forward_start_node(self, pos, yaw):
        h = np.array([np.cos(yaw), np.sin(yaw)])
        best, best_score = -1, 1e18
        for pid in self.planner.connected_nodes:
            p = self.points[pid]
            d = np.array([p["x"] - pos[0], p["y"] - pos[1]])
            dist = float(np.hypot(d[0], d[1]))
            if dist < 1e-6:
                return pid
            ahead = float(d @ h) / dist
            # 车后的节点加很大惩罚：宁可多走一段环线，也不掉头
            penalty = 0.0 if ahead > -0.2 else (1.0 - ahead) * 30.0
            if dist + penalty < best_score:
                best, best_score = pid, dist + penalty
        return best

    def _ring_route(self, goal_xy):
        pos, _, yaw = self.pose()
        start = self._forward_start_node(pos, yaw)
        end, best_d = -1, 1e18
        for pid in self.planner.connected_nodes:
            if self._is_inner_lane(pid):
                continue
            p = self.points[pid]
            d = (p["x"] - goal_xy[0]) ** 2 + (p["y"] - goal_xy[1]) ** 2
            if d < best_d:
                end, best_d = pid, d
        h = np.array([np.cos(yaw), np.sin(yaw)])
        banned = set()
        for e in self.planner.edges:
            u, v = int(e["from"]), int(e["to"])
            for a_, b_ in ((u, v), (v, u)):
                if a_ != start:
                    continue
                d = np.array([self.points[b_]["x"] - self.points[a_]["x"],
                              self.points[b_]["y"] - self.points[a_]["y"]])
                nn = float(np.hypot(d[0], d[1]))
                if nn > 1e-6 and float(d @ h) / nn < -0.2:
                    banned.add((a_, b_))
        ids = self._dijkstra_nodes(start, end, banned)
        pts = [np.array([self.points[i]["x"], self.points[i]["y"]], float)
               for i in ids]
        return pts

    def _plan_drop_route(self, g, kind):
        if kind == "ground":
            return self._plan_ground_route(g)
        g = np.asarray(g, dtype=float)
        pos, _, yaw = self.pose()
        start = self._forward_start_node(pos, yaw)
        # 交接点选择：A/B 排的附属点就在锚点上，如果让 ppo_2 直接开到锚点，
        cands = []
        for pid in self.planner.connected_nodes:
            p = np.array([self.points[pid]["x"], self.points[pid]["y"]])
            dd = float(np.hypot(p[0] - g[0], p[1] - g[1]))
            if dd < DOCK_MIN_APPROACH_M or dd > DOCK_MAX_APPROACH_M:
                continue
            ids = self._dijkstra_nodes(start, pid)
            if len(ids) < 2:
                continue
            q = self.points[ids[-2]]
            arr = float(np.arctan2(p[1] - float(q["y"]), p[0] - float(q["x"])))
            to_goal = float(np.arctan2(g[1] - p[1], g[0] - p[0]))
            mis = abs(float(np.arctan2(np.sin(arr - to_goal),
                                       np.cos(arr - to_goal))))
            cands.append((dd + 2.5 * mis, mis, dd, pid, ids))
        if not cands:
            return [], "（规划失败：没有可用路径点）"
        _score, _mis, _dd, end, ids = min(cands, key=lambda c: c[0])
        route = [np.array([self.points[i]["x"], self.points[i]["y"]], float)
                 for i in ids]
        # 去掉离车太近的前几个点，避免原地画圈
        while len(route) > 1 and float(np.hypot(route[0][0] - pos[0],
                                                route[0][1] - pos[1])) < 1.2:
            route.pop(0)
        clear = path_clearance(route + [g[:2]], self.obstacle_boxes) \
            if len(route) else float("inf")
        note = (f"路径图 {len(route)} 点 → 终点 "
                f"({route[-1][0]:+.2f},{route[-1][1]:+.2f})"
                f"（离锚点 {float(np.hypot(route[-1][0] - g[0], route[-1][1] - g[1])):.2f}m，"
                f"足迹净空 {clear:.2f}m）")
        return route, note

    def _plan_ground_route(self, g):
        """地面区（第 3 轮）：退开 → 抬货叉 → 直追锚点掉头 → 位姿闭环进站"""
        self._drop_phase = "ground_back"
        self._uturn_ticks = 0
        return [], "地面区：退开+抬货叉+直追锚点掉头+位姿闭环进站"

    def _foot_clear(self, x, y, yaw):
        return foot_clearance(float(x), float(y), float(yaw),
                              self.obstacle_boxes)

    def _safety_guard(self, v_des, steer_cmd, margin=None):
        m = NAV_SAFE_MARGIN if margin is None else float(margin)
        if abs(v_des) < 1e-6:
            return v_des, steer_cmd
        pos, _, yaw = self.pose()
        d0 = self._foot_clear(pos[0], pos[1], yaw)
        step = min(0.20, max(0.05, abs(v_des) * 0.40))
        sgn = 1.0 if v_des > 0 else -1.0
        px = float(pos[0]) + sgn * step * float(np.cos(yaw))
        py = float(pos[1]) + sgn * step * float(np.sin(yaw))
        d1 = self._foot_clear(px, py, yaw)
        if d1 >= m:
            return v_des, steer_cmd
        if d1 > d0 + 0.002 and d1 > 0.03:
            # 正在远离障碍：允许低速通过（脱困、贴边微调都要靠它）
            lim = min(abs(v_des), 0.10)
            return (lim if v_des > 0 else -lim), steer_cmd
        self.last_model = (f"安全阀(净空{d0:.2f}→{d1:.2f}m 停车)")
        return 0.0, steer_cmd

    def _path_follow(self, look_dist):
        """折线跟踪：返回 (前视点, 剩余里程, 横向偏差)"""
        r = self.route
        n = len(r)
        if n == 0:
            return None, 0.0, 0.0
        pos, _, _ = self.pose()
        p = np.array([pos[0], pos[1]], float)
        if n == 1:
            return r[0], float(np.hypot(*(r[0] - p))), 0.0
        i0 = int(np.clip(getattr(self, "_route_i", 0), 0, n - 2))
        best = [1e18, i0, 0.0, p]
        for i in range(max(0, i0 - 2), n - 1):
            a, b = r[i], r[i + 1]
            ab = b - a
            L2 = float(ab @ ab)
            t = 0.0 if L2 < 1e-9 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
            foot = a + t * ab
            dist = float(np.hypot(p[0] - foot[0], p[1] - foot[1]))
            if dist < best[0]:
                best = [dist, i, t, foot]
        d_min, i_seg, _t, foot = best
        self._route_i = i_seg
        rem = float(np.hypot(r[i_seg + 1][0] - foot[0],
                             r[i_seg + 1][1] - foot[1]))
        for i in range(i_seg + 1, n - 1):
            rem += float(np.hypot(r[i + 1][0] - r[i][0],
                                  r[i + 1][1] - r[i][1]))
        need = float(look_dist)
        pt = foot.copy()
        i = i_seg
        while True:
            nxt = r[i + 1]
            seg = float(np.hypot(nxt[0] - pt[0], nxt[1] - pt[1]))
            if seg >= need or i + 1 >= n - 1:
                if seg < 1e-9:
                    return nxt, rem, d_min
                k = need / seg
                return pt + k * (nxt - pt), rem, d_min
            need -= seg
            pt = nxt.copy()
            i += 1

    def _predock_pose(self, back=None):
        """预交接位姿 = 锚点正后方 back 米、车头对着货堆（= ppo_3 的舒适位姿）"""
        b = PREDOCK_BACK if back is None else float(back)
        g = self.guidance_goal
        return np.array([float(g[0]) - b * float(np.cos(g[2])),
                         float(g[1]) - b * float(np.sin(g[2])),
                         float(g[2])])

    def _predock_gap(self):
        """到预交接位姿的 (位置偏差 m, 车头偏差 rad)"""
        pos, _, yaw = self.pose()
        p = self._predock_pose()
        d = float(np.hypot(p[0] - pos[0], p[1] - pos[1]))
        de = float(np.arctan2(np.sin(p[2] - yaw), np.cos(p[2] - yaw)))
        return d, de

    def _leg_speed(self, look, dist_g, rem, guide_pose=None):
        """远段跟随：**全程 ppo_2**（用户第七版指定：这一轮不许程序代替模型）"""
        pos, _, yaw = self.pose()
        heading = float(np.arctan2(look[1] - pos[1], look[0] - pos[0]))
        dev = self._route_deviation(pos) if NAV_PROGRAM_FALLBACK else 0.0
        clear = self._foot_clear(pos[0], pos[1], yaw) if NAV_PROGRAM_FALLBACK \
            else float("inf")
        # 模型一步能走多远就先看一眼前面：模型步长是 50 个物理步（0.05s），
        _ms = NAV_MODEL_SPEED * (0.6 if self.has_cargo else 1.0)
        _ps = 0.18
        _pred = self._foot_clear(pos[0] + _ps * float(np.cos(yaw)),
                                 pos[1] + _ps * float(np.sin(yaw)), yaw)
        clear = min(clear, _pred)
        # 刚出车（还没走上折线的第一段）时，"离折线距离"天然就是 2~3m
        if getattr(self, "_route_i", 0) < 1:
            dev = 0.0
        # 兜底状态带回滞：一旦交给程序就先把偏差收回来再还给模型
        _herr = abs(float(np.arctan2(np.sin(heading - yaw),
                                     np.cos(heading - yaw))))
        if getattr(self, "_fallback_active", False):
            if _herr < np.deg2rad(NAV_MODEL_BACK_DEG) \
                    and dev < NAV_FALLBACK_BACK \
                    and clear >= NAV_SAFE_MARGIN * 1.5:
                self._fallback_active = False
                self._log(f"  [切回] 已对正折线（车头偏差 "
                          f"{np.rad2deg(_herr):+.0f}°、离折线 {dev:.2f}m）"
                          f" → 交回 ppo_2")
        elif self.nav_model is None \
                or _herr > np.deg2rad(NAV_MODEL_ENTER_DEG) \
                or dev > NAV_FALLBACK_DEV \
                or clear < NAV_SAFE_MARGIN * 1.5:
            self._fallback_active = True
        # 不管前面判成什么，只要有模型就走模型（程序只保净空不足时停车）
        _model_drives = self.nav_model is not None
        if _model_drives:
            _ps = 0.18
            _pred = self._foot_clear(
                pos[0] + _ps * float(np.cos(yaw)),
                pos[1] + _ps * float(np.sin(yaw)), yaw)
            if not NAV_PROGRAM_FALLBACK:
                # 安全动作（**不换控制器**，方向盘仍然由 ppo_2 给）：
                if _pred < 0.15:
                    self._pin_ticks = getattr(self, "_pin_ticks", 0) + 1
                    if self._pin_ticks > PIN_BACKOFF_TICKS:
                        if getattr(self, "_escape_start", None) is None:
                            self._escape_start = np.array(
                                [pos[0], pos[1]], dtype=float)
                        _emv = float(np.hypot(pos[0] - self._escape_start[0],
                                              pos[1] - self._escape_start[1]))
                        _rear = self._foot_clear(
                            pos[0] - 0.6 * float(np.cos(yaw)),
                            pos[1] - 0.6 * float(np.sin(yaw)), yaw)
                        if _rear < 0.06 or _emv >= ESCAPE_BACK_M:
                            self._escape_start = None
                            self._pin_ticks = 0
                            if _emv >= ESCAPE_BACK_M:
                                try:
                                    self._replan_route("脱困完成")
                                except Exception:                 # noqa: BLE001
                                    pass
                        self.last_model = (f"安全脱困倒车({_emv:.2f}/"
                                           f"{ESCAPE_BACK_M:.1f}m 尾净空{_rear:.2f})")
                        self._drive_creep(-ESCAPE_BACK_V, 0.0)
                        return
                    self.last_model = f"安全停车(净空{_pred:.2f}m)"
                    self._drive_creep(0.0, self.controller.current_steer)
                    return
                self._pin_ticks = 0
                _scale = 1.0 if _pred >= 0.55 else max(0.25,
                                                       (_pred - 0.15) / 0.40)
            else:
                _scale = 1.0
            # 引导点方位限幅（21/26 号同款）：两个模型训练时目标都在车头
            if guide_pose is not None:
                # 预交接位姿引导：直接把"要停的位姿"喂给 ppo_2 —— 它的观测里
                guide = np.array([guide_pose[0], guide_pose[1], guide_pose[2]])
            else:
                _herr = float(np.arctan2(np.sin(heading - yaw),
                                         np.cos(heading - yaw)))
                if abs(_herr) > NAV_BEARING_LIMIT:
                    _sgn = float(np.copysign(1.0, _herr))
                    _aim = yaw + _sgn * NAV_BEARING_LIMIT
                    _gx = pos[0] + NAV_AIM_DIST * np.cos(_aim)
                    _gy = pos[1] + NAV_AIM_DIST * np.sin(_aim)
                    guide = np.array([_gx, _gy, _aim])
                else:
                    guide = np.array([look[0], look[1], heading])
            _gh = float(np.arctan2(guide[1] - pos[1], guide[0] - pos[0]) - yaw)
            _gh = float(np.arctan2(np.sin(_gh), np.cos(_gh)))
            if abs(_gh) > np.deg2rad(NAV_TURN_SLOW_DEG):
                _ms = min(_ms, NAV_TURN_SLOW_SPEED)
            obs = build_obs(self.model, self.data, guide,
                            self.forklift_body_id, self.controller,
                            self.root_dof_adr, self.mast_body_id,
                            self.forks_body_id, mode=0,
                            renderer=self.renderer)
            act, _ = self.nav_model.predict(obs, deterministic=True)
            self.last_obs, self.last_mode = obs, 0
            self.last_model = (f"ppo_2(折线前视 剩"
                               f"{len(self.route) - getattr(self, '_route_i', 0)}"
                               f" 偏{np.rad2deg(heading - yaw):+.0f}°"
                               f" 离折线{dev:.2f}m)")
            max_speed = _ms * _scale
            if self.nav_kind == "pick":
                try:
                    _d = float(np.hypot(pos[0] - float(self.current_goal[0]),
                                        pos[1] - float(self.current_goal[1])))
                    if _d < 2.5:
                        max_speed = min(max_speed, 0.8)
                except Exception:                                 # noqa: BLE001
                    pass
            if _scale < 1.0:
                self.last_model += f" 限速{_scale * 100:.0f}%"
            self._drive(float(act[0]),
                        float(act[1]) * self.controller.MAX_STEER, max_speed)
            return
        # 没有模型（--no-model 调试）时没人能开车：停住
        self._hold(20)

    def _replan_route(self, why=""):
        return self._replan_route_impl(why)

    def _trim_route_head(self, route, min_d=None):
        d0 = ROUTE_HEAD_MIN_M if min_d is None else float(min_d)
        pos, _, _ = self.pose()
        r = [np.asarray(p, dtype=float) for p in route]
        while len(r) > 1 and float(np.hypot(r[0][0] - pos[0],
                                           r[0][1] - pos[1])) < d0:
            r.pop(0)
        return r

    def _prepend_clear_waypoint(self, route):
        pos, _, yaw = self.pose()
        if self._foot_clear(pos[0], pos[1], yaw) >= CLEAR_OUT_CLEAR_M:
            return route
        back = CLEAR_OUT_BACK_M
        best = None
        for a in np.linspace(-np.pi, np.pi, 17):
            ca, sa = float(np.cos(a)), float(np.sin(a))
            q = (pos[0] + back * ca, pos[1] + back * sa)
            mid = (pos[0] + 0.5 * back * ca, pos[1] + 0.5 * back * sa)
            score = (self._foot_clear(q[0], q[1], a)
                     + 0.7 * self._foot_clear(mid[0], mid[1], a))
            if best is None or score > best[0]:
                best = (score, np.array([q[0], q[1]]), a)
        self._log(f"[导航] 起点贴着货堆（净空 "
                  f"{self._foot_clear(pos[0], pos[1], yaw):.2f}m）"
                  f"→ 先朝 {np.rad2deg(best[2]):+.0f}° 让开 {back:.1f}m"
                  f"（净空 {best[0]:.2f}m）")
        return [best[1]] + list(route)

    def _replan_route_impl(self, why=""):
        """从当前位姿重新规划本腿折线（脱困/卡死看门狗用）"""
        pos, _, yaw = self.pose()
        try:
            if self.nav_kind in ("stack", "ground"):
                self.route, note = self._plan_drop_route(self.guidance_goal,
                                                         self.nav_kind)
                self._log(f"[重规划] {why}：放货腿 {note}")
            else:
                self.route = self._plan_route(self.current_goal,
                                              self._nav_node)
                self._log(f"[重规划] {why}：取货腿 {len(self.route)} 点")
        except Exception as e:                            # noqa: BLE001
            self._log(f"[重规划] 失败（沿用原折线）：{e}")
            return
        self._route_i = 0
        self._handoff_back = False

    def _stall_watchdog(self):
        """卡死看门狗：连续 2000 tick 位移不到 0.25m 就重新规划路线"""
        pos, _, _ = self.pose()
        self._wd_n = getattr(self, "_wd_n", 0) + 1
        if self._wd_n % 500 != 0:
            return
        ref = getattr(self, "_wd_ref", None)
        if ref is None or float(np.hypot(pos[0] - ref[0],
                                        pos[1] - ref[1])) > 0.25:
            self._wd_ref = pos.copy()
            self._wd_strikes = 0
            return
        self._wd_strikes = getattr(self, "_wd_strikes", 0) + 1
        if self._wd_strikes >= 4:
            self._wd_strikes = 0
            self._wd_ref = pos.copy()
            self._log(f"[看门狗] 连续 2000 tick 几乎没动（{pos[0]:+.2f},"
                      f"{pos[1]:+.2f}）→ 重新规划路线")
            self._replan_route("卡死")

    def _nav_drop_tick(self):
        """载货导航（放货腿）：折线跟踪 -> 直线进站段上 1.2~1.95m 交 ppo_3"""
        pos, _, yaw = self.pose()
        g = self.guidance_goal
        e_x, e_y, e_th = body_frame_goal(pos, yaw, g[:2], g[2])
        dist_g = float(np.hypot(e_x, e_y))
        head = float(np.arctan2(np.sin(g[2] - yaw), np.cos(g[2] - yaw)))
        speed = abs(self.speed())
        if not self._docking:
            self._stall_watchdog()
        if self._arrived_by_truth_gate():
            if self._finish_align():
                return
            return
        if self._docking:
            self._dock_tick()
            return
        along, lat, head_g = self._guide_errors()
        look, rem, dev = self._path_follow(
            DROP_LOOK_SLOW if dist_g < 3.0 else DROP_LOOK)
        if look is None:
            self._hold(20)
            return
        if dist_g <= TURNIN_ENGAGE_M and (rem <= TURNIN_ENGAGE_REM_M
                                          or dist_g <= 1.60):
            # 同取货腿：车头偏差大就用 ppo_2 + 模糊目标先转，≤70° 再交 ppo_3
            if self._handoff_turn_tick(dist_g, head_g):
                return
            return
        self._leg_speed(look, dist_g, rem)

    KAPPA_TABLE = (0.0, 0.077, 0.120, 0.250, 0.390, 0.430)
    STEER_TABLE = (0.0, 0.30, 0.50, 0.70, 0.90, 1.0636)

    def _enter_dock(self):
        """把控制权交给末段对准（交接前轮胎回零，用户指定）"""
        pos, _, yaw = self.pose()
        eg_x, eg_y, _ = body_frame_goal(pos, yaw, self.guidance_goal[:2],
                                        self.guidance_goal[2])
        _who = "ppo_3(图像已处理)"
        self._log(f"[交接] 模型 → {_who}：距引导锚点 "
                  f"{float(np.hypot(eg_x, eg_y)):.2f}m → 轮胎回零后交接")
        self._center_steer()
        # 交接前**刹停**：新版 ppo_3 是在"车停稳、目标在正前方"的
        for _ in range(40):
            self._drive_creep(0.0, 0.0, steps=50)      # 40×50×1ms = 2s 上限
            if abs(self.speed()) < 0.02:
                break
        self._log(f"[交接] 已刹停（车速 {abs(self.speed()):.3f}m/s）→ 交 {_who}")
        self._dock_ticks = 0
        self._dock_hit0 = None
        self._back1_done = False
        self._align_check_retry = 0
        self._dock_sub = "align"
        self._docking = True
        self._dock_tick()
    def _moved_from(self, start):
        p = self.pose()[0]
        return float(np.hypot(p[0] - start[0], p[1] - start[1]))

    def step(self):
        s = self.state
        if s == 12:
            self._hold()
            return "done"
        self._total_ticks = getattr(self, "_total_ticks", 0) + 1
        if self._total_ticks % 400 == 0:
            self._status_log()
        # 实时监测"撞到货堆"：一旦车体/货物碰到货堆（或围墙、障碍）
        _n = len(self.judge.events["vehicle_collision"])
        if _n > getattr(self, "_hit_seen", 0):
            for _e in self.judge.events["vehicle_collision"][
                    getattr(self, "_hit_seen", 0):]:
                self._log(f"[碰撞] {_e}（tick {self._total_ticks}，"
                          f"第 {self.round} 轮，状态 {STATE_NAMES.get(s, '?')}）")
            self._hit_seen = _n
        t = self.task()

        if s in (0, 7):
            self._nav_tick()
            return "running"

        if s == 1:                       # 取货调整高度
            # 一轮的结尾是"放完货、回到取货区准备插货"：
            if getattr(self, "_return_home", False):
                self._return_home = False
                self._log(f"[收尾] 已返回取货区并对准（{self.pose()[0][0]:+.2f},"
                          f"{self.pose()[0][1]:+.2f}）→ 三轮作业结束")
                self._enter(12)
                return "running"
            self._hold(fork_hold=False)
            self._set_fork(t["pick_h"])
            if (self._fork_reached(t["pick_h"]) and abs(self.speed()) < 0.05) \
                    or self._state_timer > 12000:
                self._log(f"[取货] 货叉就位 {self.fork_z():.3f}m"
                          f"（目标 {t['pick_h']:.3f}）")
                self._enter(2)
            return "running"

        if s == 2:                       # 前移插货
            # 进入插货状态时**每次都**刷新接触基线（否则重试后基线过期，
            if getattr(self, "_insert_armed", False) is False:
                self._insert_contact0 = len(
                    self.judge.events["vehicle_collision"])
                self._insert_armed = True
                # 取货期间把**整垛取货堆**都登记成"本车正在操作"：叉齿要插到
                self.judge.note_carrying(self._pick_stack_bodies())
            self._drive_creep(INSERT_SPEED, 0.0)
            # 主判据用 21 号的叉齿上表面射线（左右都命中才停）：它只在"叉齿
            if self._fork_ray_up():
                self._log(f"[取货] 叉齿上表面射线命中（前进 {self._moved():.2f}m）"
                          f"，刹车")
                self._insert_contact0 = 0
                self._enter(3)
            elif (self._moved() > INSERT_MAX_FORWARD_M
                  or self._state_timer > INSERT_GUARD_TICKS
                  or len(self.judge.events["vehicle_collision"])
                  > getattr(self, "_insert_contact0", 0)):
                self._insert_contact0 = 0
                self._insert_diag()
                self._log(f"[插货保护] 中止（前进 {self._moved():.2f}m、"
                          f"tick {self._state_timer}、"
                          f"接触 {len(self.judge.events['vehicle_collision'])}）"
                          f"→ 后退重试")
                self._insert_retry = getattr(self, "_insert_retry", 0) + 1
                self._enter(5)           # 退回，重新走一遍导航对准
                self._after_backoff = "retry_align"
                return "running"
            return "running"

        if s == 3:                       # 刹车后抬升
            self._hold(fork_hold=False)
            if abs(self.speed()) < 0.05:
                self.judge.note_carrying([self._body_id(self.pick_pallet),
                                          self._body_id(self.pick_cargo)])
                self._set_fork(self.fork_z() + LIFT_AFTER_INSERT_M)
                self._lift_target = self.fork_z() + LIFT_AFTER_INSERT_M
                self._enter(4)
            return "running"

        if s == 4:                       # 抬升到位
            self._hold(fork_hold=False)
            if self._fork_reached(self._lift_target, tol=0.02):
                self._log(f"[取货] 抬升到位 {self.fork_z():.3f}m")
                _o = self._pallet_fork_offset()
                if _o:
                    self._log(f"[滑移诊断] 插货完成：托盘相对货叉 "
                              f"long={_o[0]:+.3f} lat={_o[1]:+.3f} up={_o[2]:+.3f}")
                self._enter(5)
            return "running"

        if s == 5:                       # 后退撤出货堆（载着托盘）
            self._drive_creep(-PROG_BACKOUT_V_LADEN, 0.0)   # 载货段单独限速
            if self._moved() > BACKOUT_PICK_M:
                if getattr(self, "_after_backoff", None) == "retry_align":
                    self._after_backoff = None
                    n = getattr(self, "_insert_retry", 0)
                    if n <= 2:
                        self._log(f"[插货保护] 第 {n} 次重试：重新导航对准取货点")
                        self.has_cargo = False
                        self._start_nav(self._pick_goal_true,
                                        self.guidance.sample(self._pick_goal_true),
                                        self._node_id(PICK_STACK, "pick"), "pick")
                        self._enter(0)
                        return "running"
                    self._log("[插货保护] 重试次数超限，本轮放弃取货")
                    self.has_cargo = False
                    self._round_failed = True
                    self._end_round(failed=True)
                    return "running"
                self._enter(6)
            return "running"

        if s == 6:                       # 载货行驶高度
            self._hold(fork_hold=False)
            # 取完货后按 21 号的规矩把货叉落到"载货行驶高度"
            carry_h = WITH_CARGO_FORK_H
            self._set_fork(carry_h)
            if self._fork_reached(carry_h, tol=0.03):
                # 只有真的把托盘叉起来（叉上确实有货）才算取货成功；
                if not self._load_on_fork():
                    self.has_cargo = False
                    self._log("[取货] 叉上没检测到托盘 → 本轮取货失败，中止本轮")
                    self._round_failed = True
                    self._end_round(failed=True)
                    return "running"
                self.has_cargo = True
                self._log("[取货] 完成，转入载货导航")
                self._start_nav(self._drop_goal_true, self._guide_drop,
                                self._node_id(t["target"], t["kind"]),
                                "stack" if t["kind"] == "stack" else "ground")
                self._enter(7)
            return "running"

        if s == 8:                       # 放货调整高度
            self._hold(fork_hold=False)
            if not getattr(self, "_drop_armed", False):
                self._drop_armed = True
                # 放货期间把目标垛也登记成"本车正在操作"：托盘要落到堆顶，
                self.judge.note_carrying(self._target_stack_bodies())
            self._set_fork(t["drop_h"])
            self._target_T = self._current_target_top()
            if self._fork_reached(t["drop_h"], tol=0.02) \
                    and abs(self.speed()) < 0.05 \
                    or self._state_timer > 15000:
                self._log(f"[放货] 货叉就位 {self.fork_z():.3f}m，"
                          f"承托面 {self._target_T:.3f}m")
                self._enter(9)
            return "running"

        if s == 9:                       # 前移送入
            if getattr(self, "_radar_key", None) != int(self.round):
                self._radar_key = int(self.round)
                _dl, _ds = self._radar_probe()
                self._log("[载货测距] 前移前：门架→载货后表面="
                          f"{'--' if _dl is None else f'{_dl:.3f}m'}，"
                          "门架→货堆面="
                          f"{'--' if _ds is None else f'{_ds:.3f}m'}，"
                          "货堆面−载货后表面="
                          f"{'--' if (_ds is None or _dl is None) else f'{_ds - _dl:+.3f}m'}")
            _dl, _ds = self._radar_probe()
            if _dl is not None and _ds is not None:
                _v = _ds - _dl
                rem = float(np.clip(_v - PROG_FEED_GAP_TARGET_M, -0.05, 0.40))
                if not getattr(self, "_radar_fwd_logged", False):
                    self._radar_fwd_logged = True
                    self._log(f"[雷达闭环] 前移起点 v={_v:+.3f}m（目标 "
                              f"{PROG_FEED_GAP_TARGET_M:+.3f}m）→ 需要前移 {rem:.3f}m")
                if self._moved() > FORWARD_DROP_M + PROG_FEED_GAP_MAX_EXTRA:
                    rem = 0.0        # 保护：闭环异常时不许无限往前顶
            else:
                rem = FORWARD_DROP_M - self._moved()
            if rem > 0.0:
                v_cmd = min(PROG_FORWARD_V,
                            max(FEED_STOP_V,
                                float(np.sqrt(2.0 * PROG_ACCEL_LIMIT * rem))))
                self._drive_creep(v_cmd, 0.0)
            else:
                self._hold(fork_hold=False)
                if abs(self.speed()) < 0.05:
                    self._log(f"[放货] 已送入目标上方（已停稳 "
                              f"{self._moved():.3f}m）")
                    self._enter(10)
            return "running"

        if s == 10:                      # 下落释放
            self._hold(fork_hold=False)
            if getattr(self, "_diag_key", None) != (int(self.round), 10):
                self._diag_key = (int(self.round), 10)
                self._diag_place("释放前")
            if self._state_timer < 200:
                if getattr(self, "_radar_rel_key", None) != int(self.round):
                    self._radar_rel_key = int(self.round)
                    _dl, _ds = self._radar_probe()
                    self._log("[载货测距] 释放前：门架→载货后表面="
                              f"{'--' if _dl is None else f'{_dl:.3f}m'}，"
                              "门架→货堆面="
                              f"{'--' if _ds is None else f'{_ds:.3f}m'}，"
                              "货堆面−载货后表面="
                              f"{'--' if (_ds is None or _dl is None) else f'{_ds - _dl:+.3f}m'}")
                return "running"
            self._set_fork(t["release_h"])
            low = self.body_low_z(self.pick_pallet)
            if False:
                _o = self._pallet_fork_offset()
                if _o:
                    self._log(f"[滑移诊断] 释放前：托盘相对货叉 "
                              f"long={_o[0]:+.3f} lat={_o[1]:+.3f} up={_o[2]:+.3f}")
                # 把"放置误差"拆成纵向/横向（靶位姿坐标系），
                try:
                    _t = self._place_target_pose()
                    _p = np.asarray(self.data.body(self._body_id(
                        self.pick_pallet)).xpos, dtype=float)
                    _dx, _dy = _p[0] - _t[0], _p[1] - _t[1]
                    _c, _s = float(np.cos(_t[2])), float(np.sin(_t[2]))
                    self._log(f"[放置分解] 纵向 {_dx * _c + _dy * _s:+.3f}m "
                              f"横向 {-_dx * _s + _dy * _c:+.3f}m")
                except Exception:                             # noqa: BLE001
                    pass
            settled = abs(low - self._target_T) < 0.05
            if (settled and self._fork_reached(t["release_h"], tol=0.03)) \
                    or self._state_timer > 25000:
                if self.has_cargo:       # 只有真的载着货才登记"放置结果"
                    self.judge.note_placed([self._body_id(self.pick_pallet),
                                            self._body_id(self.pick_cargo)],
                                           self._place_target_pose())
                self.judge.note_carrying([])
                self.has_cargo = False
                rep = self.judge.report()
                pe = rep["place_error"]
                self._log(f"[放货] 释放完成：托盘底 {low:.3f}m / "
                          f"承托面 {self._target_T:.3f}m；放置误差 "
                          f"{pe['pos_m'] * 100:.1f}cm / {pe['yaw_deg']:.2f}°")
                self._enter(11)
            return "running"

        if s == 11:                      # 倒车退出
            if getattr(self, "_diag_key", None) != (int(self.round), 11):
                self._diag_key = (int(self.round), 11)
                self._diag_place("释放后")
            self._drive_creep(-PROG_BACKOUT_V, 0.0)
            if self._moved() > BACKOUT_DROP_M:
                # 退够距离**再**把货叉降回空载行驶高度（训练里 ctrl[4]=-0.23）：
                if not getattr(self, "_fork_home_done", False):
                    if not getattr(self, "_fork_home_started", False):
                        self._fork_home_started = True
                        # _state_timer 计的是**物理步**，退车那 2m 就把它吃到
                        self._state_timer = 0
                        self._log("[放货] 退到位，开始把货叉降回行驶高度")
                    self._hold(fork_hold=False)
                    self._set_fork(EMPTY_FORK_H)
                    if self._fork_reached(EMPTY_FORK_H, tol=0.03) \
                            or self._state_timer > 60000:
                        self._fork_home_done = True
                        self._log(f"[放货] 货叉回到空载行驶高度 "
                                  f"{self.fork_z():.2f}m")
                    return "running"
                self._fork_home_done = False
                self._fork_home_started = False
                # ---- 第 3 轮到这里直接收工 ----
                self._end_round()
            return "running"

        if s == 12:
            self._hold(fork_hold=False)
            return "done"

        return "running"

def main():
    ap = argparse.ArgumentParser(description="27 号三轮取放作业")
    ap.add_argument("--no-render", action="store_true", help="不开可视化窗口")
    ap.add_argument("--no-camera", action="store_true",
                    help="关闭门架相机渲染（观测图像置零，仅用于排查）")
    ap.add_argument("--no-model", action="store_true", help="不加载 PPO，只跑流程")
    ap.add_argument("--no-guidance", action="store_true", help="关闭引导降质")
    ap.add_argument("--seed", type=int, default=2200)
    ap.add_argument("--demo-targets", default=None,
                    help="逗号分隔的三轮放货目标，例如 A2,B2,B4；给了就不弹选择界面")
    ap.add_argument("--no-target-ui", action="store_true",
                    help="不弹放货目标选择界面，直接用默认三轮")
    ap.add_argument("--max-ticks", type=int, default=200000)
    ap.add_argument("--group", type=int, default=None,
                    help="用 model/eval_groups.json 里的第 N 组三轮任务"
                         "（7 组综合评估用；不给就用默认三轮）")
    args = ap.parse_args()

    if args.demo_targets:
        picked = [t.strip().upper() for t in args.demo_targets.split(",")
                  if t.strip()]
        bad = [t for t in picked if t not in TARGET_CHOICES]
        if len(picked) != 3 or len(set(picked)) != 3 or bad:
            raise SystemExit("--demo-targets 需要 3 个互不相同的货堆，可选："
                             + "/".join(TARGET_CHOICES))
        globals()["DEMO_TARGETS"] = picked
    elif not args.no_target_ui and args.group is None:
        try:
            picked = choose_targets_ui(DEMO_TARGETS)
        except Exception as e:                                 # noqa: BLE001
            picked = None
            print("[UI] 图形界面不可用（%s: %s），使用默认三轮 %s"
                  % (type(e).__name__, e, DEMO_TARGETS))
        if picked:
            globals()["DEMO_TARGETS"] = picked
    if args.group is not None:
        globals()["EVAL_GROUP"] = int(args.group)
        print(f"[30 号] 评估组 #{args.group}：{EVAL_TASKS_FILE}")

    # ---- 开场自报家门：确认跑的是 27 号这一份，而不是别的项目的脚本 ----
    print("=" * 78)
    print(f"[27 号] 脚本     : {os.path.abspath(__file__)}")
    print(f"[27 号] 工作目录 : {os.getcwd()}")
    print(f"[27 号] 场景     : {os.path.abspath('model/warehouse_env.xml')}"
          f"（货堆 stack_* + 地面区，**没有货架 shelf_***）")
    print(f"[27 号] 任务     : P1 第 3/2/1 层 → 放 A2 堆顶 / B2 堆顶 / 地面区 G1")
    print(f"[27 号] 权重     : ppo_2=robot/ppo_2.zip  "
          f"ppo_3=robot/ppo_3.zip（图像处理后再喂）")
    print(f"[27 号] 末段     : 先把轮胎回零 → 车头转进 ±20° → ppo_3 → 程序对准兜底")
    print("=" * 78)
    with open("model/warehouse_env.xml", encoding="utf-8") as _f:
        _wh = _f.read()
    if "shelf_" in _wh or "cargo_3" in _wh:
        print("[警告] 当前加载的 warehouse_env.xml 里出现了 21 号的货架/货物命名"
              "（shelf_*/cargo_*），说明场景文件被换过，27 号流程会跑不通。")
    if "stack_A1_L1_cargo" not in _wh:
        print("[警告] 当前加载的 warehouse_env.xml 里没有 27 号的 stack_A1_L1_cargo，"
              "请确认用的是 27 号自己的 model/warehouse_env.xml。")

    fsm = ForkliftTaskFSM(use_model=not args.no_model,
                          render=not args.no_camera,
                          seed=args.seed,
                          guidance=not args.no_guidance)

    viewer = None
    if not args.no_render:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(fsm.model, fsm.data)
        fsm.viewer = viewer          # 让物理步内部也能刷窗口（画面更流畅）
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = fsm.forklift_body_id
        viewer.cam.distance = 10.0
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -30

    ticks = 0
    status = "running"
    _sim0, _wall0 = float(fsm.data.time), time.perf_counter()
    try:
        while status == "running" and ticks < args.max_ticks:
            status = fsm.step()
            ticks += 1
            if viewer is not None:
                if not viewer.is_running():
                    break
                if not VIEW_SYNC_EVERY:
                    viewer.sync()
                _lag = (float(fsm.data.time) - _sim0) - (time.perf_counter()
                                                         - _wall0)
                if _lag > 0.0:
                    time.sleep(min(_lag, 0.05))
                else:
                    _sim0, _wall0 = float(fsm.data.time), time.perf_counter()
    finally:
        rep = fsm.judge.report()
        print("\n================ 判定结果 ================")
        print(fsm.judge.summary())
        for k in ("topple", "vehicle_collision", "cargo_collision"):
            if rep[k]:
                print(f"  {k}: {rep[k][:5]}")
        print(f"作业轮次完成: {min(fsm.round, fsm.total_rounds)}/"
              f"{fsm.total_rounds}，tick={ticks}，状态={STATE_NAMES[fsm.state]}")
        if viewer is not None:
            viewer.close()

if __name__ == "__main__":
    main()
