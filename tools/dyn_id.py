"""开发期：程序控制用的二自由度车辆模型辨识。

模型（沿行进距离 s 的微分，理想入库系）：
    dL/ds   = ψ_car - β(s)          # L: 车身横向偏移；ψ_car: 车身偏航
    dψ_car/ds = κ(s)
实测 β、κ 与转向指令 s、速度 v 的关系，用于设计 z = L + (β/κ)·ψ_car 的
双积分控制律（只有这种结构能同时把 L 和 ψ_car 收到 0）。
"""
import os
import sys

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import mujoco  # noqa: E402


def one(fsm, v, s, seconds=6.0):
    adr = fsm.root_qpos_adr
    fsm.data.qpos[adr:adr + 3] = [0.0, -6.0, 0.0]
    fsm.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    fsm.data.qvel[:] = 0.0
    mujoco.mj_forward(fsm.model, fsm.data)
    fsm.controller.reset()
    p0 = fsm.pose()[0].copy()
    n = int(seconds / fsm.dt)
    for _ in range(n):
        fsm.controller.apply_creep(v, s * fsm.controller.MAX_STEER)
        mujoco.mj_step(fsm.model, fsm.data)
    p1 = fsm.pose()[0]
    y1 = fsm.pose()[2]
    travel = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    if travel < 1e-6:
        return None
    kappa = float(y1) / travel                       # 车身偏航 / 行程
    path_dir = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    gam = path_dir / travel                          # 轨迹方向 / 行程
    beta = kappa - gam
    R = fsm.data.xmat[fsm.forklift_body_id].reshape(3, 3)
    vel = fsm.data.qvel[fsm.root_dof_adr:fsm.root_dof_adr + 3]
    beta_inst = float(np.arctan2(float(vel @ R[:, 1]), float(vel @ R[:, 0])))
    return dict(travel=travel, kappa=kappa, gam=gam, beta=beta,
                beta_inst=beta_inst, yaw=y1)


def main():
    import test_4_warehouse as F

    fsm = F.ForkliftTaskFSM(use_model=False, render=False, seed=0, verbose=False)
    fsm.controller.set_fork_raw_target(0.0)
    print(f"{'v':>5} {'s':>6} | {'行程':>7} {'κ=Δψ/Δs':>9} {'γ=Δdir/Δs':>10} "
          f"{'β=κ-γ':>8} {'β瞬时':>8} | {'β/κ':>6}")
    for v in (0.12, 0.20):
        for s in (0.10, 0.20, 0.30, 0.45):
            for sign in (1, -1):
                r = one(fsm, v, sign * s)
                if r is None:
                    continue
                ratio = r["beta"] / r["kappa"] if abs(r["kappa"]) > 1e-6 else np.nan
                print(f"{v:5.2f} {sign * s:+6.2f} | {r['travel']:7.3f} "
                      f"{r['kappa']:9.3f} {r['gam']:10.3f} {r['beta']:8.3f} "
                      f"{r['beta_inst']:8.3f} | {ratio:6.2f}")


if __name__ == "__main__":
    main()
