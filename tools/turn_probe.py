"""开发期：标定"原地摆头"——前进/后退交替时车身转多快、位置漂多少。"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=float, default=0.22)
    ap.add_argument("--half", type=int, default=120)
    ap.add_argument("--steer", type=float, default=0.45)
    ap.add_argument("--ticks", type=int, default=4000)
    ap.add_argument("--flip-reverse", action="store_true",
                    help="倒车段舵向取反（默认与前进段同号）")
    args = ap.parse_args()

    import test_4_warehouse as F

    fsm = F.ForkliftTaskFSM(use_model=False, render=False, seed=0, verbose=False)
    adr = fsm.root_qpos_adr
    fsm.data.qpos[adr:adr + 3] = [0.0, -6.0, 0.0]
    fsm.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    fsm.data.qvel[:] = 0.0
    mujoco.mj_forward(fsm.model, fsm.data)
    fsm.controller.reset()
    p0 = fsm.pose()[0].copy()
    y0 = fsm.pose()[2]
    print(f"v={args.v} 半周期={args.half}tick 舵={args.steer}")
    for t in range(args.ticks):
        k = t % (2 * args.half)
        forward = k < args.half
        if forward:
            fsm.controller.apply_creep(args.v, args.steer * fsm.controller.MAX_STEER)
        else:
            sgn = -1.0 if args.flip_reverse else 1.0
            fsm.controller.apply_creep(-args.v,
                                       sgn * args.steer * fsm.controller.MAX_STEER)
        mujoco.mj_step(fsm.model, fsm.data)
        if (t + 1) % 400 == 0:
            p = fsm.pose()[0]
            y = fsm.pose()[2]
            print(f"  t={t + 1:5d} 车身转角={np.rad2deg(y - y0):+7.1f}° "
                  f"位置漂移=({p[0] - p0[0]:+.3f},{p[1] - p0[1]:+.3f}) "
                  f"|漂|={np.hypot(p[0] - p0[0], p[1] - p0[1]):.3f}m")
        if (t + 1) % args.half == 0:
            p = fsm.pose()[0]
            y = fsm.pose()[2]
            print(f"    p{t // args.half + 1:3d} "
                  f"{'进' if (t // args.half) % 2 == 0 else '退'} "
                  f"y={y - y0:+.4f}rad 车=({p[0] - p0[0]:+.3f},{p[1] - p0[1]:+.3f}) "
                  f"速={fsm.speed():+.3f} 轮={fsm.data.joint('spin_front_left').qvel[0]:+.2f}")


if __name__ == "__main__":
    main()
