"""开发期探针：打印三轮的放货锚点、进站几何与足迹净空（不跑仿真）。

用法： python tools/geom_probe.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import test_4_warehouse as T  # noqa: E402


def main():
    fsm = T.ForkliftTaskFSM(use_model=False, render=False, seed=0)
    for rnd in (1, 2, 3):
        fsm.round = rnd
        t = fsm.task()
        g = fsm._target_goal()
        print(f"第 {rnd} 轮：目标 {t['target']}（{t['kind']}）"
              f" → 放货锚点 {np.round(g, 3)}，车头 {np.degrees(g[2]):+.0f}°")
        for side in (1.0, -1.0):
            _a, E, lane_pt, arc, Q = fsm._drop_geom(g, side)
            print(f"    进站侧 {'左' if side > 0 else '右'}："
                  f"圆弧起点 E={np.round(E, 2)}，转弯道取位点 "
                  f"{np.round(lane_pt, 2)}，弧终点 Q={np.round(Q, 2)}，"
                  f"直线进站 {float(np.hypot(*(np.asarray(g[:2]) - Q))):.2f}m，"
                  f"尾巴最小净空 {fsm._entry_tail_clear(g, side):.2f}m")
        pick = fsm._pick_goal()
        print(f"    取货锚点 {np.round(pick, 3)}，车头 {np.degrees(pick[2]):+.0f}°")


if __name__ == "__main__":
    main()
