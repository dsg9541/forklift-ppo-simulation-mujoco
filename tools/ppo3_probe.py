"""开发期探针：标定 ppo_3 视觉对准的"可收敛包线"。

把车按 (距锚点距离, 车头偏差, 横向偏移) 摆好，直接进入末段对接流程
（_enter_dock -> ppo_3 -> 程序对准兜底），看能不能收敛、几次决策收敛。

用法： python tools/ppo3_probe.py [--round 1]
"""
import argparse
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import test_4_warehouse as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fsm = T.ForkliftTaskFSM(use_model=True, render=True, seed=args.seed)
    fsm.round = args.round
    fsm.has_cargo = True
    fsm._begin_round()
    fsm.has_cargo = True
    t = fsm.task()
    d = fsm.data
    adr = fsm.root_qpos_adr
    # 上层已经搬走
    for up in range(t["pick_layer"] + 1, 4):
        for kind in ("pallet", "cargo"):
            bid = fsm._body_id(f"stack_P1_L{up}_{kind}")
            jid = fsm.model.body_jntadr[bid]
            qa = int(fsm.model.jnt_qposadr[jid])
            d.qpos[qa:qa + 3] = [60.0, 60.0, 0.3]
            d.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
    g = fsm._target_goal()                     # 真值锚点（不带噪声）
    fsm.use_guidance = False
    fsm.guidance_goal = g.copy()
    fsm._guide_drop = g.copy()
    fsm.current_goal = g.copy()
    u = np.array([np.cos(g[2]), np.sin(g[2])])
    n = np.array([-np.sin(g[2]), np.cos(g[2])])
    fsm._start_nav(g, g, fsm._node_id(t["target"], t["kind"]),
                   "stack" if t["kind"] == "stack" else "ground")
    fsm._enter(7)

    print(f"[标定] 目标 {t['target']}，锚点 {np.round(g, 2)}，"
          f"车头 {np.degrees(g[2]):+.0f}°")
    # 按**训练分布**摆放：训练脚本里目标相对车头是
    #   e_x = d·cos(θ/2)、e_y = ±d·sin(θ/2)、e_theta = ±θ（θ = 最大 90°）
    # 所以要测"大角度"，就得把车摆成"目标在车头前方的那个相对位姿"，
    # 而不是"目标在车正后方、只是车头歪了"。
    for dist in (3.0, 4.0, 5.0):
        for head_deg in (0.0, -30.0, -60.0, -85.0):
            for lat_sign in (0.0, 1.0):
                th = np.deg2rad(-head_deg)          # e_theta = -head_deg
                half = th / 2.0
                yaw = g[2] + np.deg2rad(head_deg)
                rel = np.array([dist * np.cos(half),
                                lat_sign * dist * np.sin(half)])
                ca, sa = np.cos(yaw), np.sin(yaw)
                pos = np.array(g[:2]) - np.array([ca * rel[0] - sa * rel[1],
                                                  sa * rel[0] + ca * rel[1]])
                d.qpos[adr:adr + 2] = pos
                d.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0,
                                           np.sin(yaw / 2)]
                d.qvel[fsm.root_dof_adr:fsm.root_dof_adr + 6] = 0.0
                d.qpos[fsm.lift_qpos_adr] = T.WITH_CARGO_FORK_H - 0.25
                d.ctrl[4] = T.WITH_CARGO_FORK_H - 0.25
                mujoco.mj_forward(fsm.model, d)
                fsm.controller.reset()
                fsm._docking = False
                fsm._dock_retry = 0
                fsm._handoff_back = False
                fsm._handoff_retry = 0
                fsm.dock_aligner.reset()
                fsm.state = 7
                fsm._state_timer = 0
                fsm._block_cnt = 0
                fsm._stuck = False
                fsm._leg_start = pos.copy()
                fsm._enter_dock(True)
                n0 = len(fsm.judge.events["vehicle_collision"])
                k = 0
                while k < 800 and fsm.state == 7:
                    fsm.step()
                    k += 1
                    if fsm.state != 7:
                        break
                p, _, yw = fsm.pose()
                e = T.body_frame_goal(p, yw, g[:2], g[2])
                err = float(np.hypot(e[0], e[1]))
                hit = len(fsm.judge.events["vehicle_collision"]) - n0
                print(f"[标定] 距离{dist:.1f}m 车头{head_deg:+.0f}° "
                      f"横向{'左' if lat_sign else '中'} → tick={k:4d} "
                      f"状态={fsm.state} "
                      f"末位置误差={err * 100:5.1f}cm "
                      f"车头偏差={np.degrees(e[2]):+5.1f}° 新碰撞={hit}")


if __name__ == "__main__":
    main()
