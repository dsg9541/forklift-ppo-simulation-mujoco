"""开发期探针：把车摆到取货锚点附近，扫（前向偏移 / 横向偏移 / 偏航）网格，
打印相机横向误差、门架雷达前向距离、左右镜像对称度，以及两条雷达剖面。

用法（env_isaacsim）：
    python tools/dock_probe.py --round 1
"""
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

from f_c import build_lidar_scan  # noqa: E402
from f_c.align_vision import (face_fit, image_lateral_error,  # noqa: E402
                              stack_mask)

FWD_LO, FWD_HI = 0, 20
MAST_LO = 99
MAST_SITES = ("lidar_fork_top_left", "lidar_fork_top_right",
              "lidar_fork_mid_left", "lidar_fork_mid_right")


def bring_to(fsm, pos_xy, yaw):
    adr = fsm.root_qpos_adr
    fsm.data.qpos[adr:adr + 3] = [pos_xy[0], pos_xy[1], 0.0]
    half = yaw / 2.0
    fsm.data.qpos[adr + 3:adr + 7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
    fsm.data.qvel[:] = 0.0
    mujoco.mj_forward(fsm.model, fsm.data)


def measure(fsm):
    fsm.renderer.update_scene(fsm.data, camera="mast_cam")
    img = np.transpose(fsm.renderer.render(), (2, 0, 1))
    lat, npix = image_lateral_error(img)
    lidar = np.asarray(build_lidar_scan(fsm.model, fsm.data,
                                        fsm.forklift_body_id, fsm.mast_body_id,
                                        fsm.forks_body_id), dtype=float) * 8.0
    fwd = lidar[FWD_LO:FWD_HI]
    valid = fwd < 7.99
    dist = float(fwd[valid].min()) if valid.any() else float("inf")
    sym = float(np.mean(np.abs(fwd - fwd[::-1])))
    psi, d_perp, npts = face_fit(fwd)
    mast = lidar[MAST_LO:].reshape(4, 3, 15)
    names = {}
    for idx, name in enumerate(MAST_SITES):
        names[name] = mast[idx]
    return dict(lat=lat, npix=npix, dist=dist, sym=sym, fwd=fwd, mast=names,
                img=img, psi=psi, d_perp=d_perp, npts=npts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--profile", action="store_true", help="打印雷达剖面向量")
    args = ap.parse_args()

    import test_4_warehouse as F

    fsm = F.ForkliftTaskFSM(use_model=False, render=True, seed=0, verbose=False)
    fsm.round = args.round
    fsm._begin_round()
    goal = fsm._pick_goal_true
    print(f"锚点（真值） goal={np.round(goal, 3)}  "
          f"堆心={np.round(fsm.body_pos('stack_P1_L1_cargo'), 3)}")
    fsm.controller.set_fork_target_height(fsm.task()["pick_h"])
    for _ in range(400):
        fsm.controller.apply_control(0.0, 0.0, 1.0, False)
        mujoco.mj_step(fsm.model, fsm.data)
    mujoco.mj_forward(fsm.model, fsm.data)

    print("\n== 基准位（锚点）==")
    bring_to(fsm, goal[:2], goal[2])
    m = measure(fsm)
    print(f"lat={m['lat']} npix={m['npix']} dist={m['dist']:.3f} sym={m['sym']:.3f}")
    if args.profile:
        print("fwd rays:", np.round(m["fwd"], 2))
        for k, v in m["mast"].items():
            print(f"  {k}: min={v.min():.2f} mean={v.mean():.2f}")

    print("\n== 横向偏移（车向左为正 = 世界 +x）==")
    for lat_off in (-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30):
        p = np.array([goal[0] + lat_off, goal[1]])
        bring_to(fsm, p, goal[2])
        m = measure(fsm)
        ls = "--" if m["lat"] is None else f"{m['lat']:+.3f}"
        ps = "--" if m["psi"] is None else f"{np.rad2deg(m['psi']):+6.2f}°"
        print(f"  车横向{lat_off:+.2f}m → 图像横向={ls} 像素={m['npix']:5d} "
              f"前向距离={m['dist']:.3f} 对称={m['sym']:.3f} "
              f"拟合法向={ps} 垂距="
              f"{'--' if m['d_perp'] is None else format(m['d_perp'], '.3f')} "
              f"命中={m['npts']}")

    print("\n== 前向偏移（正=更靠近货堆）==")
    for fwd_off in (-0.60, -0.40, -0.20, -0.10, 0.0, 0.10, 0.20):
        p = np.array([goal[0], goal[1] - fwd_off])
        bring_to(fsm, p, goal[2])
        m = measure(fsm)
        ls = "--" if m["lat"] is None else f"{m['lat']:+.3f}"
        ps = "--" if m["psi"] is None else f"{np.rad2deg(m['psi']):+6.2f}°"
        print(f"  前向{fwd_off:+.2f}m → 图像横向={ls} 像素={m['npix']:5d} "
              f"前向距离={m['dist']:.3f} 对称={m['sym']:.3f} "
              f"拟合法向={ps} 垂距="
              f"{'--' if m['d_perp'] is None else format(m['d_perp'], '.3f')} "
              f"命中={m['npts']}")

    print("\n== 偏航（正=车头偏左/逆时针）==")
    for dyaw in (-15.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 15.0):
        bring_to(fsm, goal[:2], goal[2] + np.deg2rad(dyaw))
        m = measure(fsm)
        ls = "--" if m["lat"] is None else f"{m['lat']:+.3f}"
        ps = "--" if m["psi"] is None else f"{np.rad2deg(m['psi']):+6.2f}°"
        print(f"  dyaw={dyaw:+5.1f}° → 图像横向={ls} 像素={m['npix']:5d} "
              f"前向距离={m['dist']:.3f} 对称={m['sym']:.3f} "
              f"拟合法向={ps} 垂距="
              f"{'--' if m['d_perp'] is None else format(m['d_perp'], '.3f')} "
              f"命中={m['npts']}")


if __name__ == "__main__":
    main()
