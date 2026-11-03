"""开发期：检查取货插入的几何（货叉能不能从托盘叉孔进去）。"""
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


def zrange(fsm, bid):
    lo, hi = np.inf, -np.inf
    adr = fsm.model.body_geomadr[bid]
    for gid in range(adr, adr + fsm.model.body_geomnum[bid]):
        R = fsm.data.geom_xmat[gid].reshape(3, 3)
        size = fsm.model.geom_size[gid]
        if fsm.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            ext = float(np.abs(R[2, :]).dot(size))
        else:
            ext = float(fsm.model.geom_rbound[gid])
        lo = min(lo, float(fsm.data.geom_xpos[gid][2]) - ext)
        hi = max(hi, float(fsm.data.geom_xpos[gid][2]) + ext)
    return lo, hi


def main():
    import test_4_warehouse as F

    fsm = F.ForkliftTaskFSM(use_model=False, render=False, seed=0, verbose=False)
    anchor = fsm._pick_goal_true
    adr = fsm.root_qpos_adr
    print(f"锚点 {np.round(anchor, 3)}")
    for rnd in (1, 2, 3):
        fsm.round = rnd
        fsm._begin_round()
        t = fsm.task()
        half = anchor[2] / 2.0
        back = float(os.environ.get("BACK", "0.0"))
        fsm.data.qpos[adr:adr + 3] = [anchor[0], anchor[1] - back, 0.0]
        fsm.data.qpos[adr + 3:adr + 7] = [np.cos(half), 0, 0, np.sin(half)]
        fsm.data.qvel[:] = 0.0
        fsm.controller.reset()
        fsm.controller.set_fork_target_height(t["pick_h"])
        mujoco.mj_forward(fsm.model, fsm.data)
        for _ in range(6000):     # 让货叉开环走到位
            fsm.controller.apply_creep(0.0, 0.0)
            mujoco.mj_step(fsm.model, fsm.data)
        pl = fsm._body_id(f"stack_P1_L{int(t['pick_layer'])}_pallet")
        cg = fsm._body_id(f"stack_P1_L{int(t['pick_layer'])}_cargo")
        fk = fsm.fork_z()
        print(f"第{rnd}轮 取 L{t['pick_layer']} pick_h={t['pick_h']:.3f} "
              f"货叉下表面={fk:.3f}（叉齿 z {fk:.3f}~{fk + 0.05:.3f}）")
        print(f"        托盘 z {zrange(fsm, pl)[0]:.3f}~{zrange(fsm, pl)[1]:.3f}"
              f"  货物 z {zrange(fsm, cg)[0]:.3f}~{zrange(fsm, cg)[1]:.3f}")
        # 前移 1.2m 看会不会撞
        judge = F.Judge(fsm.model, fsm.data, fsm.forklift_body_id)
        hit = False
        for k in range(30000):
            fsm.controller.apply_creep(0.08, 0.0)
            mujoco.mj_step(fsm.model, fsm.data)
            judge.step(fsm.dt)
            if fsm._fork_ray_up() and not hit:
                hit = True
                fj = fsm.data.body(fsm.forks_body_id).xpos
                pl = fsm.body_pos(f"stack_P1_L{int(t['pick_layer'])}_pallet")
                print(f"        [射线命中] k={k} 前进 {anchor[1] - fsm.pose()[0][1] - back:.3f}m "
                      f"托盘中心距车体前 {float(np.hypot(pl[0] - fsm.pose()[0][0], pl[1] - fsm.pose()[0][1])):.2f}m "
                      f"（叉齿 1.27~2.27m）")
            if k % 4000 == 0:
                print(f"        k={k:5d} 车y={fsm.pose()[0][1]:.3f} "
                      f"（前进 {anchor[1] - fsm.pose()[0][1] - back:.3f}m）"
                      f" 车速={fsm.speed():+.3f}")
        p = fsm.pose()[0]
        pairs = set()
        for i in range(fsm.data.ncon):
            c = fsm.data.contact[i]
            b1 = int(fsm.model.geom_bodyid[c.geom1])
            b2 = int(fsm.model.geom_bodyid[c.geom2])
            g1 = mujoco.mj_id2name(fsm.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            g2 = mujoco.mj_id2name(fsm.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            n1 = mujoco.mj_id2name(fsm.model, mujoco.mjtObj.mjOBJ_BODY, b1) or "world"
            n2 = mujoco.mj_id2name(fsm.model, mujoco.mjtObj.mjOBJ_BODY, b2) or "world"
            pairs.add(f"{n1}:{g1} <-> {n2}:{g2}")
        print(f"        前移后 车y={p[1]:.3f}（前移 {anchor[1] - p[1]:.3f}m）"
              f" 碰撞={judge.events['vehicle_collision'][:2]} "
              f"翻倒={judge.events['topple'][:1]}")
        print("        接触: " + "; ".join(sorted(pairs)[:8]))
        fx = fsm.data.body(fsm.forks_body_id).xpos
        print(f"        叉齿中心 ({fx[0]:.2f},{fx[1]:.2f},{fx[2]:.2f})"
              f" 货堆 ({fsm.body_pos('stack_P1_L1_cargo')[0]:.2f},"
              f"{fsm.body_pos('stack_P1_L1_cargo')[1]:.2f})")
        fk_ids = set()
        for bid in range(fsm.model.nbody):
            cur = bid
            while cur > 0:
                if cur == fsm.forklift_body_id:
                    fk_ids.add(bid)
                    break
                cur = int(fsm.model.body_parentid[cur])
        touch = []
        for i in range(fsm.data.ncon):
            c = fsm.data.contact[i]
            b1 = int(fsm.model.geom_bodyid[c.geom1])
            b2 = int(fsm.model.geom_bodyid[c.geom2])
            if b1 in fk_ids or b2 in fk_ids:
                n1 = mujoco.mj_id2name(fsm.model, mujoco.mjtObj.mjOBJ_BODY, b1) or "w"
                n2 = mujoco.mj_id2name(fsm.model, mujoco.mjtObj.mjOBJ_BODY, b2) or "w"
                touch.append(f"{n1}<->{n2}")
        wl = fsm.data.joint("spin_front_left").qvel[0]
        wr = fsm.data.joint("spin_front_right").qvel[0]
        print(f"        车体接触={touch} 车速={fsm.speed():+.3f} "
              f"轮速={wl:+.2f}/{wr:+.2f} ctrl={np.round(fsm.data.ctrl[:4], 1)}")
        # 几何对照：托盘三个叉孔块 vs 两根叉齿的 (横向, 高度) 区间
        bid_pl = fsm._body_id(f"stack_P1_L{int(t['pick_layer'])}_pallet")
        adr2 = fsm.model.body_geomadr[bid_pl]
        for gid in range(adr2, adr2 + fsm.model.body_geomnum[bid_pl]):
            nm = mujoco.mj_id2name(fsm.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            gp = fsm.data.geom_xpos[gid]
            sz = fsm.model.geom_size[gid]
            print(f"        托盘geom {nm}: x[{gp[0] - sz[1]:.3f},{gp[0] + sz[1]:.3f}] "
                  f"z[{gp[2] - sz[2]:.3f},{gp[2] + sz[2]:.3f}]")
        for gid in range(fsm.model.body_geomadr[fsm.forks_body_id],
                         fsm.model.body_geomadr[fsm.forks_body_id]
                         + fsm.model.body_geomnum[fsm.forks_body_id]):
            nm = mujoco.mj_id2name(fsm.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            gp = fsm.data.geom_xpos[gid]
            sz = fsm.model.geom_size[gid]
            print(f"        货叉geom {nm}: x[{gp[0] - sz[1]:.3f},{gp[0] + sz[1]:.3f}] "
                  f"z[{gp[2] - sz[2]:.3f},{gp[2] + sz[2]:.3f}]")


if __name__ == "__main__":
    main()
