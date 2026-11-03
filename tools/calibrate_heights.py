"""货叉高度实测标定（Task 4）——27 号场景。

产出 model/height_table.json：
  A. 各货堆静置后的真实承托面（堆体会沉降，名义层距 0.55 不成立）
  B. 取货插入窗口：第 i 层托盘"下层货物顶面"与"顶板下沿"之间的货叉可用区间
  C. 放货高度：载货偏移的几何关系（由 26 号实测关系 1.34/0.75/0.12 校验）

注意：C 只是几何推导出的初值；真正落位会由端到端运行（tools/run_round_dry.py）
用真实取放流程再确认一遍，必要时微调。

用法（env_isaacsim）：python tools/calibrate_heights.py
"""
import json
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

from f_c import merge_warehouse_and_forklift  # noqa: E402

LAYERS = {"A1": 1, "A2": 2, "A3": 1, "A4": 2,
          "B1": 2, "B2": 1, "B3": 2, "B4": 1, "P1": 3}
SETTLE = 4000


def body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def geom_aabb_z(model, data, bid):
    """body 的所有 geom 在世界 z 方向的最低/最高点。"""
    lo, hi = np.inf, -np.inf
    adr = model.body_geomadr[bid]
    for gid in range(adr, adr + model.body_geomnum[bid]):
        pos = data.geom_xpos[gid]
        R = data.geom_xmat[gid].reshape(3, 3)
        size = model.geom_size[gid]
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            ext = float(np.abs(R[2, :]).dot(size))
        else:
            ext = float(model.geom_rbound[gid])
        lo = min(lo, pos[2] - ext)
        hi = max(hi, pos[2] + ext)
    return float(lo), float(hi)


def main():
    fl = open("model/forklift_train.xml", encoding="utf-8").read()
    wh = open("model/warehouse_env.xml", encoding="utf-8").read()
    model = mujoco.MjModel.from_xml_string(
        merge_warehouse_and_forklift(wh, fl))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for _ in range(SETTLE):
        mujoco.mj_step(model, data)

    # ---------- A. 静置后的真实承托面 ----------
    tops = {}
    for sid, n in LAYERS.items():
        _, tops[sid] = geom_aabb_z(model, data,
                                   body_id(model, f"stack_{sid}_L{n}_cargo"))
    print("=== A. 静置后的真实堆顶（承托面）===")
    for sid in sorted(tops):
        nominal = 0.55 * LAYERS[sid]
        print(f"  stack_{sid}: {LAYERS[sid]} 层  顶面 z={tops[sid]:.4f}  "
              f"(名义 {nominal:.2f}, 沉降 {(nominal - tops[sid]) * 100:.1f} cm)")
    per_layer = np.mean([(0.55 * LAYERS[s] - t) / LAYERS[s] for s, t in tops.items()])
    print(f"  => 每层平均沉降 {per_layer * 100:.2f} cm，"
          f"实际层距 ≈ {0.55 - per_layer:.4f} m")

    # ---------- B. P1 取货插入窗口 ----------
    print("\n=== B. P1 取货插入窗口（货叉下表面可用区间）===")
    windows = {}
    for i in (1, 2, 3):
        pallet = body_id(model, f"stack_P1_L{i}_pallet")
        p_lo, p_hi = geom_aabb_z(model, data, pallet)
        if i == 1:
            below_top = 0.0
        else:
            _, below_top = geom_aabb_z(
                model, data, body_id(model, f"stack_P1_L{i - 1}_cargo"))
        top_deck_lo = None
        adr = model.body_geomadr[pallet]
        for gid in range(adr, adr + model.body_geomnum[pallet]):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if gname and gname.startswith("pallet_top"):
                R = data.geom_xmat[gid].reshape(3, 3)
                size = model.geom_size[gid]
                top_deck_lo = float(data.geom_xpos[gid][2]
                                    - np.abs(R[2, :]).dot(size))
        lo_fork = float(below_top)               # 叉齿要高于下层货物顶面
        hi_fork = float(top_deck_lo - 0.05)      # 齿顶低于顶板下沿（齿厚 5cm）
        windows[f"L{i}"] = dict(lo=lo_fork, hi=hi_fork,
                                mid=0.5 * (lo_fork + hi_fork),
                                pallet_bottom=p_lo, pallet_top=p_hi,
                                support_top=float(below_top),
                                deck_bottom=float(top_deck_lo))
        print(f"  L{i}: 可用 [{lo_fork:.3f}, {hi_fork:.3f}]  推荐 "
              f"{(lo_fork + hi_fork) / 2:.3f}   （下层货物顶面 {below_top:.3f}"
              f" / 顶板下沿 {top_deck_lo:.3f}）")

    # ---------- C. 放货高度（几何 + 26 号实测关系校验）----------
    # 托盘：顶板下沿 = 原点 + 0.12，支撑块最低点 = 原点 - 0.03，叉齿厚 0.05
    # 载货时托盘顶板下沿压在齿顶 => 原点 = 货叉下表面 - 0.07，最低点 = 货叉下表面 - 0.10
    # 26 号实测关系：货架层板顶 1.22 -> dropoff 1.34；0.62 -> 0.75；地面 0 -> 0.12
    #   => dropoff = 承托面 + 0.12
    print("\n=== C. 放货高度（carry=平移到位高度, release=下落到脱齿高度）===")
    gap = 0.10          # 载货时托盘最低点相对货叉下表面的落差
    drop = {}
    for tag, sid in (("on2", "A2"), ("on1", "B2")):
        T = float(tops[sid])
        drop[tag] = dict(target=sid, T=T,
                         carry_h=round(T + gap + 0.02, 3),
                         release_h=round(T + 0.03, 3))
    drop["ground"] = dict(target="ground", T=0.0, carry_h=0.12, release_h=0.03)
    for k, v in drop.items():
        print(f"  {k:6s} 承托面 T={v['T']:.3f}  carry={v['carry_h']:.3f}  "
              f"release={v['release_h']:.3f}")
    print("  （release 比 26 号的 0.20m 下放量小：货堆顶面下方就是货物，"
          "再多下放叉齿会插进下层货物）")

    table = dict(settled_layer_step=float(0.55 - per_layer),
                 stack_tops={k: float(v) for k, v in tops.items()},
                 insert_windows=windows, dropoff=drop,
                 note="dropoff/release 为几何初值，端到端运行确认")
    with open("model/height_table.json", "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print("\n已写出 model/height_table.json")


if __name__ == "__main__":
    main()
