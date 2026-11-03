"""27 号场景自检：加载 / 命名 / 静置沉降 / 路径图连通性与转角。

用法（env_isaacsim）：
    python tools/validate_scene.py
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
LAYER_STEP = 0.55
SETTLE_STEPS = 4000          # 4 s


def load_model():
    with open("model/forklift_train.xml", encoding="utf-8") as f:
        fl = f.read()
    with open("model/warehouse_env.xml", encoding="utf-8") as f:
        wh = f.read()
    merged = merge_warehouse_and_forklift(wh, fl)
    return mujoco.MjModel.from_xml_string(merged)


def check_names(model):
    """(名称, MuJoCo 物体类型) 列表 -> 缺失清单。"""
    B, J, C = (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT,
               mujoco.mjtObj.mjOBJ_CAMERA)
    need = [("forklift", B), ("forks", B), ("lift_joint", J),
            ("mast_cam", C), ("drop_zone_G1", B)]
    need += [(f"stack_{sid}_L{i}_{part}", B)
             for sid, n in LAYERS.items()
             for i in range(1, n + 1) for part in ("pallet", "cargo")]
    need += [(f"obstacle_crate_{i}", B) for i in range(1, 5)]
    need += [(f"obstacle_wall_{k}", B) for k in "nsew"]
    missing = [n for n, kind in need
               if mujoco.mj_name2id(model, kind, n) < 0]
    return need, missing


def settle_and_measure(model, data):
    """静置后测量每个货堆：顶层货物顶面高度、位移、倾角。"""
    start = {sid: data.body(f"stack_{sid}_L1_cargo").xpos.copy()
             for sid in LAYERS}
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
    out = {}
    for sid, n in LAYERS.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                f"stack_{sid}_L{n}_cargo")
        pos = data.body(bid).xpos.copy()
        rot = data.xmat[bid].reshape(3, 3)
        tilt = np.degrees(np.arccos(np.clip(rot[2, 2], -1.0, 1.0)))
        drift = float(np.hypot(pos[0] - start[sid][0],
                               pos[1] - start[sid][1]))
        out[sid] = dict(top=float(pos[2] + 0.15), drift=drift,
                        tilt=float(tilt), nominal=LAYER_STEP * n)
    return out


def check_path():
    with open("model/path_graph.json", encoding="utf-8") as f:
        g = json.load(f)
    pts = {p["id"]: p for p in g["points"]}
    adj = {pid: [] for pid in pts}
    for e in g["edges"]:
        adj[e["from"]].append(e["to"])
    start = next(p["id"] for p in g["points"] if p["attached"] is False)
    # 从起点做 BFS
    seen = {start}
    frontier = [start]
    while frontier:
        cur = frontier.pop()
        for nxt in adj[cur]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    targets = [p for p in g["points"] if p["attached"]]
    unreachable = [p["id"] for p in targets if p["id"] not in seen]
    # 相邻连线转角（只在普通点 + 附属点的折线上看）
    turns = []
    for e in g["edges"]:
        a, b = pts[e["from"]], pts[e["to"]]
        for e2 in g["edges"]:
            if e2["from"] != b["id"]:
                continue
            c = pts[e2["to"]]
            v1 = np.array([b["x"] - a["x"], b["y"] - a["y"]])
            v2 = np.array([c["x"] - b["x"], c["y"] - b["y"]])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cos = float(np.clip(v1.dot(v2) / (n1 * n2), -1.0, 1.0))
            turns.append((b["id"], float(np.degrees(np.arccos(cos)))))
    return dict(points=len(pts), edges=len(g["edges"]),
                unreachable=unreachable, turns=turns)


def main():
    print("=== (a) 加载 ===")
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert np.isfinite(data.qpos).all(), "qpos 出现 NaN/Inf"
    print(f"  OK  nbody={model.nbody} ngeom={model.ngeom} nq={model.nq}")

    print("=== (b) 命名 ===")
    need, missing = check_names(model)
    print(f"  需要 {len(need)} 个 body，缺失 {len(missing)}")
    if missing:
        print("  缺失:", missing[:20])
        raise SystemExit(1)

    print("=== (c) 静置沉降（4 s）===")
    m = settle_and_measure(model, data)
    bad = []
    for sid in sorted(m):
        r = m[sid]
        drop = r["nominal"] - r["top"]
        print(f"  stack_{sid}: 顶面 {r['top']:.3f} (名义 {r['nominal']:.2f}, "
              f"沉降 {drop * 100:+.1f}cm)  位移 {r['drift'] * 100:.2f}cm  "
              f"倾角 {r['tilt']:.2f}°")
        # 每层沉降约 7cm（托盘支撑块 3cm + 货物落座 4cm），3 层堆总沉降 ~21cm
        per_layer_drop = drop / LAYERS[sid]
        if (r["drift"] > 0.05 or r["tilt"] > 5.0
                or per_layer_drop > 0.09 or per_layer_drop < -0.005):
            bad.append(sid)
    print("  异常:", bad if bad else "无")

    print("=== (d) 路径图 ===")
    p = check_path()
    print(f"  点 {p['points']}、边 {p['edges']}，不可达附属点 {p['unreachable']}")
    worst = sorted(p["turns"], key=lambda t: -t[1])[:6]
    print("  转角最大的 6 处:", [(f"#{i}", f"{a:.1f}°") for i, a in worst])
    ok_turn = all(a <= 92.0 for _, a in p["turns"])
    print(f"  全部转角 <= 92°: {ok_turn}")

    ok = (not missing) and (not p["unreachable"]) and ok_turn and not bad
    print("\n=== 结论:", "全部通过 ===" if ok else "存在问题，见上 ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
