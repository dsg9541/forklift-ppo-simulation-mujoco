"""JSON 存取模块 - 路径点与环境物体信息存取。

路径 JSON 格式：
{
  "points": [{"id": 0, "x": -5.0, "y": 0.0, "attached": false, "kind": ""}, ...],
  "edges": [{"from": 0, "to": 1}, ...]
}

字段说明：
- attached: 是否为红色附属路径点（随货架/货物移动）。
- kind: 特殊路径点归属类型，"shelf"=货架附属点，"goods"=货物附属点，
  普通路径点留空（""）。加载时用于辅助匹配对应父物体。

环境物体信息 JSON 格式：
{
  "environment": [
    {"type": "shelf", "name": "shelf_1", "x":6.0, "y":0.0, "z":0.0,
     "yaw":0.0, "size": {"L":0.8, "W":2.0, "H":2.0}},
    {"type": "goods", "name": "cargo_1", "x":0.0, "y":0.0,
     "bottom_z":0.0, "layer":1, "yaw":0.0,
     "pallet_z":0.075, "cargo_z":0.4, "size": {...}},
    ...
  ]
}
"""
import json
from items import AttachedPathPointItem


def save_path(filepath, points, edges):
    """保存路径点到 JSON。

    Args:
        filepath: 输出文件路径
        points: PathPointItem 列表（含红色附属点）
        edges: PathEdgeItem 列表

    红色附属点通过 attached:true 标记区分；kind 字段记录归属类型：
    "shelf"（货架附属点）、"goods"（货物附属点），普通路径点为 ""。
    """
    def _kind_of(p):
        if not isinstance(p, AttachedPathPointItem):
            return ""
        parent = getattr(p, "parent_item", None)
        return getattr(parent, "item_type", "") or ""

    data = {
        "points": [
            {"id": p.pid, "x": float(p.x), "y": float(p.y),
             "attached": isinstance(p, AttachedPathPointItem),
             "kind": _kind_of(p)}
            for p in points
        ],
        "edges": [
            {"from": int(e.src_pid), "to": int(e.dst_pid)}
            for e in edges
        ],
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_path(filepath):
    """读取路径 JSON，返回 (points, edges) 两个列表。

    points: [{"id":..,"x":..,"y":..}, ...]
    edges: [{"from":..,"to":..}, ...]
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = data.get("points", [])
    edges = data.get("edges", [])
    return points, edges


def export_environment_info(items, filepath):
    """导出环境物体（货架/障碍物/货物/叉车框）的名称、坐标、尺寸等信息到 JSON。

    Args:
        items: BaseItem 列表（仅含环境物体，不含路径点）
        filepath: 输出 JSON 文件路径
    """
    env = []
    for it in items:
        t = it.item_type
        entry = {
            "type": t,
            "name": it.name,
            "x": float(it.x),
            "y": float(it.y),
            "yaw": float(it.yaw),
        }
        if t == "goods":
            # 货物：含堆叠信息
            entry["bottom_z"] = float(it.bottom_z)
            entry["layer"] = int(it.layer)
            entry["pallet_z"] = float(it.pallet_z)
            entry["cargo_z"] = float(it.cargo_z)
            entry["size"] = {
                "pallet_L": float(it.PALLET_L),
                "pallet_W": float(it.PALLET_W),
                "pallet_H": float(it.PALLET_H),
                "cargo_L": float(it.CARGO_L),
                "cargo_W": float(it.CARGO_W),
                "cargo_H": float(it.CARGO_H),
            }
        elif t == "stack":
            # 货堆：层数 + 每层底面高度 + 名义顶面（目标位姿以最下层为准）
            entry["layers"] = int(it.layers)
            entry["layer_bottoms"] = [float(it.bottom_z(i))
                                      for i in range(1, int(it.layers) + 1)]
            entry["top_z"] = float(it.top_z)
            entry["size"] = {
                "pallet_L": float(it.PALLET_L),
                "pallet_W": float(it.PALLET_W),
                "pallet_H": float(it.PALLET_H),
                "cargo_L": float(it.CARGO_L),
                "cargo_W": float(it.CARGO_W),
                "cargo_H": float(it.CARGO_H),
                "layer_step": float(it.LAYER_STEP),
            }
        elif t == "drop_zone":
            # 地面指定放置区：贴地标记
            entry["z"] = float(it.z)
            entry["size"] = {"L": float(it.L), "W": float(it.W),
                             "H": float(it.H)}
        else:
            # 货架/障碍物/叉车框：含 z 与三围尺寸
            entry["z"] = float(it.z)
            entry["size"] = {"L": float(it.L), "W": float(it.W),
                             "H": float(it.H)}
        env.append(entry)
    data = {"environment": env}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
