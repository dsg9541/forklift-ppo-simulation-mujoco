"""生成 27 号场景：两排货堆 + 取货堆（去掉货架与地面放置区）。

产物（写到 ..\\model\\，同时镜像到 妙妙工具\\warehouse\\27\\）：
  1. warehouse_env.xml   环境物体（货堆 / 障碍 / 围墙 / 叉车框）
  2. path_graph.json     路径点与连线（环形主干 + 起点接入段 + 红色附属点）
  3. warehouse_info.json 环境物体信息（含货堆层数）

布局（俯视，x 向东、y 向北）：
  y=+11.5  -------- 北墙 --------
  y=+7.0    [A1][A2][A3][A4]      货堆 1~3 层，行内间距 3.5m
  y=+4.15   .... 北道（A 排引导点）....
  y=-1.0    [B1][B2][B3][B4]
  y=-3.85   .... 南道（B 排引导点）....   [P1] 取货堆(8,-6) 3 层
  y=-8.5   -------- 南墙 --------

运行：python 妙妙工具/build_27_scene.py     （base 环境，PyQt5 在那里）
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYLIBS = os.path.normpath(os.path.join(_HERE, "..", "pylibs"))
if os.path.isdir(_PYLIBS) and _PYLIBS not in sys.path:
    sys.path.insert(0, _PYLIBS)
_EDITOR = os.path.join(_HERE, "robot_map_editor")
if os.path.isdir(_EDITOR) and _EDITOR not in sys.path:
    sys.path.insert(0, _EDITOR)

from PyQt5.QtWidgets import QApplication

import xml_handler
import json_handler
from items import (StackItem, ObstacleItem, ForkliftItem,
                    AttachedPathPointItem)

# ============================ 场景数据 ============================
ROW_Y = {"A": 7.0, "B": -1.0}
STACK_XS = [-10.0, -6.5, -3.0, 0.5]          # 行内间距 3.5m
LAYERS = {"A1": 1, "A2": 2, "A3": 1, "A4": 2,
          "B1": 2, "B2": 1, "B3": 2, "B4": 1}
PICK = ("P1", 8.0, -6.0, 3)                  # 取货堆：3 层
APPROACH_DIST = 2.85                         # 引导点距目标中心距离

CRATES = [(12.6, -6.8), (-14.6, 8.6), (-14.6, -7.4), (12.6, 8.6)]
CRATE_S = 0.6

WALLS = {
    "n": (0.0, 11.5, 31.5, 0.5),
    "s": (0.0, -8.5, 31.5, 0.5),
    "w": (-15.5, 1.5, 0.5, 20.5),
    "e": (15.5, 1.5, 0.5, 20.5),
}

FORKLIFT_INIT = (12.5, 3.1, 180.0)

# 环形主干：西道 x=-13 -> 南道 y=-3.85 -> 东段 x=8 -> 北道 y=4.15 -> 回西道
# 元素：("pt", x, y) 普通点；("stack", 行名, 序号, x) A/B 排附属点；
#       ("pick", x, y) 取货附属点
RING = [
    ("pt", -13.0, 2.9),        # 西北切角终点（向南）
    ("pt", -13.0, -2.6),       # 西南切角起点
    ("pt", -11.75, -3.85),     # 西南切角终点（向东）
    ("stack", "B", 1, -10.0),  # B1
    ("stack", "B", 2, -6.5),   # B2
    ("stack", "B", 3, -3.0),   # B3
    ("stack", "B", 4, 0.5),    # B4
    ("pt", 5.6, -3.85),
    ("pick", 8.0, -3.15),      # P1 进叉位（朝南）
    ("pt", 8.0, 2.9),          # 东北切角起点
    ("pt", 6.75, 4.15),        # 东北切角终点（向西）
    ("stack", "A", 4, 0.5),    # A4
    ("stack", "A", 3, -3.0),   # A3
    ("stack", "A", 2, -6.5),   # A2
    ("stack", "A", 1, -10.0),  # A1
    ("pt", -11.75, 4.15),      # 西北切角起点
]
START_NODES = [(12.5, 3.1), (9.0, 4.15)]     # 起点接入段（连到北道）


def build_items():
    items = []
    for row, y in ROW_Y.items():
        for i, x in enumerate(STACK_XS):
            sid = f"{row}{i + 1}"
            items.append(StackItem(x=x, y=y, layers=LAYERS[sid], yaw=90.0,
                                   name=sid))
    items.append(StackItem(x=PICK[1], y=PICK[2], layers=PICK[3], yaw=-90.0,
                           name=PICK[0]))
    for i, (cx, cy) in enumerate(CRATES, start=1):
        items.append(ObstacleItem(x=cx, y=cy, z=0.0, yaw=0.0,
                                  L=CRATE_S, W=CRATE_S, H=CRATE_S,
                                  name=f"obstacle_crate_{i}"))
    for key, (x, y, L, W) in WALLS.items():
        items.append(ObstacleItem(x=x, y=y, z=0.0, yaw=0.0,
                                  L=L, W=W, H=2.0,
                                  name=f"obstacle_wall_{key}"))
    items.append(ForkliftItem(x=FORKLIFT_INIT[0], y=FORKLIFT_INIT[1], z=0.0,
                              yaw=FORKLIFT_INIT[2], name="forklift_init"))
    return items


def build_path(items):
    """返回 (points, edges)。points 的顺序即 pid；附属点携带父物体。"""
    by_name = {getattr(it, "name", ""): it for it in items}
    points = []
    edges = []

    class _P:
        def __init__(self, pid, x, y):
            self.pid, self.x, self.y = pid, x, y

    def add_point(obj):
        points.append(obj)
        return len(points) - 1          # pid

    ring_pids = []
    for entry in RING:
        kind = entry[0]
        if kind == "pt":
            pid = add_point(_P(len(points), entry[1], entry[2]))
        elif kind == "stack":
            _, row, idx, x = entry
            parent = by_name[f"{row}{idx}"]
            pid = add_point(AttachedPathPointItem(
                len(points), parent, x, ROW_Y[row] - APPROACH_DIST))
        elif kind == "pick":
            parent = by_name[PICK[0]]
            pid = add_point(AttachedPathPointItem(len(points), parent,
                                                  entry[1], entry[2]))
        ring_pids.append(pid)
        if len(ring_pids) > 1:
            edges.append((ring_pids[-2], ring_pids[-1]))
    edges.append((ring_pids[-1], ring_pids[0]))     # 闭环

    # 起点接入段：起点 -> 北道切入 -> 环上最近的普通点（6.75, 4.15）
    start_pids = [add_point(_P(len(points), x, y)) for x, y in START_NODES]
    edges.append((start_pids[0], start_pids[1]))
    joint = None
    for pid in ring_pids:
        p = points[pid]
        if abs(p.x - 6.75) < 1e-6 and abs(p.y - 4.15) < 1e-6:
            joint = pid
            break
    assert joint is not None, "找不到接入点"
    edges.append((start_pids[1], joint))

    class _E:
        def __init__(self, src, dst):
            self.src_pid, self.dst_pid = src, dst

    return points, [_E(s, d) for s, d in edges]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    app = QApplication([])
    items = build_items()
    points, edges = build_path(items)

    out_dir = os.path.normpath(os.path.join(_HERE, "..", "model"))
    mirror_dir = os.path.join(_HERE, "warehouse", "27")
    for d in (out_dir, mirror_dir):
        os.makedirs(d, exist_ok=True)
        xml_handler.export_xml(items, os.path.join(d, "warehouse_env.xml"))
        json_handler.save_path(os.path.join(d, "path_graph.json"),
                               points, edges)
        json_handler.export_environment_info(
            items, os.path.join(d, "warehouse_info.json"))

    n_stack = sum(1 for it in items if it.item_type == "stack")
    n_obs = sum(1 for it in items if it.item_type == "obstacle")
    n_att = sum(1 for p in points if isinstance(p, AttachedPathPointItem))
    print(f"货堆 {n_stack}（含取货堆 P1）、障碍/墙 {n_obs}（无地面放置区）")
    print(f"路径点 {len(points)}（红色附属点 {n_att}）、连线 {len(edges)}")
    print(f"已写出: {out_dir}")
    print(f"已镜像: {mirror_dir}")


if __name__ == "__main__":
    main()
