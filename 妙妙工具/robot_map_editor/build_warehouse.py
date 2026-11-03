"""仓库布局生成脚本（v2）。

生成：
1. warehouse_env.xml  - 环境物体（货架/障碍物/货物/叉车框），MuJoCo 风格
2. warehouse_path.json - 路径点与连线（钝角平滑路径）
3. warehouse_info.json - 环境物体信息汇总

仓库布局（三区域，俯视，原点在中心）：

  Y+ 北
  ↑
  │  [A1][A2][A3][A4]                    通道Y=11
  │                                       
  │  ·········P14→P15→P16→P17→P18·····  路径
  │                                       
  │  [B1][B2][B3][B4]    [取货区3层]      通道Y=2 / 取货区(10,5)
  │                                       
  │  P6←······P7→P8→P9→P10→P11→P12·····  路径
  │                                       
  │  [C1][C2][C3][C4]                    通道Y=-1
  │                                       
  │  P0→P1→P2→P3→P4→P5                   路径（叉车起点）
  └────────────────────────────────────→ X+

路径设计：S形经过三排货架，所有转角 45°（夹角 135°钝角），间距 3-5m。
"""
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYLIBS = os.path.normpath(os.path.join(_HERE, "..", "pylibs"))
if os.path.isdir(_PYLIBS) and _PYLIBS not in sys.path:
    sys.path.insert(0, _PYLIBS)

from PyQt5.QtWidgets import QApplication

import xml_handler
import json_handler
from items import (ShelfItem, ObstacleItem, GoodsItem, ForkliftItem)


def build_warehouse():
    """构建仓库环境物体列表。"""
    items = []

    # ===== 1. 货架区（3排 × 4个 = 12个货架，yaw=90 横向）=====
    # 货架尺寸 0.8×2.0×2.0，yaw=90 时 X方向2.0m，Y方向0.8m
    shelf_xs = [-10.0, -7.0, -4.0, -1.0]
    # 第一排 Y=10
    for i, sx in enumerate(shelf_xs):
        items.append(ShelfItem(x=sx, y=10.0, z=0.0, yaw=90.0,
                                name=f"shelf_A{i+1}"))
    # 第二排 Y=4
    for i, sx in enumerate(shelf_xs):
        items.append(ShelfItem(x=sx, y=4.0, z=0.0, yaw=90.0,
                                name=f"shelf_B{i+1}"))
    # 第三排 Y=-2
    for i, sx in enumerate(shelf_xs):
        items.append(ShelfItem(x=sx, y=-2.0, z=0.0, yaw=90.0,
                                name=f"shelf_C{i+1}"))

    # ===== 2. 取货区（1个位置，3层堆叠）=====
    # 货物尺寸 0.7×0.8，堆叠层间距 0.55m
    items.append(GoodsItem(x=10.0, y=5.0, bottom_z=0.0,
                            yaw=0.0, name="cargo_pick_L1"))
    items.append(GoodsItem(x=10.0, y=5.0, bottom_z=0.55,
                            yaw=0.0, name="cargo_pick_L2"))
    items.append(GoodsItem(x=10.0, y=5.0, bottom_z=1.10,
                            yaw=0.0, name="cargo_pick_L3"))

    # ===== 3. 叉车初始位置框（朝西，即朝向货架区）=====
    items.append(ForkliftItem(x=10.0, y=-2.0, z=0.0, yaw=180.0,
                                name="forklift_init"))

    # ===== 4. 障碍物（仓库围墙，name 以 obstacle 开头以便导入识别）=====
    # 北墙
    items.append(ObstacleItem(x=-2.0, y=14.0, z=0.0, yaw=0.0,
                               L=30.0, W=0.5, H=2.0, name="obstacle_wall_n"))
    # 南墙
    items.append(ObstacleItem(x=-2.0, y=-5.0, z=0.0, yaw=0.0,
                               L=30.0, W=0.5, H=2.0, name="obstacle_wall_s"))
    # 西墙
    items.append(ObstacleItem(x=-15.5, y=4.5, z=0.0, yaw=0.0,
                               L=0.5, W=20.0, H=2.0, name="obstacle_wall_w"))
    # 东墙
    items.append(ObstacleItem(x=13.0, y=4.5, z=0.0, yaw=0.0,
                               L=0.5, W=20.0, H=2.0, name="obstacle_wall_e"))

    return items


def build_path_points():
    """构建环形平滑路径（所有转角 45°，夹角 135°钝角，间距 3-5m）。

    路径流程（逆时针环绕货架区）：
    1. 从叉车起点(10,-2)向西经过第三排下方通道(Y=-2)
    2. 左端西北转北，沿货架左侧向上（照顾三排左侧）
    3. 顶部东北转东，经过第一排上方通道(Y=12)
    4. 右端东南转南，到达取货区(10,5)附近
    """
    points = [
        # === 第一段：向西经过第三排下方 Y=-2 ===
        (10.0, -2.0),    # P0  起点（叉车位置）
        (6.0, -2.0),     # P1  西
        (2.0, -2.0),     # P2  西
        (-2.0, -2.0),    # P3  西
        (-6.0, -2.0),    # P4  西
        (-10.0, -2.0),   # P5  西（第三排左端）

        # === 左侧向上（西北→北→北）===
        (-13.0, 1.0),    # P6  西北(45°)
        (-13.0, 5.0),    # P7  北（经过第二排Y=4左侧）
        (-13.0, 9.0),    # P8  北（经过第一排Y=10左侧）

        # === 顶部向东（东北→东）===
        (-10.0, 12.0),   # P9  东北(45°)
        (-6.0, 12.0),    # P10 东
        (-2.0, 12.0),    # P11 东
        (2.0, 12.0),     # P12 东
        (6.0, 12.0),     # P13 东（第一排上方）

        # === 右侧向下到取货区（东南→南）===
        (9.0, 9.0),      # P14 东南(45°)
        (9.0, 5.0),      # P15 南（到达取货区Y=5附近）
    ]
    return points


def build_path_edges(n_points):
    """构建路径连线（依次连接相邻路径点）。"""
    return [(i, i + 1) for i in range(n_points - 1)]


def verify_angles_and_spacing(points):
    """验证所有路径转角均为钝角（夹角>90°）且间距 3-5m。"""
    import math
    print("\n=== 路径转角与间距验证 ===")
    all_angle_ok = True
    all_spacing_ok = True

    # 间距检查
    print("间距:")
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        d = math.hypot(x1 - x0, y1 - y0)
        ok = 3.0 <= d <= 5.0
        if not ok:
            all_spacing_ok = False
        print(f"  P{i}->P{i+1}: {d:.2f}m {'✓' if ok else '✗ (需3-5m)'}")

    # 转角检查
    print("\n转角:")
    for i in range(1, len(points) - 1):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        dx1, dy1 = x1 - x0, y1 - y0
        dx2, dy2 = x2 - x1, y2 - y1
        a1 = math.degrees(math.atan2(dy1, dx1))
        a2 = math.degrees(math.atan2(dy2, dx2))
        diff = abs(a2 - a1)
        if diff > 180:
            diff = 360 - diff
        interior = 180 - diff
        ok = interior >= 90  # 钝角或平角
        if not ok:
            all_angle_ok = False
        print(f"  P{i}: 方向 {a1:+.0f}°→{a2:+.0f}°  "
              f"转角={diff:.0f}°  夹角={interior:.0f}°  "
              f"{'✓' if ok else '✗'}")

    print(f"\n  夹角结论: {'所有夹角≥90°（钝角/平角）✓' if all_angle_ok else '存在锐角 ✗'}")
    print(f"  间距结论: {'所有间距3-5m ✓' if all_spacing_ok else '存在超出范围的间距 ✗'}")
    return all_angle_ok and all_spacing_ok


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    app = QApplication([])

    items = build_warehouse()
    points = build_path_points()
    edges = build_path_edges(len(points))

    verify_angles_and_spacing(points)

    out_dir = os.path.join(_HERE, "..", "warehouse")
    os.makedirs(out_dir, exist_ok=True)

    xml_path = os.path.join(out_dir, "warehouse_env.xml")
    path_json = os.path.join(out_dir, "warehouse_path.json")
    info_json = os.path.join(out_dir, "warehouse_info.json")

    xml_handler.export_xml(items, xml_path)
    print(f"\n已导出环境 XML: {xml_path}")

    # 路径 JSON 代理对象
    class _P:
        def __init__(self, pid, x, y):
            self.pid, self.x, self.y = pid, x, y
    class _E:
        def __init__(self, src, dst):
            self.src_pid, self.dst_pid = src, dst
    p_objs = [_P(i, x, y) for i, (x, y) in enumerate(points)]
    e_objs = [_E(s, d) for s, d in edges]
    json_handler.save_path(path_json, p_objs, e_objs)
    print(f"已导出路径 JSON: {path_json}")

    json_handler.export_environment_info(items, info_json)
    print(f"已导出环境信息 JSON: {info_json}")

    print(f"\n=== 仓库布局汇总 ===")
    print(f"  货架: {sum(1 for it in items if it.item_type == 'shelf')} 个 (3排×4)")
    print(f"  货物: {sum(1 for it in items if it.item_type == 'goods')} 个 (取货区3层)")
    print(f"  障碍物(墙): {sum(1 for it in items if it.item_type == 'obstacle')} 个")
    print(f"  叉车框: {sum(1 for it in items if it.item_type == 'forklift')} 个")
    print(f"  路径点: {len(points)} 个")
    print(f"  路径连线: {len(edges)} 条")


if __name__ == "__main__":
    main()
