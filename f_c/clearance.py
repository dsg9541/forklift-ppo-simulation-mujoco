"""车体足迹净空检查（27 号部署侧安全网）。

叉车在 xy 平面上简化成一个"凸矩形足迹"：车体 + 配重 + 门架 + 货叉，
再加上货叉上驮着的托盘（载货时托盘在叉齿上方，投影仍落在足迹里）。
场景里的货堆/围墙/障碍是轴对齐矩形，于是净空 = 足迹边界点到这些矩形的
最小距离。

用途（都在部署侧，与训练无关）：
  · 选转向方向（起步圆弧 / 掉头圆弧）——挑净空大的那一侧；
  · 判断路径图里的一条边"车能不能走"；
  · 行驶中的实时安全阀——净空不足就停车/减速，绝不允许把货堆顶翻。

坐标系：世界 xy（z 忽略：货叉/托盘的高度（0.4~1.1m）与货堆高度重叠，
所以二维判断是偏保守的、安全的）。
"""

import numpy as np

# 足迹（车体坐标系：x 向前，y 向左）。用**两段矩形**而不是一个大矩形：
#   车体段: chassis(±1.17×±0.60) + 配重(-1.40) + 门架(±0.50)
#           → x ∈ [-1.45, 1.25]、y ∈ [-0.63, 0.63]
#   货叉段: 左右叉齿 x ∈ [1.27, 2.27]、y ∈ ±0.225，再加 5cm 余量
#           → x ∈ [1.20, 2.32]、y ∈ [-0.28, 0.28]
#   驮着的托盘（载货时）：0.7×0.8 → x ∈ [1.20, 2.10]、y ∈ ±0.45
# 用一个"把货叉也算成 0.63 宽"的大矩形会**过度保守**：实测它把
# "货叉已经伸进两垛货之间的缝隙"这种完全合法的动作判成净空 0，
# 车会直接停在原地（第一次调试就是这么卡死的）。
FOOT_RECTS = (
    (-1.42, 1.20, -0.61, 0.61),      # 车体 + 配重 + 门架（实车 ±0.60）
    (1.22, 2.29, -0.25, 0.25),       # 货叉（实车 ±0.225，叉尖 2.27m）
    (1.22, 2.09, -0.42, 0.42),       # 货叉上托盘（0.7×0.8，±0.4）
)

_STEP = 0.10


def _boundary(step=_STEP):
    """两段矩形边界上的采样点（车体坐标系）。"""
    pts = []
    for (x0, x1, y0, y1) in FOOT_RECTS:
        xs = np.arange(x0, x1 + 1e-9, step)
        ys = np.arange(y0, y1 + 1e-9, step)
        for x in xs:
            pts.append((float(x), float(y0)))
            pts.append((float(x), float(y1)))
        for y in ys:
            pts.append((float(x0), float(y)))
            pts.append((float(x1), float(y)))
    return np.asarray(pts, dtype=float)


BOUNDARY = _boundary()


def footprint_points(x, y, yaw, points=None):
    """把足迹采样点旋到世界系。"""
    p = BOUNDARY if points is None else points
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.stack([x + c * p[:, 0] - s * p[:, 1],
                     y + s * p[:, 0] + c * p[:, 1]], axis=1)


def clearance(x, y, yaw, boxes):
    """车心在 (x, y, yaw) 时，足迹到所有矩形的最小距离 (m)。"""
    if boxes is None or len(boxes) == 0:
        return float("inf")
    pts = footprint_points(x, y, yaw)
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    px = pts[:, 0][:, None]
    py = pts[:, 1][:, None]
    dx = np.maximum(np.maximum(b[None, :, 0] - px, 0.0), px - b[None, :, 1])
    dy = np.maximum(np.maximum(b[None, :, 2] - py, 0.0), py - b[None, :, 3])
    return float(np.sqrt(dx * dx + dy * dy).min())


def boxes_from_info(path="model/warehouse_info.json"):
    """从场景信息 JSON 生成障碍矩形（货堆/围墙/木箱）。"""
    import json

    with open(path, encoding="utf-8") as f:
        info = json.load(f)
    boxes = []
    for it in info.get("environment", []):
        kind = it.get("type")
        x, y = float(it["x"]), float(it["y"])
        if kind == "stack":
            # 托盘比货物大：0.7 × 0.8（全尺寸）。栈的 yaw=±90°，所以
            # 世界系里 x 半宽 = W/2 = 0.4、y 半宽 = L/2 = 0.35。
            sz = it.get("size", {})
            L = float(sz.get("pallet_L", 0.7))
            W = float(sz.get("pallet_W", 0.8))
            yaw90 = abs(abs(float(it.get("yaw", 90.0))) - 90.0) < 1e-6
            hx, hy = (W / 2.0, L / 2.0) if yaw90 else (L / 2.0, W / 2.0)
            boxes.append((x - hx, x + hx, y - hy, y + hy))
        elif kind == "obstacle":
            sz = it.get("size", {})
            hx = float(sz.get("L", 0.0)) / 2.0
            hy = float(sz.get("W", 0.0)) / 2.0
            yaw = abs(float(it.get("yaw", 0.0)))
            if abs(yaw - 90.0) < 1e-6:
                hx, hy = hy, hx
            boxes.append((x - hx, x + hx, y - hy, y + hy))
    return boxes


def path_clearance(pts, boxes, step=0.20):
    """折线路径（按切线朝向）走过时足迹的最小净空。"""
    pts = [np.asarray(p, dtype=float) for p in pts]
    if len(pts) < 2:
        return float("inf")
    best = float("inf")
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg = b - a
        n = max(2, int(np.hypot(seg[0], seg[1]) / step) + 1)
        yaw = float(np.arctan2(seg[1], seg[0]))
        for t in np.linspace(0.0, 1.0, n):
            p = a + t * seg
            best = min(best, clearance(float(p[0]), float(p[1]), yaw, boxes))
    return best
