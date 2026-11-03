"""图元类模块 - 所有可在画布上放置的元素的图形表示。

坐标系约定：
- 逻辑坐标：X 向右，Y 向上，Z 向上（机器人坐标系）
- 场景坐标：Qt 默认 X 向右，Y 向下；为兼容 Y 向上的习惯，
  BaseItem.setPos 时把逻辑 Y 取反存入场景坐标。
- Yaw：单位为度，逆时针为正（机器人约定）。
  绘制时调用 setRotation(-yaw)，因 Qt 场景 Y 向下时旋转方向相反。
"""
import math
from PyQt5.QtCore import Qt, QRectF, QPointF, QLineF
from PyQt5.QtGui import (QBrush, QPen, QColor, QPainter, QFont, QPolygonF,
                          QPainterPath, QTransform)
from PyQt5.QtWidgets import (QGraphicsObject, QGraphicsItem,
                              QGraphicsLineItem)

# ---- 元素类型常量 ----
TYPE_SHELF = "shelf"
TYPE_OBSTACLE = "obstacle"
TYPE_GOODS = "goods"
TYPE_FORKLIFT = "forklift"
TYPE_PATH_POINT = "path_point"
TYPE_STACK = "stack"          # 货堆（N 层托盘+货物）+ 27 号起的主目标类型
TYPE_DROP_ZONE = "drop_zone"  # 地面指定放置区（贴地标记，无碰撞）


# ----------------------------------------------------------------------
# 基类
# ----------------------------------------------------------------------
class BaseItem(QGraphicsObject):
    """所有地图图元的基类，保存逻辑坐标 (x, y, z) 与 yaw。"""

    def __init__(self, item_type, x=0.0, y=0.0, z=0.0, yaw=0.0, name=""):
        super().__init__()
        self.item_type = item_type
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.yaw = float(yaw)
        self.name = name or item_type
        # 附属路径点 pid（货架/货物专属，None 表示无）；详见 AttachedPathPointItem
        self.attached_pid = None
        # 同步场景位置（Y 取反）
        self.setPos(self.x, -self.y)
        self.setRotation(-self.yaw)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        # Z 值用于绘制层级（数值越大越在上）
        self.setZValue(10)

    # ---- 逻辑坐标与场景坐标同步 ----
    def update_logical_from_scene(self):
        """拖拽结束后调用，从场景 pos 同步逻辑坐标。"""
        self.x = float(self.pos().x())
        self.y = -float(self.pos().y())
        self.yaw = -float(self.rotation())

    def set_logical(self, x=None, y=None, z=None, yaw=None):
        """从属性面板修改坐标/yaw 时调用。"""
        if x is not None:
            self.x = float(x)
        if y is not None:
            self.y = float(y)
        if z is not None:
            self.z = float(z)
        if yaw is not None:
            self.yaw = float(yaw)
            self.setRotation(-self.yaw)
        self.setPos(self.x, -self.y)
        self.update()

    # ---- 供子类使用的辅助：绘制选中高亮 ----
    def _draw_selected(self, painter):
        if self.isSelected():
            painter.setPen(QPen(QColor(255, 165, 0), 0.05, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.boundingRect().adjusted(-0.05, -0.05, 0.05, 0.05))

    def itemChange(self, change, value):
        # 选中变化时重绘以显示高亮
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.update()
        return super().itemChange(change, value)

    # ---- 属性面板使用 ----
    def get_properties(self):
        """返回属性面板需显示的 (属性名, 当前值, 单位/提示) 列表。"""
        return [
            ("name", self.name, "str"),
            ("x", self.x, "m"),
            ("y", self.y, "m"),
            ("z", self.z, "m"),
            ("yaw", self.yaw, "deg"),
        ]

    def set_property(self, name, value):
        """属性面板修改单个属性。子类可重写以处理自定义属性。"""
        if name == "name":
            self.name = str(value)
            self.update()
        elif name == "x":
            self.set_logical(x=value)
        elif name == "y":
            self.set_logical(y=value)
        elif name == "z":
            self.set_logical(z=value)
        elif name == "yaw":
            self.set_logical(yaw=value)


# ----------------------------------------------------------------------
# 货架
# ----------------------------------------------------------------------
class ShelfItem(BaseItem):
    """货架：2D 俯视为 0.8m × 2.0m 矩形，四角小方块表示立柱。"""

    DEFAULT_L = 0.8
    DEFAULT_W = 2.0
    DEFAULT_H = 2.0

    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0,
                 L=DEFAULT_L, W=DEFAULT_W, H=DEFAULT_H, name="shelf"):
        self.L = float(L)
        self.W = float(W)
        self.H = float(H)
        super().__init__(TYPE_SHELF, x, y, z, yaw, name)

    def boundingRect(self):
        return QRectF(-self.L / 2 - 0.05, -self.W / 2 - 0.05,
                      self.L + 0.1, self.W + 0.1)

    def paint(self, painter, option, widget=None):
        rect = QRectF(-self.L / 2, -self.W / 2, self.L, self.W)
        # 主体填充（棕色半透明）
        painter.setBrush(QBrush(QColor(139, 90, 43, 120)))
        painter.setPen(QPen(QColor(80, 50, 20), 0.03))
        painter.drawRect(rect)
        # 四角立柱（深色小方块）
        post = 0.08
        painter.setBrush(QBrush(QColor(80, 50, 20)))
        painter.setPen(Qt.NoPen)
        for sx in (-1, 1):
            for sy in (-1, 1):
                painter.drawRect(QRectF(sx * (self.L / 2 - post / 2) - post / 2,
                                          sy * (self.W / 2 - post / 2) - post / 2,
                                          post, post))
        # 标签
        self._draw_label(painter, self.name)
        self._draw_selected(painter)

    def _draw_label(self, painter, text):
        font = QFont("Arial")
        font.setPointSizeF(0.18)
        painter.setFont(font)
        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawText(QRectF(-self.L / 2, -self.W / 2, self.L, self.W),
                        Qt.AlignCenter, text)

    def get_properties(self):
        return super().get_properties() + [
            ("L", self.L, "m"),
            ("W", self.W, "m"),
            ("H", self.H, "m"),
        ]

    def set_property(self, name, value):
        if name == "L":
            self.L = float(value)
            self.update()
        elif name == "W":
            self.W = float(value)
            self.update()
        elif name == "H":
            self.H = float(value)
            self.update()
        else:
            super().set_property(name, value)


# ----------------------------------------------------------------------
# 障碍物（仅矩形）
# ----------------------------------------------------------------------
class ObstacleItem(BaseItem):
    """矩形障碍物。"""

    DEFAULT_L = 0.5
    DEFAULT_W = 0.5
    DEFAULT_H = 0.5

    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0,
                 L=DEFAULT_L, W=DEFAULT_W, H=DEFAULT_H, name="obstacle"):
        self.L = float(L)
        self.W = float(W)
        self.H = float(H)
        super().__init__(TYPE_OBSTACLE, x, y, z, yaw, name)

    def boundingRect(self):
        return QRectF(-self.L / 2 - 0.05, -self.W / 2 - 0.05,
                      self.L + 0.1, self.W + 0.1)

    def paint(self, painter, option, widget=None):
        rect = QRectF(-self.L / 2, -self.W / 2, self.L, self.W)
        painter.setBrush(QBrush(QColor(180, 60, 60, 160)))
        painter.setPen(QPen(QColor(120, 30, 30), 0.03))
        painter.drawRect(rect)
        # 斜线纹理
        painter.setPen(QPen(QColor(120, 30, 30, 120), 0.02))
        step = 0.1
        d = 0.0
        clip_path = QPainterPath()
        clip_path.addRect(rect)
        painter.save()
        painter.setClipPath(clip_path)
        max_d = self.L + self.W
        while d < max_d:
            painter.drawLine(QPointF(-self.L / 2 + d, -self.W / 2),
                              QPointF(-self.L / 2 + d - self.W, self.W / 2))
            d += step
        painter.restore()
        self._draw_label(painter, self.name)
        self._draw_selected(painter)

    def _draw_label(self, painter, text):
        font = QFont("Arial")
        font.setPointSizeF(0.15)
        painter.setFont(font)
        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawText(QRectF(-self.L / 2, -self.W / 2, self.L, self.W),
                        Qt.AlignCenter, text)

    def get_properties(self):
        return super().get_properties() + [
            ("L", self.L, "m"),
            ("W", self.W, "m"),
            ("H", self.H, "m"),
        ]

    def set_property(self, name, value):
        if name == "L":
            self.L = float(value)
            self.update()
        elif name == "W":
            self.W = float(value)
            self.update()
        elif name == "H":
            self.H = float(value)
            self.update()
        else:
            super().set_property(name, value)


# ----------------------------------------------------------------------
# 货物 + 托盘
# ----------------------------------------------------------------------
class GoodsItem(BaseItem):
    """货物（含托盘）：2D 俯视为 0.7m × 0.8m 矩形。

    属性：
      - bottom_z: 托盘底面距地高度（用于堆叠）
      - layer: 层数（由 bottom_z 推算或直接设置）
    """

    PALLET_L = 0.7
    PALLET_W = 0.8
    PALLET_H = 0.15      # 托盘总高（顶板+支撑）
    CARGO_L = 0.62
    CARGO_W = 0.8
    CARGO_H = 0.3
    LAYER_STEP = 0.55   # 每加一层底面抬升量

    def __init__(self, x=0.0, y=0.0, bottom_z=0.0, yaw=0.0,
                 name="goods"):
        self.bottom_z = float(bottom_z)
        # 货物视角取较大尺寸绘制
        self.L = self.PALLET_L
        self.W = self.PALLET_W
        # z 字段保存托盘 body z（= bottom_z + 0.075）
        super().__init__(TYPE_GOODS, x, y, z=self.bottom_z + 0.075,
                          yaw=yaw, name=name)
        self.setZValue(20)  # 货物绘制在货架之上

    @property
    def layer(self):
        return int(round(self.bottom_z / self.LAYER_STEP)) + 1

    @layer.setter
    def layer(self, n):
        self.bottom_z = (int(n) - 1) * self.LAYER_STEP
        self.z = self.bottom_z + 0.075

    @property
    def cargo_z(self):
        """货物 body z（= 底面 + 0.4，与用户 XML 一致）。"""
        return self.bottom_z + 0.4

    @property
    def pallet_z(self):
        """托盘 body z（= 底面 + 0.075）。"""
        return self.bottom_z + 0.075

    def boundingRect(self):
        return QRectF(-self.L / 2 - 0.05, -self.W / 2 - 0.05,
                      self.L + 0.1, self.W + 0.1)

    def paint(self, painter, option, widget=None):
        rect = QRectF(-self.L / 2, -self.W / 2, self.L, self.W)
        # 透明度随层数降低（提示堆叠）
        alpha = max(60, 220 - (self.layer - 1) * 50)
        # 托盘底色
        painter.setBrush(QBrush(QColor(120, 80, 40, alpha)))
        painter.setPen(QPen(QColor(60, 40, 20), 0.03))
        painter.drawRect(rect)
        # 货物（中心稍小矩形）
        cargo_rect = QRectF(-self.CARGO_L / 2, -self.CARGO_W / 2,
                            self.CARGO_L, self.CARGO_W)
        painter.setBrush(QBrush(QColor(220, 140, 60, alpha)))
        painter.setPen(QPen(QColor(120, 70, 20), 0.03))
        painter.drawRect(cargo_rect)
        # 层数标签
        font = QFont("Arial")
        font.setPointSizeF(0.18)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawText(rect, Qt.AlignCenter, f"L{self.layer}\n{self.name}")
        self._draw_selected(painter)

    def get_properties(self):
        return [
            ("name", self.name, "str"),
            ("x", self.x, "m"),
            ("y", self.y, "m"),
            ("bottom_z", self.bottom_z, "m"),
            ("layer", self.layer, "int"),
            ("yaw", self.yaw, "deg"),
            ("L(托盘)", self.L, "m"),
            ("W(托盘)", self.W, "m"),
        ]

    def set_property(self, name, value):
        if name == "bottom_z":
            self.bottom_z = float(value)
            self.z = self.bottom_z + 0.075
            self.update()
        elif name == "layer":
            self.layer = int(value)
            self.update()
        else:
            super().set_property(name, value)


# ----------------------------------------------------------------------
# 叉车初始位置框
# ----------------------------------------------------------------------
class ForkliftItem(BaseItem):
    """叉车初始位置框：2.34m × 1.2m 矩形，前端三角形指示朝向。

    前端短边不绘制（开放），用三角形指向 yaw 方向。
    """

    L = 2.34
    W = 1.2
    H = 0.3  # 底盘高度

    def __init__(self, x=-6.0, y=0.0, z=0.0, yaw=0.0, name="forklift_init"):
        super().__init__(TYPE_FORKLIFT, x, y, z, yaw, name)
        self.setZValue(30)  # 显示在最上层

    def boundingRect(self):
        return QRectF(-self.L / 2 - 0.1, -self.W / 2 - 0.1,
                      self.L + 0.2, self.W + 0.2)

    def paint(self, painter, option, widget=None):
        L, W = self.L, self.W
        # 半透明填充
        painter.setBrush(QBrush(QColor(60, 130, 220, 80)))
        painter.setPen(QPen(QColor(30, 80, 180), 0.04))
        painter.drawRect(QRectF(-L / 2, -W / 2, L, W))
        # 三条边（左、右、后），前端开放
        painter.setPen(QPen(QColor(30, 80, 180), 0.06))
        painter.setBrush(Qt.NoBrush)
        path = QPainterPath()
        path.moveTo(L / 2, -W / 2)
        path.lineTo(-L / 2, -W / 2)
        path.lineTo(-L / 2, W / 2)
        path.lineTo(L / 2, W / 2)
        painter.drawPath(path)
        # 前端三角形（指向 +X，即 yaw=0 时朝右）
        tri = QPolygonF([
            QPointF(L / 2, 0.0),
            QPointF(L / 2 - 0.4, -W / 2 * 0.5),
            QPointF(L / 2 - 0.4, W / 2 * 0.5),
        ])
        painter.setBrush(QBrush(QColor(30, 80, 180)))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(tri)
        # 标签
        font = QFont("Arial")
        font.setPointSizeF(0.16)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawText(QRectF(-L / 2, -W / 2, L * 0.7, W),
                        Qt.AlignCenter, self.name)
        self._draw_selected(painter)

    def get_properties(self):
        return [
            ("name", self.name, "str"),
            ("x", self.x, "m"),
            ("y", self.y, "m"),
            ("z", self.z, "m"),
            ("yaw", self.yaw, "deg"),
        ]


# ----------------------------------------------------------------------
# 路径点
# ----------------------------------------------------------------------
class PathPointItem(QGraphicsObject):
    """路径点：小圆点带序号。不导出至环境 XML。"""

    RADIUS = 0.15

    def __init__(self, pid, x=0.0, y=0.0):
        super().__init__()
        self.pid = int(pid)
        self.x = float(x)
        self.y = float(y)
        self.setPos(self.x, -self.y)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setZValue(100)

    def update_logical_from_scene(self):
        self.x = float(self.pos().x())
        self.y = -float(self.pos().y())

    def set_logical(self, x=None, y=None):
        if x is not None:
            self.x = float(x)
        if y is not None:
            self.y = float(y)
        self.setPos(self.x, -self.y)
        self.update()

    def boundingRect(self):
        return QRectF(-self.RADIUS - 0.05, -self.RADIUS - 0.05,
                      (self.RADIUS + 0.05) * 2, (self.RADIUS + 0.05) * 2)

    def paint(self, painter, option, widget=None):
        painter.setBrush(QBrush(QColor(40, 180, 80)))
        painter.setPen(QPen(QColor(20, 100, 40), 0.03))
        painter.drawEllipse(QPointF(0, 0), self.RADIUS, self.RADIUS)
        font = QFont("Arial")
        font.setPointSizeF(0.15)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(QRectF(-self.RADIUS, -self.RADIUS,
                                 self.RADIUS * 2, self.RADIUS * 2),
                        Qt.AlignCenter, str(self.pid))

    def get_properties(self):
        return [
            ("pid", self.pid, "int"),
            ("x", self.x, "m"),
            ("y", self.y, "m"),
        ]


# ----------------------------------------------------------------------
# 附属路径点（货架/货物专属，红色，随父物体移动）
# ----------------------------------------------------------------------
ATTACHED_DISTANCE = 2.85  # 附属点距父物体正后方的距离（m）


def attached_point_pos(parent):
    """计算父物体正后方 ATTACHED_DISTANCE 米处的世界坐标 (x, y)。

    面朝方向 = (cos yaw, sin yaw)（机器人约定，Y 向上、逆时针为正），
    正后方即反方向。
    """
    yaw_rad = math.radians(parent.yaw)
    return (parent.x - ATTACHED_DISTANCE * math.cos(yaw_rad),
            parent.y - ATTACHED_DISTANCE * math.sin(yaw_rad))


class AttachedPathPointItem(PathPointItem):
    """附属路径点：红色小球，随货架/货物移动，不可独立移动/选中。

    - 颜色为红色，区别于普通路径点（绿色）；
    - 不可移动、不可选中（随父物体移动，由场景同步位置）；
    - 仍是 PathPointItem 子类，故可在 path_points 字典中、可参与连线。
    """

    def __init__(self, pid, parent_item, x=0.0, y=0.0):
        super().__init__(pid, x, y)
        self.parent_item = parent_item
        # 附属点不可独立移动/选中，避免被拖拽或误删
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        # 层级低于普通路径点(100)，高于连线(90)
        self.setZValue(95)

    def paint(self, painter, option, widget=None):
        painter.setBrush(QBrush(QColor(220, 40, 40)))
        painter.setPen(QPen(QColor(120, 20, 20), 0.03))
        painter.drawEllipse(QPointF(0, 0), self.RADIUS, self.RADIUS)
        font = QFont("Arial")
        font.setPointSizeF(0.15)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(QRectF(-self.RADIUS, -self.RADIUS,
                                 self.RADIUS * 2, self.RADIUS * 2),
                        Qt.AlignCenter, str(self.pid))


# ----------------------------------------------------------------------
# 路径连线（带箭头）
# ----------------------------------------------------------------------
class PathEdgeItem(QGraphicsLineItem):
    """路径连线：从一个 PathPointItem 指向另一个，带箭头。"""

    SELECT_WIDTH = 0.2  # 命中测试的带宽（场景单位 m）

    def __init__(self, src_pid, dst_pid, src_pos, dst_pos):
        super().__init__()
        self.src_pid = int(src_pid)
        self.dst_pid = int(dst_pid)
        # 场景坐标（Y 已取反）
        self.setLine(QLineF(src_pos.x(), src_pos.y(),
                            dst_pos.x(), dst_pos.y()))
        pen = QPen(QColor(40, 180, 80), 0.06)
        pen.setCapStyle(Qt.RoundCap)
        self.setPen(pen)
        self.setZValue(90)
        # 允许选中
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

    def update_endpoints(self, src_pos, dst_pos):
        self.setLine(QLineF(src_pos.x(), src_pos.y(),
                            dst_pos.x(), dst_pos.y()))

    def boundingRect(self):
        r = super().boundingRect().adjusted(-self.SELECT_WIDTH, -self.SELECT_WIDTH,
                                            self.SELECT_WIDTH, self.SELECT_WIDTH)
        return r

    def shape(self):
        """加宽命中区域，便于用鼠标点选细线。"""
        path = QPainterPath()
        line = self.line()
        dx, dy = line.dx(), line.dy()
        length = math.hypot(dx, dy)
        if length < 1e-9:
            path.moveTo(line.p1())
            return path
        # 单位法向量
        nx, ny = -dy / length, dx / length
        w = self.SELECT_WIDTH
        poly = QPolygonF([
            QPointF(line.p1().x() + nx * w, line.p1().y() + ny * w),
            QPointF(line.p2().x() + nx * w, line.p2().y() + ny * w),
            QPointF(line.p2().x() - nx * w, line.p2().y() - ny * w),
            QPointF(line.p1().x() - nx * w, line.p1().y() - ny * w),
        ])
        path.addPolygon(poly)
        path.closeSubpath()
        return path

    def paint(self, painter, option, widget=None):
        # 选中时先绘制高亮底色
        if self.isSelected():
            painter.setPen(QPen(QColor(255, 165, 0), 0.14))
            painter.drawLine(self.line())
        super().paint(painter, option, widget)
        # 绘制箭头
        line = self.line()
        angle = math.atan2(line.dy(), line.dx())
        arrow_size = 0.35
        p1 = line.p2()
        p2 = QPointF(p1.x() - arrow_size * math.cos(angle - math.pi / 6),
                     p1.y() - arrow_size * math.sin(angle - math.pi / 6))
        p3 = QPointF(p1.x() - arrow_size * math.cos(angle + math.pi / 6),
                     p1.y() - arrow_size * math.sin(angle + math.pi / 6))
        arrow = QPolygonF([p1, p2, p3])
        painter.setBrush(QBrush(QColor(40, 180, 80)))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(arrow)


# ----------------------------------------------------------------------
# 货堆（27 号起的主目标类型：一摞托盘+货物，可叠 1~3 层）
# ----------------------------------------------------------------------
class StackItem(BaseItem):
    """货堆：2D 俯视为 0.8m(X) × 0.7m(Y)（yaw=±90 时），按层叠绘制。

    导出的 MuJoCo body 命名（FSM 与路径规划器按名字取"最下方货物"）：
        stack_<ID>_L<i>_pallet   z = bottom_z(i) + 0.075
        stack_<ID>_L<i>_cargo    z = bottom_z(i) + 0.4
    层距 LAYER_STEP = 0.55（名义值；实测承托面由 tools/calibrate_heights.py 标定）。
    """

    PALLET_L = 0.7
    PALLET_W = 0.8
    PALLET_H = 0.15
    CARGO_L = 0.62
    CARGO_W = 0.8
    CARGO_H = 0.3
    LAYER_STEP = 0.52   # 与训练环境（train_mohu_10._set_goods_layers 的 i*0.52）一致
    MAX_LAYERS = 3          # 叉车起升上限决定的最高堆层

    def __init__(self, x=0.0, y=0.0, layers=3, yaw=90.0, name="stack",
                 z=0.0):
        self.layers = max(1, int(layers))
        # 俯视绘制按托盘外框（yaw=90 时长边在 X）
        self.L = self.PALLET_W
        self.W = self.PALLET_L
        super().__init__(TYPE_STACK, x, y, z, yaw, name)
        self.setZValue(15)

    # ---- 几何 ----
    @property
    def full_name(self):
        return self.name if self.name.startswith("stack_") else f"stack_{self.name}"

    def bottom_z(self, i):
        """第 i 层（1 起）底面高度。"""
        return (int(i) - 1) * self.LAYER_STEP

    def pallet_z(self, i):
        return self.bottom_z(i) + 0.075

    def cargo_z(self, i):
        return self.bottom_z(i) + 0.4

    @property
    def top_z(self):
        """名义顶面高度（= 最上层货物顶面）。"""
        return self.LAYER_STEP * self.layers

    def boundingRect(self):
        return QRectF(-self.L / 2 - 0.05, -self.W / 2 - 0.05,
                      self.L + 0.1, self.W + 0.1)

    def paint(self, painter, option, widget=None):
        rect = QRectF(-self.L / 2, -self.W / 2, self.L, self.W)
        # 每层稍微内缩，视觉上表现"堆"
        for i in range(self.layers):
            shrink = 0.035 * i
            r = rect.adjusted(shrink, shrink, -shrink, -shrink)
            alpha = max(90, 230 - i * 45)
            painter.setBrush(QBrush(QColor(202, 139, 75, alpha)))
            painter.setPen(QPen(QColor(91, 58, 26), 0.03))
            painter.drawRect(r)
        # 托盘外框
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(60, 40, 20), 0.04))
        painter.drawRect(rect)
        # 层数 + 名称
        font = QFont("Arial")
        font.setPointSizeF(0.16)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(40, 20, 0)))
        painter.drawText(rect, Qt.AlignCenter,
                         f"{self.name}\n{self.layers}层")
        self._draw_selected(painter)

    def get_properties(self):
        return [
            ("name", self.name, "str"),
            ("x", self.x, "m"),
            ("y", self.y, "m"),
            ("layers", self.layers, "int"),
            ("yaw", self.yaw, "deg"),
        ]

    def set_property(self, name, value):
        if name == "layers":
            self.layers = max(1, min(self.MAX_LAYERS, int(value)))
            self.update()
        else:
            super().set_property(name, value)


# ----------------------------------------------------------------------
# 地面指定放置区（贴地标记，无碰撞，仅表示"目标位置可以是地面区域"）
# ----------------------------------------------------------------------
class DropZoneItem(BaseItem):
    """地面放置区：默认 1.2m × 1.2m 的贴地标记。

    导出为 `drop_zone_<ID>`：一个薄板 box，contype=0 conaffinity=0（不参与碰撞、不挡雷达）。
    """

    DEFAULT_L = 1.2
    DEFAULT_W = 1.2
    H = 0.01

    def __init__(self, x=0.0, y=0.0, yaw=90.0, L=DEFAULT_L, W=DEFAULT_W,
                 name="G1", z=0.0):
        self.L = float(L)
        self.W = float(W)
        super().__init__(TYPE_DROP_ZONE, x, y, z, yaw, name)
        self.setZValue(5)

    @property
    def full_name(self):
        return self.name if self.name.startswith("drop_zone_") \
            else f"drop_zone_{self.name}"

    def boundingRect(self):
        return QRectF(-self.L / 2 - 0.05, -self.W / 2 - 0.05,
                      self.L + 0.1, self.W + 0.1)

    def paint(self, painter, option, widget=None):
        rect = QRectF(-self.L / 2, -self.W / 2, self.L, self.W)
        painter.setBrush(QBrush(QColor(120, 220, 120, 90)))
        pen = QPen(QColor(20, 120, 20), 0.05)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(rect)
        font = QFont("Arial")
        font.setPointSizeF(0.16)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(10, 90, 10)))
        painter.drawText(rect, Qt.AlignCenter, f"地面区\n{self.name}")
        self._draw_selected(painter)

    def get_properties(self):
        return [
            ("name", self.name, "str"),
            ("x", self.x, "m"),
            ("y", self.y, "m"),
            ("L", self.L, "m"),
            ("W", self.W, "m"),
            ("yaw", self.yaw, "deg"),
        ]

    def set_property(self, name, value):
        if name == "L":
            self.L = float(value)
            self.update()
        elif name == "W":
            self.W = float(value)
            self.update()
        else:
            super().set_property(name, value)


# 需要红色附属路径点的父物体类型（货架保留只为兼容旧场景文件）
ATTACHABLE_TYPES = (ShelfItem, GoodsItem, StackItem, DropZoneItem)
