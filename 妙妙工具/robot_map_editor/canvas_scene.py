"""画布场景模块 - 负责网格、坐标轴、背景绘制与图元管理。"""
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QLineF
from PyQt5.QtGui import (QBrush, QPen, QColor, QPainter, QFont, QPolygonF)
from PyQt5.QtWidgets import (QGraphicsScene, QGraphicsItem)
from items import (TYPE_SHELF, TYPE_OBSTACLE, TYPE_GOODS, TYPE_FORKLIFT,
                    TYPE_PATH_POINT, TYPE_STACK, TYPE_DROP_ZONE,
                    PathPointItem, PathEdgeItem,
                    AttachedPathPointItem, attached_point_pos,
                    ShelfItem, GoodsItem, StackItem, DropZoneItem)


class CanvasScene(QGraphicsScene):
    """画布场景：管理图元、网格、坐标轴、路径连线。"""

    # 自定义信号：属性面板修改后通知刷新（selectionChanged 用基类信号）
    property_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSceneRect(-200, -200, 400, 400)
        self.show_grid = True
        self.show_axes = True
        self.grid_step = 1.0       # 主网格 1m
        self.grid_sub = 0.5        # 次网格 0.5m
        # 路径连线和点集合
        self.edges = []
        self.path_points = {}      # pid -> PathPointItem
        self._next_pid = 0

    # ---- 背景绘制 ----
    def drawBackground(self, painter, rect):
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(245, 245, 245)))
        painter.drawRect(rect)

        # 次网格
        if self.show_grid:
            left = rect.left()
            top = rect.top()
            right = rect.right()
            bottom = rect.bottom()
            # 次网格线
            painter.setPen(QPen(QColor(220, 220, 220), 0.01))
            x = self._snap(left, self.grid_sub)
            while x <= right:
                painter.drawLine(QPointF(x, top), QPointF(x, bottom))
                x += self.grid_sub
            y = self._snap(top, self.grid_sub)
            while y <= bottom:
                painter.drawLine(QPointF(left, y), QPointF(right, y))
                y += self.grid_sub
            # 主网格线
            painter.setPen(QPen(QColor(180, 180, 180), 0.02))
            x = self._snap(left, self.grid_step)
            while x <= right:
                painter.drawLine(QPointF(x, top), QPointF(x, bottom))
                x += self.grid_step
            y = self._snap(top, self.grid_step)
            while y <= bottom:
                painter.drawLine(QPointF(left, y), QPointF(right, y))
                y += self.grid_step

        # 坐标轴
        if self.show_axes:
            # 动态计算线宽（场景坐标），使其在屏幕上保持固定像素大小
            scale_x = painter.transform().m11()  # 1 米 = scale_x 像素
            if scale_x <= 0:
                scale_x = 30.0
            # 期望屏幕像素值（场景坐标下的宽度 = 屏幕像素 / scale_x）
            axis_px = max(0.005, 2.0 / scale_x)     # 轴线宽 ≈ 2px
            tick_px = max(0.003, 1.5 / scale_x)     # 刻度线宽 ≈ 1.5px

            painter.setPen(QPen(QColor(220, 50, 50), axis_px))  # X 红
            painter.drawLine(QPointF(rect.left(), 0), QPointF(rect.right(), 0))
            painter.setPen(QPen(QColor(50, 180, 50), axis_px))  # Y 绿（场景中 -Y 向上）
            painter.drawLine(QPointF(0, rect.top()), QPointF(0, rect.bottom()))
            # 原点
            painter.setBrush(QBrush(QColor(0, 0, 0)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(0, 0), 0.06, 0.06)

            # ---- 坐标刻度数值（刻度线在场景坐标绘制）----
            tick_label_step = 2.0  # 每 2m 标一个数值
            tick_len = 0.08  # 刻度线长度（场景坐标）
            # 收集需要绘制的标签（场景坐标 -> 世界坐标）
            x_ticks = []  # [(场景x, 文本)]
            x_start = self._snap(rect.left(), tick_label_step)
            x = x_start
            while x <= rect.right():
                if abs(x) > 0.01:  # 跳过原点
                    # 刻度线
                    painter.setPen(QPen(QColor(220, 50, 50), tick_px))
                    painter.drawLine(QPointF(x, -tick_len), QPointF(x, tick_len))
                    x_ticks.append((x, f"{x:.0f}"))
                x += tick_label_step
            y_ticks = []  # [(场景y, 世界Y, 文本)]
            y_start = self._snap(rect.top(), tick_label_step)
            y = y_start
            while y <= rect.bottom():
                if abs(y) > 0.01:  # 跳过原点
                    # 刻度线
                    painter.setPen(QPen(QColor(50, 180, 50), tick_px))
                    painter.drawLine(QPointF(-tick_len, y), QPointF(tick_len, y))
                    world_y = -y
                    y_ticks.append((y, world_y, f"{world_y:.0f}"))
                y += tick_label_step

            # ---- 文字标签：切换到像素坐标，使用固定像素字体 ----
            # 这样无论画布缩放多少，标签文字都清晰可见
            saved_transform = painter.transform()
            painter.save()
            painter.resetTransform()
            # 固定像素字体
            font = QFont("Arial")
            font.setPixelSize(11)
            painter.setFont(font)
            # 标签矩形尺寸（像素）
            lbl_w_px = 30
            lbl_h_px = 14
            offset_px = 6  # 标签与轴的间距（像素）

            # X 轴标签：在 X 轴下方
            painter.setPen(QPen(QColor(180, 40, 40)))
            for scene_x, text in x_ticks:
                # 场景点 (scene_x, 0) -> 像素点
                pt = saved_transform.map(QPointF(scene_x, 0))
                rect_px = QRectF(pt.x() - lbl_w_px / 2,
                                 pt.y() + offset_px,
                                 lbl_w_px, lbl_h_px)
                painter.drawText(rect_px, Qt.AlignCenter, text)

            # Y 轴标签：在 Y 轴右侧
            painter.setPen(QPen(QColor(40, 140, 40)))
            for scene_y, world_y, text in y_ticks:
                # 场景点 (0, scene_y) -> 像素点
                pt = saved_transform.map(QPointF(0, scene_y))
                rect_px = QRectF(pt.x() + offset_px,
                                 pt.y() - lbl_h_px / 2,
                                 lbl_w_px, lbl_h_px)
                painter.drawText(rect_px, Qt.AlignCenter, text)

            # X/Y 轴名标签
            font.setPixelSize(14)
            font.setBold(True)
            painter.setFont(font)
            # X 标签在 X 轴右端下方
            pt_x = saved_transform.map(QPointF(rect.right(), 0))
            painter.setPen(QPen(QColor(220, 50, 50)))
            painter.drawText(QRectF(pt_x.x() - 24, pt_x.y() + offset_px,
                                     20, 18), Qt.AlignCenter, "X")
            # Y 标签在 Y 轴顶端右侧
            pt_y = saved_transform.map(QPointF(0, rect.top()))
            painter.setPen(QPen(QColor(50, 180, 50)))
            painter.drawText(QRectF(pt_y.x() + offset_px, pt_y.y() - 4,
                                     20, 18), Qt.AlignCenter, "Y")
            painter.restore()

    @staticmethod
    def _snap(value, step):
        return int(value / step) * step

    # ---- 路径点管理 ----
    def add_path_point(self, x, y, pid=None):
        """添加路径点，返回新 PathPointItem。"""
        if pid is None:
            pid = self.alloc_pid()
        else:
            self._next_pid = max(self._next_pid, pid + 1)
        pt = PathPointItem(pid, x, y)
        self.addItem(pt)
        self.path_points[pid] = pt
        return pt

    def alloc_pid(self):
        """分配一个新的路径点 pid。"""
        pid = self._next_pid
        self._next_pid += 1
        return pid

    def remove_path_point(self, pid):
        pt = self.path_points.pop(pid, None)
        if pt is not None:
            self.removeItem(pt)
            # 删除关联连线
            for e in list(self.edges):
                if e.src_pid == pid or e.dst_pid == pid:
                    self.removeItem(e)
                    self.edges.remove(e)

    def create_edge(self, src_pid, dst_pid):
        """创建路径连线对象但不加入场景（供命令使用）。

        同方向连线只允许存在一次：若 src_pid -> dst_pid 已存在，返回 None。
        反向（dst_pid -> src_pid）不受影响，可单独添加。
        """
        src = self.path_points.get(src_pid)
        dst = self.path_points.get(dst_pid)
        if not src or not dst:
            return None
        # 同方向去重
        for e in self.edges:
            if e.src_pid == src_pid and e.dst_pid == dst_pid:
                return None
        return PathEdgeItem(src_pid, dst_pid, src.pos(), dst.pos())

    def has_edge(self, src_pid, dst_pid):
        """是否存在指定方向的连线。"""
        return any(e.src_pid == src_pid and e.dst_pid == dst_pid
                   for e in self.edges)

    def add_edge(self, src_pid, dst_pid):
        """创建并立即加入场景（用于直接加载，不走命令栈）。"""
        edge = self.create_edge(src_pid, dst_pid)
        if edge is not None:
            self.addItem(edge)
            self.edges.append(edge)
        return edge

    def refresh_edge(self, edge):
        """当路径点位置变化时刷新连线的两个端点。"""
        src = self.path_points.get(edge.src_pid)
        dst = self.path_points.get(edge.dst_pid)
        if src and dst:
            edge.update_endpoints(src.pos(), dst.pos())

    def refresh_edges_for(self, pid):
        for e in self.edges:
            if e.src_pid == pid or e.dst_pid == pid:
                self.refresh_edge(e)

    # ---- 附属路径点（货架/货物专属，随父物体移动） ----
    def ensure_attached_for(self, parent):
        """确保父物体（货架/货物）拥有附属路径点。

        - 若已绑定且仍存在于 path_points，直接返回；
        - 否则若正后方位置已有普通路径点（容差 0.05m），则采用之；
        - 否则新建红色附属路径点并绑定。
        用于导入环境 XML / 读取路径 JSON 后重建附属关系。
        """
        pid = getattr(parent, "attached_pid", None)
        if pid is not None and pid in self.path_points:
            return self.path_points[pid]
        parent.attached_pid = None
        ax, ay = attached_point_pos(parent)
        # 采用已存在于正后方位置的普通路径点
        for opid, pt in self.path_points.items():
            if isinstance(pt, AttachedPathPointItem):
                continue
            if abs(pt.x - ax) < 0.05 and abs(pt.y - ay) < 0.05:
                parent.attached_pid = opid
                return pt
        # 新建红色附属点
        npid = self.alloc_pid()
        npt = AttachedPathPointItem(npid, parent, ax, ay)
        self.addItem(npt)
        self.path_points[npid] = npt
        parent.attached_pid = npid
        return npt

    def sync_attached_for(self, parent):
        """根据父物体当前位置/朝向同步其附属路径点位置并刷新相关连线。

        在父物体移动或属性变更后调用；若附属点引用失效则清除。
        """
        pid = getattr(parent, "attached_pid", None)
        if pid is None:
            return
        pt = self.path_points.get(pid)
        if pt is None:
            # 引用失效（例如路径被清空），清除悬挂引用
            parent.attached_pid = None
            return
        ax, ay = attached_point_pos(parent)
        pt.set_logical(ax, ay)
        self.refresh_edges_for(pid)

    def remove_attached_for(self, parent):
        """移除父物体的红色附属点及其关联连线（直接场景操作，不走命令栈）。

        用于导入环境 XML 替换物体时清理旧附属点。
        若附属点是被采用的普通路径点（绿色），则保留不动。
        """
        pid = getattr(parent, "attached_pid", None)
        if pid is None:
            return
        parent.attached_pid = None
        apt = self.path_points.get(pid)
        if apt is None or not isinstance(apt, AttachedPathPointItem):
            return
        if apt.scene() is not None:
            self.removeItem(apt)
        self.path_points.pop(pid, None)
        for e in list(self.edges):
            if e.src_pid == pid or e.dst_pid == pid:
                if e.scene() is not None:
                    self.removeItem(e)
                if e in self.edges:
                    self.edges.remove(e)

    def find_goods_at(self, x, y, exclude=None):
        """查找位于 (x, y) 容差 0.05m 内的货物（用于堆叠判定/所有权转移）。"""
        result = []
        for it in self.items():
            if not isinstance(it, GoodsItem):
                continue
            if exclude is not None and it is exclude:
                continue
            if abs(it.x - x) < 0.05 and abs(it.y - y) < 0.05:
                result.append(it)
        return result

    # ---- 元素类型过滤 ----
    def environment_items(self):
        """返回可用于导出 XML 的环境物体（不含路径点/连线）。"""
        result = []
        for it in self.items():
            if it.type() != 0:  # QGraphicsItem 子类
                pass
            if hasattr(it, "item_type") and it.item_type in (
                    TYPE_SHELF, TYPE_OBSTACLE, TYPE_GOODS, TYPE_FORKLIFT,
                    TYPE_STACK, TYPE_DROP_ZONE):
                result.append(it)
        return result

    def all_items_excluding_path(self):
        return self.environment_items()
