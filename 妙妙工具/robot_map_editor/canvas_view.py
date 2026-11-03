"""画布视图模块 - 处理缩放、平移、坐标转换。"""
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QPainter, QCursor
from PyQt5.QtWidgets import QGraphicsView


class CanvasView(QGraphicsView):
    """画布视图：滚轮缩放，中键/右键平移。"""

    # 鼠标在场景中的逻辑坐标 (世界坐标，Y 向上)
    mouse_moved = pyqtSignal(float, float)
    scale_changed = pyqtSignal(float)

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # 初始视图：以原点为中心，约显示 ±10m 范围
        self.scale_factor = 30.0   # 1m = 30 像素
        self.scale(self.scale_factor, self.scale_factor)
        self.centerOn(QPointF(0, 0))

        self._panning = False
        self._pan_last = None
        self._prev_drag_mode = None
        self._prev_anchor = None

    # ---- 缩放 ----
    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            # Ctrl+滚轮大幅缩放
            factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        else:
            factor = 1.1 if event.angleDelta().y() > 0 else 1 / 1.1
        self.apply_zoom(factor)

    def apply_zoom(self, factor):
        new_scale = self.scale_factor * factor
        if new_scale < 2.0 or new_scale > 2000.0:
            return
        self.scale_factor = new_scale
        self.scale(factor, factor)
        self.scale_changed.emit(self.scale_factor)

    # ---- 平移：中键或右键拖拽 ----
    # 使用增量式视图坐标差 + NoAnchor，保证 1:1 跟手、与缩放比例无关，
    # 避免 AnchorUnderMouse 与手动 translate 叠加补偿导致跳动/灵敏度过高。
    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton or event.button() == Qt.RightButton:
            self._panning = True
            self._pan_last = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            # 临时禁用拖拽模式与变换锚点
            self._prev_drag_mode = self.dragMode()
            self.setDragMode(QGraphicsView.NoDrag)
            self._prev_anchor = self.transformationAnchor()
            self.setTransformationAnchor(QGraphicsView.NoAnchor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            # 增量式平移：按视图像素差移动。
            # translate(dx,dy) 会与缩放矩阵叠加相乘（dx 被放大 scale 倍），
            # 故需除以 scale_factor 抵消，得到 1:1 跟手、与缩放比例无关。
            delta = event.pos() - self._pan_last
            s = self.scale_factor if self.scale_factor else 1.0
            self.translate(delta.x() / s, delta.y() / s)
            self._pan_last = event.pos()
            event.accept()
            return
        super().mouseMoveEvent(event)
        # 发送鼠标的世界坐标
        scene_pt = self.mapToScene(event.pos())
        self.mouse_moved.emit(float(scene_pt.x()), -float(scene_pt.y()))

    def mouseReleaseEvent(self, event):
        if self._panning and (event.button() == Qt.MiddleButton or
                              event.button() == Qt.RightButton):
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            self.setDragMode(getattr(self, "_prev_drag_mode",
                                      QGraphicsView.RubberBandDrag))
            self.setTransformationAnchor(getattr(
                self, "_prev_anchor", QGraphicsView.AnchorUnderMouse))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ---- 工具切换时禁用橡皮筋选择（放置模式） ----
    def set_placement_mode(self, on):
        if on:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.RubberBandDrag)
            self.setCursor(Qt.ArrowCursor)
