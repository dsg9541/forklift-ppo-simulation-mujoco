"""撤销/重做命令模块 - 基于 QUndoStack/QUndoCommand。

所有命令的 redo/undo 都设计为幂等：先检查 item 是否已在场景中再决定 add/remove，
避免外部代码已经预先操作过场景时产生 "item has already been added" 警告。
"""
from PyQt5.QtWidgets import QUndoCommand
from items import (PathPointItem, PathEdgeItem, ShelfItem, GoodsItem,
                    AttachedPathPointItem, attached_point_pos)


def _safe_add(scene, item):
    if item.scene() is None:
        scene.addItem(item)
        return True
    return False


def _safe_remove(scene, item):
    if item.scene() is not None:
        scene.removeItem(item)
        return True
    return False


class AddItemCommand(QUndoCommand):
    """添加图元到场景（幂等）。

    可选地为货架/货物创建附属路径点（create_attached=True）。
    附属点生命周期与本次添加绑定：redo 创建，undo 移除。
    """

    def __init__(self, scene, item, text="添加图元", create_attached=False):
        super().__init__(text)
        self.scene = scene
        self.item = item
        self.create_attached = create_attached
        self._attached_pt = None  # 懒创建的附属点

    def _maybe_make_attached(self):
        if self._attached_pt is not None:
            return
        if not self.create_attached:
            return
        if not isinstance(self.item, (ShelfItem, GoodsItem)):
            return
        ax, ay = attached_point_pos(self.item)
        pid = self.scene.alloc_pid()
        self._attached_pt = AttachedPathPointItem(pid, self.item, ax, ay)

    def redo(self):
        _safe_add(self.scene, self.item)
        if isinstance(self.item, PathPointItem):
            self.scene.path_points[self.item.pid] = self.item
        self._maybe_make_attached()
        if self._attached_pt is not None:
            _safe_add(self.scene, self._attached_pt)
            self.scene.path_points[self._attached_pt.pid] = self._attached_pt
            self.item.attached_pid = self._attached_pt.pid

    def undo(self):
        # 先移除附属点及其关联连线
        if self._attached_pt is not None:
            apid = self._attached_pt.pid
            _safe_remove(self.scene, self._attached_pt)
            self.scene.path_points.pop(apid, None)
            for e in list(self.scene.edges):
                if e.src_pid == apid or e.dst_pid == apid:
                    _safe_remove(self.scene, e)
                    if e in self.scene.edges:
                        self.scene.edges.remove(e)
            self.item.attached_pid = None
        _safe_remove(self.scene, self.item)
        # 若是路径点，从字典移除并清理关联连线
        if isinstance(self.item, PathPointItem):
            self.scene.path_points.pop(self.item.pid, None)
            for e in list(self.scene.edges):
                if e.src_pid == self.item.pid or e.dst_pid == self.item.pid:
                    _safe_remove(self.scene, e)
                    if e in self.scene.edges:
                        self.scene.edges.remove(e)


class RemoveItemsCommand(QUndoCommand):
    """从场景删除一组图元（含关联路径连线与附属点），幂等。

    支持的图元类型：
      - PathPointItem：从场景移除并从 path_points 字典移除，同时移除其关联连线；
      - PathEdgeItem：从场景移除并从 edges 列表移除（用户显式选中删除）；
      - ShelfItem/GoodsItem：若拥有附属点则一并处理——
            * 货物堆叠：同位置仍有未删除货物时，将附属点所有权转移给后者（附属点保留）；
            * 否则随父物体删除附属点及其关联连线。
      - 其他环境物体：仅从场景移除。
    """

    def __init__(self, scene, items, text="删除图元"):
        super().__init__(text)
        self.scene = scene
        self.items = list(items)
        # 收集随路径点自动删除的关联连线（排除已显式选中的）
        self.assoc_edges = []
        for it in self.items:
            if isinstance(it, PathPointItem):
                for edge in list(scene.edges):
                    if (edge.src_pid == it.pid or
                            edge.dst_pid == it.pid):
                        if (edge not in self.assoc_edges and
                                edge not in self.items):
                            self.assoc_edges.append(edge)
        # 附属点处理：随父删除 或 所有权转移（货物堆叠）
        self.attached_remove = []    # [(parent, attached_pt, [edges])]
        self.attached_transfer = []  # [(parent, attached_pid, new_owner)]
        for it in self.items:
            apid = getattr(it, "attached_pid", None)
            if apid is None:
                continue
            apt = scene.path_points.get(apid)
            if apt is None:
                continue
            # 货物：同位置若仍有未删除货物，则转移所有权
            if isinstance(it, GoodsItem) and hasattr(scene, "find_goods_at"):
                new_owner = None
                for cand in scene.find_goods_at(it.x, it.y, exclude=it):
                    if cand not in self.items:
                        new_owner = cand
                        break
                if new_owner is not None:
                    self.attached_transfer.append((it, apid, new_owner))
                    continue
            # 否则随父物体删除附属点
            edges = [e for e in list(scene.edges)
                     if e.src_pid == apid or e.dst_pid == apid]
            self.attached_remove.append((it, apt, edges))

    def redo(self):
        # 移除自动关联连线
        for e in self.assoc_edges:
            _safe_remove(self.scene, e)
            if e in self.scene.edges:
                self.scene.edges.remove(e)
        # 转移附属点所有权（附属点本身保留）
        for parent, apid, new_owner in self.attached_transfer:
            new_owner.attached_pid = apid
            parent.attached_pid = None
        # 移除随父物体删除的附属点
        for parent, apt, edges in self.attached_remove:
            for e in edges:
                _safe_remove(self.scene, e)
                if e in self.scene.edges:
                    self.scene.edges.remove(e)
            _safe_remove(self.scene, apt)
            self.scene.path_points.pop(apt.pid, None)
            parent.attached_pid = None
        # 移除显式选中的图元
        for it in self.items:
            _safe_remove(self.scene, it)
            if isinstance(it, PathPointItem):
                self.scene.path_points.pop(it.pid, None)
            elif isinstance(it, PathEdgeItem):
                if it in self.scene.edges:
                    self.scene.edges.remove(it)

    def undo(self):
        # 恢复显式选中的图元
        for it in self.items:
            _safe_add(self.scene, it)
            if isinstance(it, PathPointItem):
                self.scene.path_points[it.pid] = it
            elif isinstance(it, PathEdgeItem):
                if it not in self.scene.edges:
                    self.scene.edges.append(it)
        # 恢复随父物体删除的附属点
        for parent, apt, edges in self.attached_remove:
            _safe_add(self.scene, apt)
            self.scene.path_points[apt.pid] = apt
            parent.attached_pid = apt.pid
            for e in edges:
                if _safe_add(self.scene, e):
                    if e not in self.scene.edges:
                        self.scene.edges.append(e)
        # 撤销所有权转移
        for parent, apid, new_owner in self.attached_transfer:
            parent.attached_pid = apid
            new_owner.attached_pid = None
        # 恢复自动关联连线
        for e in self.assoc_edges:
            if _safe_add(self.scene, e):
                if e not in self.scene.edges:
                    self.scene.edges.append(e)


class MoveItemCommand(QUndoCommand):
    """图元移动（拖拽）。保存移动前后的场景位置。"""

    def __init__(self, item, old_pos, new_pos, text="移动图元"):
        super().__init__(text)
        self.item = item
        self.old_pos = old_pos
        self.new_pos = new_pos

    def _apply(self, pos):
        self.item.setPos(pos)
        if hasattr(self.item, "update_logical_from_scene"):
            self.item.update_logical_from_scene()
        self.item.update()
        scene = self.item.scene()
        if scene:
            if isinstance(self.item, PathPointItem):
                if hasattr(scene, "refresh_edges_for"):
                    scene.refresh_edges_for(self.item.pid)
            # 货架/货物移动后同步其附属点
            if hasattr(scene, "sync_attached_for"):
                scene.sync_attached_for(self.item)

    def undo(self):
        self._apply(self.old_pos)

    def redo(self):
        self._apply(self.new_pos)


class PropertyChangeCommand(QUndoCommand):
    """属性面板修改单个属性。"""

    def __init__(self, item, prop_name, old_value, new_value,
                 text="修改属性"):
        super().__init__(text)
        self.item = item
        self.prop_name = prop_name
        self.old_value = old_value
        self.new_value = new_value

    def undo(self):
        self.item.set_property(self.prop_name, self.old_value)
        self._notify()

    def redo(self):
        self.item.set_property(self.prop_name, self.new_value)
        self._notify()

    def _notify(self):
        scene = self.item.scene()
        # 用自定义信号通知属性面板刷新
        if scene is not None and hasattr(scene, "property_changed"):
            scene.property_changed.emit()
        self.item.update()
        if scene:
            # 路径点移动需刷新连线
            if isinstance(self.item, PathPointItem):
                if hasattr(scene, "refresh_edges_for"):
                    scene.refresh_edges_for(self.item.pid)
            # 货架/货物属性变更（如 yaw）后同步其附属点
            if hasattr(scene, "sync_attached_for"):
                scene.sync_attached_for(self.item)


class AddEdgeCommand(QUndoCommand):
    """添加路径连线（幂等）。"""

    def __init__(self, scene, edge, text="添加连线"):
        super().__init__(text)
        self.scene = scene
        self.edge = edge

    def undo(self):
        _safe_remove(self.scene, self.edge)
        if self.edge in self.scene.edges:
            self.scene.edges.remove(self.edge)

    def redo(self):
        _safe_add(self.scene, self.edge)
        if self.edge not in self.scene.edges:
            self.scene.edges.append(self.edge)
