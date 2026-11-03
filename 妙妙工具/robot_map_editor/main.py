"""主窗口模块 - 三栏布局（工具栏/画布/属性面板）+ 菜单与撤销栈。

入口：python main.py
"""
import sys
import os

# 让本地依赖能找到 PyQt5（若安装到 pylibs 目录）- 必须在 PyQt5 导入前
_HERE = os.path.dirname(os.path.abspath(__file__))
_PYLIBS = os.path.normpath(os.path.join(_HERE, "..", "pylibs"))
if os.path.isdir(_PYLIBS) and _PYLIBS not in sys.path:
    sys.path.insert(0, _PYLIBS)

from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QColor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QSplitter,
                              QVBoxLayout, QHBoxLayout, QPushButton,
                              QToolBar, QAction, QLabel, QFormLayout,
                              QLineEdit, QDoubleSpinBox, QSpinBox,
                              QListWidget, QListWidgetItem, QFileDialog,
                              QMessageBox, QStatusBar, QGroupBox, QFrame,
                              QButtonGroup, QToolButton, QSizePolicy,
                              QGraphicsItem)

from canvas_scene import CanvasScene
from canvas_view import CanvasView
from items import (TYPE_SHELF, TYPE_OBSTACLE, TYPE_GOODS, TYPE_FORKLIFT,
                    TYPE_PATH_POINT, TYPE_STACK, TYPE_DROP_ZONE,
                    ShelfItem, ObstacleItem, GoodsItem, ForkliftItem,
                    StackItem, DropZoneItem,
                    PathPointItem, PathEdgeItem,
                    AttachedPathPointItem, attached_point_pos,
                    ATTACHABLE_TYPES)
from commands import (AddItemCommand, RemoveItemsCommand, MoveItemCommand,
                       PropertyChangeCommand, AddEdgeCommand)
import xml_handler
import json_handler


# 工具模式枚举
TOOL_SELECT = "select"
TOOL_STACK = "stack"          # 货堆（27 号起替代货架）
TOOL_DROP_ZONE = "drop_zone"  # 地面指定放置区
TOOL_OBSTACLE = "obstacle"
TOOL_GOODS = "goods"
TOOL_FORKLIFT = "forklift"
TOOL_PATH_POINT = "path_point"
TOOL_CONNECT = "connect"


class PropertyPanel(QWidget):
    """右侧属性面板：显示并编辑选中元素的属性。"""

    propertyApplied = pyqtSignal(str, object)  # (prop_name, new_value)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_item = None
        self._widgets = {}
        self._building = False
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("属性面板")
        title.setStyleSheet("font-weight:bold; font-size:13px;")
        layout.addWidget(title)

        self.info_label = QLabel("未选中元素")
        self.info_label.setStyleSheet("color:#666;")
        layout.addWidget(self.info_label)

        # 表单组
        self.form_group = QGroupBox("属性")
        self.form_layout = QFormLayout(self.form_group)
        self.form_layout.setLabelAlignment(Qt.AlignRight)
        layout.addWidget(self.form_group)

        # 应用按钮
        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("应用")
        self.apply_btn.clicked.connect(self._on_apply)
        btn_layout.addStretch()
        btn_layout.addWidget(self.apply_btn)
        layout.addLayout(btn_layout)

        layout.addStretch()

        # 路径点列表（仅显示，便于查看序号）
        self.path_list_group = QGroupBox("路径点列表")
        pl = QVBoxLayout(self.path_list_group)
        self.path_list = QListWidget()
        pl.addWidget(self.path_list)
        layout.addWidget(self.path_list_group)

    def clear_form(self):
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._widgets.clear()

    def set_item(self, item):
        """切换显示的元素。"""
        self._building = True
        try:
            self.clear_form()
            self.current_item = item
            if item is None:
                self.info_label.setText("未选中元素")
                return
            self.info_label.setText(
                f"选中: {getattr(item, 'name', '')} "
                f"({getattr(item, 'item_type', type(item).__name__)})")
            props = item.get_properties() if hasattr(item, "get_properties") else []
            for name, value, unit in props:
                w = self._make_editor(value, unit)
                self._widgets[name] = (w, unit)
                self.form_layout.addRow(f"{name} ({unit}):", w)
        finally:
            self._building = False

    def _make_editor(self, value, unit):
        if unit == "int":
            w = QSpinBox()
            w.setRange(-100000, 100000)
            w.setValue(int(value))
            return w
        if unit == "str":
            w = QLineEdit()
            w.setText(str(value))
            return w
        # 默认 double
        w = QDoubleSpinBox()
        w.setDecimals(4)
        w.setRange(-100000.0, 100000.0)
        w.setSingleStep(0.1)
        w.setValue(float(value))
        return w

    def _read_value(self, w, unit):
        if unit == "int":
            return int(w.value())
        if unit == "str":
            return w.text()
        return float(w.value())

    def _on_apply(self):
        """点击应用：对每个属性触发 PropertyChangeCommand。"""
        if self.current_item is None:
            return
        # 主窗口会订阅此信号，逐项发命令以支持撤销
        for name, (w, unit) in self._widgets.items():
            new_val = self._read_value(w, unit)
            old_val = self._get_old_value(name)
            if old_val is None:
                continue
            if old_val != new_val:
                self.propertyApplied.emit(name, new_val)

    def _get_old_value(self, name):
        if self.current_item is None:
            return None
        for pname, pval, _ in self.current_item.get_properties():
            if pname == name:
                return pval
        return None

    def refresh_values(self):
        """从当前元素重新读取值刷新编辑器（外部修改后调用）。"""
        if self.current_item is None:
            return
        self._building = True
        try:
            for name, (w, unit) in self._widgets.items():
                val = self._get_old_value(name)
                if val is None:
                    continue
                if unit == "int":
                    w.setValue(int(val))
                elif unit == "str":
                    w.setText(str(val))
                else:
                    w.setValue(float(val))
        finally:
            self._building = False

    def update_path_list(self, scene):
        self.path_list.clear()
        for pid in sorted(scene.path_points.keys()):
            pt = scene.path_points[pid]
            item = QListWidgetItem(f"#{pid}  ({pt.x:.2f}, {pt.y:.2f})")
            self.path_list.addItem(item)


class MainWindow(QMainWindow):
    """主窗口。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("机器人 2D 地图编辑器")
        self.resize(1400, 850)

        # 工具状态
        self.current_tool = TOOL_SELECT
        self._connect_first_pid = None  # 连线模式下的第一个点

        # 场景与视图
        self.scene = CanvasScene()
        self.view = CanvasView(self.scene)
        self.view.set_placement_mode(False)

        # 撤销栈
        from PyQt5.QtWidgets import QUndoStack, QUndoView
        self.undo_stack = QUndoStack(self)
        self.undo_stack.canUndoChanged.connect(self._update_undo_actions)
        self.undo_stack.canRedoChanged.connect(self._update_undo_actions)

        # 拖拽移动追踪
        self._moving_item = None
        self._move_start_pos = None

        # 三栏布局
        self._init_ui()
        self._init_toolbar()
        self._init_menu()

        # 信号连接
        # selectionChanged 是基类信号，直接连接（不要在子类重写）
        self.scene.selectionChanged.connect(self._on_selection_changed)
        self.scene.property_changed.connect(self._on_selection_changed)
        self.view.mouse_moved.connect(self._on_mouse_moved)
        self.view.scale_changed.connect(self._on_scale_changed)

        # 默认放置一个叉车初始框
        self._add_default_forklift()

    # ----------------------------------------------------------
    def _add_default_forklift(self):
        fk = ForkliftItem(x=-6.0, y=0.0, z=0.0, yaw=0.0)
        self.undo_stack.push(AddItemCommand(self.scene, fk, "放置叉车初始框"))

    # ----------------------------------------------------------
    # UI 初始化
    # ----------------------------------------------------------
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        # 左侧工具栏
        self.tool_panel = self._build_tool_panel()
        splitter.addWidget(self.tool_panel)
        # 中间画布
        canvas_container = QFrame()
        canvas_container.setFrameShape(QFrame.NoFrame)
        cl = QVBoxLayout(canvas_container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(self.view)
        splitter.addWidget(canvas_container)
        # 右侧属性面板
        self.property_panel = PropertyPanel()
        self.property_panel.propertyApplied.connect(self._on_property_applied)
        splitter.addWidget(self.property_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([120, 1000, 320])
        layout.addWidget(splitter)

        # 状态栏
        self.status_tool = QLabel("工具: 选择")
        self.status_coord = QLabel("坐标: (0.00, 0.00)")
        self.status_zoom = QLabel("缩放: 30 px/m")
        sb = QStatusBar()
        sb.addWidget(self.status_tool)
        sb.addWidget(self.status_coord)
        sb.addPermanentWidget(self.status_zoom)
        self.setStatusBar(sb)

    def _build_tool_panel(self):
        panel = QFrame()
        panel.setFixedWidth(110)
        panel.setStyleSheet("QFrame{background:#f0f0f0;}")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        title = QLabel("工具栏")
        title.setStyleSheet("font-weight:bold;")
        layout.addWidget(title)

        # 工具按钮组
        self.tool_btn_group = QButtonGroup(self)
        self.tool_btn_group.setExclusive(True)
        tools = [
            (TOOL_SELECT, "选择/拖拽"),
            (TOOL_STACK, "货堆"),
            (TOOL_DROP_ZONE, "地面区"),
            (TOOL_OBSTACLE, "障碍物"),
            (TOOL_GOODS, "货物+托盘"),
            (TOOL_FORKLIFT, "叉车框"),
            (TOOL_PATH_POINT, "路径点"),
            (TOOL_CONNECT, "连线"),
        ]
        for tid, label in tools:
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.setMinimumHeight(32)
            btn.setMinimumWidth(100)
            if tid == TOOL_SELECT:
                btn.setChecked(True)
            btn.clicked.connect(lambda _=False, t=tid: self._set_tool(t))
            self.tool_btn_group.addButton(btn)
            layout.addWidget(btn)

        layout.addWidget(self._sep())

        # 视图操作
        self.btn_undo = QPushButton("撤销")
        self.btn_redo = QPushButton("重做")
        self.btn_undo.clicked.connect(self.undo_stack.undo)
        self.btn_redo.clicked.connect(self.undo_stack.redo)
        layout.addWidget(self.btn_undo)
        layout.addWidget(self.btn_redo)

        layout.addWidget(self._sep())
        self.btn_toggle_grid = QPushButton("网格 ON")
        self.btn_toggle_grid.setCheckable(True)
        self.btn_toggle_grid.setChecked(True)
        self.btn_toggle_grid.clicked.connect(self._toggle_grid)
        layout.addWidget(self.btn_toggle_grid)

        self.btn_toggle_axes = QPushButton("坐标轴 ON")
        self.btn_toggle_axes.setCheckable(True)
        self.btn_toggle_axes.setChecked(True)
        self.btn_toggle_axes.clicked.connect(self._toggle_axes)
        layout.addWidget(self.btn_toggle_axes)

        self.btn_delete = QPushButton("删除选中")
        self.btn_delete.clicked.connect(self._delete_selected)
        layout.addWidget(self.btn_delete)

        layout.addWidget(self._sep())
        # 路径存取
        self.btn_save_path = QPushButton("保存路径")
        self.btn_save_path.clicked.connect(self._save_path)
        layout.addWidget(self.btn_save_path)
        self.btn_load_path = QPushButton("读取路径")
        self.btn_load_path.clicked.connect(self._load_path)
        layout.addWidget(self.btn_load_path)

        layout.addWidget(self._sep())
        # XML 存取
        self.btn_export_xml = QPushButton("导出环境 XML")
        self.btn_export_xml.clicked.connect(self._export_xml)
        layout.addWidget(self.btn_export_xml)
        self.btn_import_xml = QPushButton("读取环境 XML")
        self.btn_import_xml.clicked.connect(self._import_xml)
        layout.addWidget(self.btn_import_xml)

        layout.addWidget(self._sep())
        # 物体信息 JSON 导出
        self.btn_export_info = QPushButton("导出物体信息 JSON")
        self.btn_export_info.clicked.connect(self._export_env_info)
        layout.addWidget(self.btn_export_info)

        layout.addStretch()
        return panel

    @staticmethod
    def _sep():
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setFrameShadow(QFrame.Sunken)
        return f

    def _init_toolbar(self):
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        self.addToolBar(tb)

    def _init_menu(self):
        menubar = self.menuBar()
        m_file = menubar.addMenu("文件")
        a_export = QAction("导出环境 XML...", self)
        a_export.triggered.connect(self._export_xml)
        m_file.addAction(a_export)
        a_import = QAction("读取环境 XML...", self)
        a_import.triggered.connect(self._import_xml)
        m_file.addAction(a_import)
        m_file.addSeparator()
        a_export_info = QAction("导出物体信息 JSON...", self)
        a_export_info.triggered.connect(self._export_env_info)
        m_file.addAction(a_export_info)
        m_file.addSeparator()
        a_save_path = QAction("保存路径 JSON...", self)
        a_save_path.triggered.connect(self._save_path)
        m_file.addAction(a_save_path)
        a_load_path = QAction("读取路径 JSON...", self)
        a_load_path.triggered.connect(self._load_path)
        m_file.addAction(a_load_path)
        m_file.addSeparator()
        a_quit = QAction("退出", self)
        a_quit.triggered.connect(self.close)
        m_file.addAction(a_quit)

        m_edit = menubar.addMenu("编辑")
        a_undo = QAction("撤销", self)
        a_undo.triggered.connect(self.undo_stack.undo)
        m_edit.addAction(a_undo)
        a_redo = QAction("重做", self)
        a_redo.triggered.connect(self.undo_stack.redo)
        m_edit.addAction(a_redo)
        self.act_undo = a_undo
        self.act_redo = a_redo

        m_view = menubar.addMenu("视图")
        a_grid = QAction("切换网格", self)
        a_grid.triggered.connect(self._toggle_grid)
        m_view.addAction(a_grid)
        a_axes = QAction("切换坐标轴", self)
        a_axes.triggered.connect(self._toggle_axes)
        m_view.addAction(a_axes)
        a_reset = QAction("重置视图", self)
        a_reset.triggered.connect(self._reset_view)
        m_view.addAction(a_reset)

    # ----------------------------------------------------------
    # 工具与视图切换
    # ----------------------------------------------------------
    def _set_tool(self, tool):
        self.current_tool = tool
        self.view.set_placement_mode(tool != TOOL_SELECT)
        names = {
            TOOL_SELECT: "选择/拖拽",
            TOOL_STACK: "货堆",
            TOOL_DROP_ZONE: "地面区",
            TOOL_OBSTACLE: "障碍物",
            TOOL_GOODS: "货物+托盘",
            TOOL_FORKLIFT: "叉车框",
            TOOL_PATH_POINT: "路径点",
            TOOL_CONNECT: "连线",
        }
        self.status_tool.setText(f"工具: {names.get(tool, tool)}")
        # 取消之前的连线选择
        self._connect_first_pid = None
        if tool != TOOL_SELECT:
            self.scene.clearSelection()

    def _toggle_grid(self):
        self.scene.show_grid = not self.scene.show_grid
        self.scene.invalidate(self.scene.sceneRect())
        self.btn_toggle_grid.setChecked(self.scene.show_grid)
        self.btn_toggle_grid.setText(
            "网格 ON" if self.scene.show_grid else "网格 OFF")

    def _toggle_axes(self):
        self.scene.show_axes = not self.scene.show_axes
        self.scene.invalidate(self.scene.sceneRect())
        self.btn_toggle_axes.setChecked(self.scene.show_axes)
        self.btn_toggle_axes.setText(
            "坐标轴 ON" if self.scene.show_axes else "坐标轴 OFF")

    def _reset_view(self):
        self.view.resetTransform()
        self.view.scale_factor = 30.0
        self.view.scale(30.0, 30.0)
        self.view.centerOn(QPointF(0, 0))
        self.status_zoom.setText("缩放: 30 px/m")

    def _on_scale_changed(self, factor):
        self.status_zoom.setText(f"缩放: {factor:.1f} px/m")

    def _on_mouse_moved(self, x, y):
        self.status_coord.setText(f"坐标: ({x:.2f}, {y:.2f})")

    def _update_undo_actions(self):
        can_u = self.undo_stack.canUndo()
        can_r = self.undo_stack.canRedo()
        self.btn_undo.setEnabled(can_u)
        self.btn_redo.setEnabled(can_r)
        if hasattr(self, "act_undo"):
            self.act_undo.setEnabled(can_u)
            self.act_redo.setEnabled(can_r)

    # ----------------------------------------------------------
    # 选中变化
    # ----------------------------------------------------------
    def _on_selection_changed(self):
        sel = self.scene.selectedItems()
        # 取第一个可识别的图元
        chosen = None
        for it in sel:
            if hasattr(it, "item_type") or isinstance(it, PathPointItem):
                chosen = it
                break
        self.property_panel.set_item(chosen)
        self.property_panel.update_path_list(self.scene)

    def _on_property_applied(self, prop_name, new_value):
        item = self.property_panel.current_item
        if item is None:
            return
        # 读取旧值用于撤销
        old_val = None
        for pname, pval, _ in item.get_properties():
            if pname == prop_name:
                old_val = pval
                break
        if old_val is None or old_val == new_value:
            return
        cmd = PropertyChangeCommand(item, prop_name, old_val, new_value,
                                     f"修改 {prop_name}")
        self.undo_stack.push(cmd)
        # 刷新面板
        self.property_panel.refresh_values()
        # 若是路径点，刷新连线
        if isinstance(item, PathPointItem):
            self.scene.refresh_edges_for(item.pid)
        self.property_panel.update_path_list(self.scene)

    # ----------------------------------------------------------
    # 画布事件（覆盖 view 的事件）
    # ----------------------------------------------------------
    def eventFilter(self, obj, event):
        # 拦截视口的鼠标按下/释放以处理放置和拖拽
        # QGraphicsView 的鼠标事件实际发给 viewport()，必须过滤 viewport
        from PyQt5.QtCore import QEvent
        if obj is self.view.viewport():
            if event.type() == QEvent.MouseButtonPress:
                if self._on_view_mouse_press(event):
                    return True
            elif event.type() == QEvent.MouseButtonRelease:
                if self._on_view_mouse_release(event):
                    return True
        return super().eventFilter(obj, event)

    def _on_view_mouse_press(self, event):
        """左键按下：放置工具或选择/拖拽。"""
        if event.button() != Qt.LeftButton:
            return False
        # 中/右键平移由 view 自身处理
        if self.view._panning:
            return False

        scene_pt = self.view.mapToScene(event.pos())
        world_x = float(scene_pt.x())
        world_y = -float(scene_pt.y())

        tool = self.current_tool
        if tool == TOOL_SELECT:
            # 让 view 走默认选择/拖拽流程
            # 追踪移动开始（用于命令栈）
            item = self.scene.itemAt(scene_pt, self.view.transform())
            if item and (hasattr(item, "item_type") or
                         isinstance(item, PathPointItem)):
                if item.flags() & QGraphicsItem.ItemIsMovable:
                    self._moving_item = item
                    self._move_start_pos = item.pos()
            return False  # 让默认处理继续

        if tool == TOOL_STACK:
            # 货堆默认 3 层、yaw=+90（叉车从南侧 2.85m 进叉）
            stack = StackItem(x=world_x, y=world_y, layers=3, yaw=90.0,
                              name=f"stack_{len(self.scene.environment_items())}")
            self.undo_stack.push(AddItemCommand(self.scene, stack, "放置货堆",
                                                create_attached=True))
            self._set_tool(TOOL_SELECT)
            return True
        if tool == TOOL_DROP_ZONE:
            zone = DropZoneItem(x=world_x, y=world_y, yaw=90.0,
                                name=f"G{len(self.scene.environment_items())}")
            self.undo_stack.push(AddItemCommand(self.scene, zone, "放置地面区",
                                                create_attached=True))
            self._set_tool(TOOL_SELECT)
            return True
        if tool == TOOL_OBSTACLE:
            ob = ObstacleItem(x=world_x, y=world_y, z=0.0, yaw=0.0)
            self.undo_stack.push(AddItemCommand(self.scene, ob, "放置障碍物"))
            self._set_tool(TOOL_SELECT)
            return True
        if tool == TOOL_GOODS:
            # 若点击处已有货物，则堆叠一层（bottom_z 抬升）
            bottom_z = 0.0
            is_stack = False
            for it in self.scene.items():
                if isinstance(it, GoodsItem):
                    if abs(it.x - world_x) < 0.01 and abs(it.y - world_y) < 0.01:
                        is_stack = True
                        bottom_z = max(bottom_z, it.bottom_z + GoodsItem.LAYER_STEP)
            g = GoodsItem(x=world_x, y=world_y, bottom_z=bottom_z, yaw=0.0)
            # 多层堆叠只在首层创建附属点
            self.undo_stack.push(AddItemCommand(self.scene, g, "放置货物",
                                                create_attached=not is_stack))
            self._set_tool(TOOL_SELECT)
            return True
        if tool == TOOL_FORKLIFT:
            fk = ForkliftItem(x=world_x, y=world_y, z=0.0, yaw=0.0)
            self.undo_stack.push(AddItemCommand(self.scene, fk, "放置叉车框"))
            self._set_tool(TOOL_SELECT)
            return True
        if tool == TOOL_PATH_POINT:
            pt = self.scene.add_path_point(world_x, world_y)
            self.undo_stack.push(AddItemCommand(self.scene, pt, "放置路径点"))
            self.property_panel.update_path_list(self.scene)
            # 保持工具不变（可连续放置）
            return True
        if tool == TOOL_CONNECT:
            # 查找点击的路径点
            clicked = None
            for it in self.scene.items(scene_pt):
                if isinstance(it, PathPointItem):
                    clicked = it
                    break
            if clicked is None:
                self.statusBar().showMessage("请点击路径点以连线", 2000)
                return True
            if self._connect_first_pid is None:
                self._connect_first_pid = clicked.pid
                self.statusBar().showMessage(
                    f"已选起点 #{clicked.pid}，请点击终点", 3000)
            else:
                if clicked.pid == self._connect_first_pid:
                    self.statusBar().showMessage("起点终点相同，已取消", 2000)
                    self._connect_first_pid = None
                elif self.scene.has_edge(self._connect_first_pid,
                                          clicked.pid):
                    # 同方向连线已存在，提示并取消
                    self.statusBar().showMessage(
                        f"#{self._connect_first_pid}→#{clicked.pid} "
                        f"连线已存在", 2500)
                    self._connect_first_pid = None
                else:
                    edge = self.scene.create_edge(self._connect_first_pid,
                                                   clicked.pid)
                    if edge:
                        self.undo_stack.push(
                            AddEdgeCommand(self.scene, edge, "添加连线"))
                        self.statusBar().showMessage("连线已添加", 1500)
                    self._connect_first_pid = None
            return True
        return False

    def _on_view_mouse_release(self, event):
        if event.button() != Qt.LeftButton:
            return False
        # 若正在拖拽图元，则压入移动命令
        if self._moving_item is not None:
            new_pos = self._moving_item.pos()
            if new_pos != self._move_start_pos:
                item = self._moving_item
                old_pos = self._move_start_pos
                # 同步逻辑坐标到新位置
                if hasattr(item, "update_logical_from_scene"):
                    item.update_logical_from_scene()
                # 创建命令：redo 会再次 setPos(new_pos)（无视觉变化），undo 会 setPos(old_pos)
                cmd = MoveItemCommand(item, old_pos, new_pos, "移动图元")
                self.undo_stack.push(cmd)
                # 刷新面板
                self.property_panel.refresh_values()
                if isinstance(item, PathPointItem):
                    self.scene.refresh_edges_for(item.pid)
                self.property_panel.update_path_list(self.scene)
            self._moving_item = None
            self._move_start_pos = None
        return False

    # ----------------------------------------------------------
    # 删除选中
    # ----------------------------------------------------------
    def _delete_selected(self):
        sel = [it for it in self.scene.selectedItems()
               if hasattr(it, "item_type")
               or isinstance(it, (PathPointItem, PathEdgeItem))]
        if not sel:
            return
        # 不再预先移除，交由 RemoveItemsCommand 统一处理（幂等）
        cmd = RemoveItemsCommand(self.scene, sel, "删除图元")
        self.undo_stack.push(cmd)
        self.property_panel.set_item(None)
        self.property_panel.update_path_list(self.scene)

    # ----------------------------------------------------------
    # 文件 IO
    # ----------------------------------------------------------
    def _export_xml(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出环境 XML", "", "MuJoCo XML (*.xml)")
        if not path:
            return
        try:
            env_items = self.scene.environment_items()
            xml_handler.export_xml(env_items, path)
            QMessageBox.information(self, "导出成功",
                                     f"已导出 {len(env_items)} 个环境物体到：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _export_env_info(self):
        """导出环境物体的名称、坐标、尺寸等信息到独立 JSON 文件。"""
        path, _ = QFileDialog.getSaveFileName(
            self, "导出物体信息 JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            env_items = self.scene.environment_items()
            if not env_items:
                QMessageBox.information(self, "无物体", "当前画布上没有环境物体")
                return
            json_handler.export_environment_info(env_items, path)
            QMessageBox.information(
                self, "导出成功",
                f"已导出 {len(env_items)} 个环境物体信息到：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _import_xml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "读取环境 XML", "", "MuJoCo XML (*.xml)")
        if not path:
            return
        try:
            new_items = xml_handler.import_xml(path)
            # 清除现有环境物体及其红色附属点（保留普通路径点）
            for it in self.scene.environment_items():
                self.scene.remove_attached_for(it)
                self.scene.removeItem(it)
            # 添加新物体
            for it in new_items:
                self.scene.addItem(it)
            # 为货架/货物建立附属路径点
            for it in new_items:
                if isinstance(it, ATTACHABLE_TYPES):
                    self.scene.ensure_attached_for(it)
            self.undo_stack.clear()
            self.scene.invalidate(self.scene.sceneRect())
            self.property_panel.update_path_list(self.scene)
            QMessageBox.information(self, "读取成功",
                                     f"已读取 {len(new_items)} 个环境物体")
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))

    def _save_path(self):
        if not self.scene.path_points:
            QMessageBox.information(self, "无路径点", "当前画布上没有路径点")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存路径 JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            points = [self.scene.path_points[pid]
                      for pid in sorted(self.scene.path_points.keys())]
            json_handler.save_path(path, points, self.scene.edges)
            QMessageBox.information(self, "保存成功",
                                     f"已保存 {len(points)} 个路径点、"
                                     f"{len(self.scene.edges)} 条连线")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def _load_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "读取路径 JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            points_data, edges_data = json_handler.load_path(path)
            # 清除现有路径（含红色附属点）并重置父物体的 attached_pid
            for pt in list(self.scene.path_points.values()):
                self.scene.removeItem(pt)
            self.scene.path_points.clear()
            for e in list(self.scene.edges):
                self.scene.removeItem(e)
            self.scene.edges.clear()
            for it in self.scene.items():
                if isinstance(it, ATTACHABLE_TYPES):
                    it.attached_pid = None
            # 重新构建：先加载普通路径点，红色附属点稍后按原 pid 重建
            attached_entries = []
            for p in points_data:
                if p.get("attached"):
                    attached_entries.append(p)
                    continue
                self.scene.add_path_point(p["x"], p["y"], pid=p["id"])
            # 用原 pid 重建红色附属点，并通过坐标 + kind 匹配绑定到对应父物体
            for ap in attached_entries:
                ax, ay = float(ap["x"]), float(ap["y"])
                ap_kind = ap.get("kind", "") or ""
                # 收集所有正后方位置匹配且尚未绑定附属点的父物体
                candidates = []
                for it in self.scene.items():
                    if not isinstance(it, ATTACHABLE_TYPES):
                        continue
                    if getattr(it, "attached_pid", None) is not None:
                        continue
                    px, py = attached_point_pos(it)
                    if abs(px - ax) < 0.05 and abs(py - ay) < 0.05:
                        candidates.append(it)
                if not candidates:
                    # 找不到对应父物体，作为普通路径点加载
                    self.scene.add_path_point(ax, ay, pid=ap["id"])
                    continue
                # 优先按 kind 匹配类型一致的父物体；其次选最底层货物
                # （bottom_z 最小，避免堆叠层抢占附属点）
                if ap_kind:
                    typed = [c for c in candidates
                             if getattr(c, "item_type", "") == ap_kind]
                    if typed:
                        candidates = typed
                matched_parent = min(
                    candidates, key=lambda x: getattr(x, "bottom_z", 0.0))
                pid = ap["id"]
                npt = AttachedPathPointItem(pid, matched_parent, ax, ay)
                self.scene.addItem(npt)
                self.scene.path_points[pid] = npt
                if self.scene._next_pid <= pid:
                    self.scene._next_pid = pid + 1
                matched_parent.attached_pid = pid
            # 加载连线（此时所有 pid 都已就位）
            for e in edges_data:
                self.scene.add_edge(e["from"], e["to"])
            # 为尚未拥有附属点的货架/货物补建红色附属点
            for it in list(self.scene.items()):
                if isinstance(it, ATTACHABLE_TYPES):
                    if getattr(it, "attached_pid", None) is None:
                        self.scene.ensure_attached_for(it)
            self.scene.invalidate(self.scene.sceneRect())
            self.property_panel.update_path_list(self.scene)
            QMessageBox.information(self, "读取成功",
                                     f"已读取 {len(points_data)} 个路径点、"
                                     f"{len(edges_data)} 条连线")
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))


def main():
    app = QApplication(sys.argv)
    # 设置默认字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    win = MainWindow()
    win.show()
    # 安装事件过滤器到 viewport（QGraphicsView 的鼠标事件发给 viewport）
    win.view.viewport().installEventFilter(win)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

