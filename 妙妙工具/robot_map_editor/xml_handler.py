"""XML 导入/导出模块 - MuJoCo 风格环境物体存取。

导出范围：货架、障碍物、货物+托盘、叉车初始位置框。
路径点不写入此 XML。

XML 约定（与用户提供的示例一致）：
- body pos 使用世界坐标，euler="0 0 yaw" 单位为弧度
- geom size 为半尺寸（MuJoCo 约定）
- 托盘/货物 body 含 freejoint 与各 geom（带 mass/rgba/friction/contype/conaffinity）
- 叉车初始位置框：单个 box geom，contype=0 conaffinity=0（无碰撞，仅显示）
"""
import xml.etree.ElementTree as ET
import math
import re
from items import (ShelfItem, ObstacleItem, GoodsItem, ForkliftItem,
                    StackItem, DropZoneItem,
                    TYPE_SHELF, TYPE_OBSTACLE, TYPE_GOODS, TYPE_FORKLIFT,
                    TYPE_STACK, TYPE_DROP_ZONE)

# 默认属性常量（与用户示例保持一致）
PALLET_GEOMS = [
    # (name, size, pos, rgba, mass, friction)
    ("pallet_top", "0.35 0.4 0.015", "0 0 0.135", "0.6 0.4 0.2 1", 15, "1.5 1 1"),
    ("pallet_block_center", "0.35 0.035 0.06", "0 0 0.06", "0.3 0.2 0.1 1", 2, "1.5 0.1 0.1"),
    ("pallet_block_left", "0.35 0.04 0.08", "0 0.35 0.05", "0.3 0.2 0.1 1", 2, "1.5 0.1 0.1"),
    ("pallet_block_right", "0.35 0.04 0.08", "0 -0.35 0.05", "0.3 0.2 0.1 1", 2, "1.5 0.1 0.1"),
]


def _fmt(v):
    """格式化数值，避免科学计数法。"""
    return f"{float(v):.6g}"


def _yaw_to_rad(yaw_deg):
    return math.radians(float(yaw_deg))


# ----------------------------------------------------------------------
# 导出
# ----------------------------------------------------------------------
def export_xml(items, filepath):
    """将环境物体列表导出为 MuJoCo 风格 XML。

    Args:
        items: BaseItem 列表（仅含 TYPE_SHELF/OBSTACLE/GOODS/FORKLIFT）
        filepath: 输出 XML 文件路径
    """
    root = ET.Element("mujoco", attrib={"model": "robot_env"})
    worldbody = ET.SubElement(root, "worldbody")

    # 用于区分同名元素，按类型计数
    counters = {TYPE_SHELF: 0, TYPE_OBSTACLE: 0,
                TYPE_GOODS: 0, TYPE_FORKLIFT: 0,
                TYPE_STACK: 0, TYPE_DROP_ZONE: 0}

    for item in items:
        t = item.item_type
        counters[t] = counters.get(t, 0) + 1
        idx = counters[t]
        if t == TYPE_SHELF:
            _export_shelf(worldbody, item, idx)
        elif t == TYPE_OBSTACLE:
            _export_obstacle(worldbody, item, idx)
        elif t == TYPE_GOODS:
            _export_goods(worldbody, item, idx)
        elif t == TYPE_FORKLIFT:
            _export_forklift(worldbody, item, idx)
        elif t == TYPE_STACK:
            _export_stack(worldbody, item)
        elif t == TYPE_DROP_ZONE:
            _export_drop_zone(worldbody, item)

    _indent(root)
    tree = ET.ElementTree(root)
    tree.write(filepath, encoding="utf-8", xml_declaration=True)


def _indent(elem, level=0):
    """美化 XML 缩进。"""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def _add_geom(parent, name, size, pos, rgba, mass=None,
              friction=None, contype="1", conaffinity="1",
              geom_type="box"):
    attrib = {"type": geom_type, "size": size, "pos": pos, "rgba": rgba,
              "contype": contype, "conaffinity": conaffinity}
    if name:
        attrib["name"] = name
    if mass is not None:
        attrib["mass"] = _fmt(mass)
    if friction is not None:
        attrib["friction"] = friction
    ET.SubElement(parent, "geom", attrib)


def _export_shelf(worldbody, item, idx):
    """货架：4 立柱 + 2 隔板（依据用户示例 XML）。"""
    name = item.name or f"shelf_{idx}"
    yaw = _yaw_to_rad(item.yaw)
    body = ET.SubElement(worldbody, "body", attrib={
        "name": name,
        "pos": f"{_fmt(item.x)} {_fmt(item.y)} {_fmt(item.z)}",
        "euler": f"0 0 {_fmt(yaw)}",
    })
    # 4 立柱（半尺寸 0.04×0.04×1.0），z=1.0
    posts = [(-0.4, -1.0), (-0.4, 1.0), (0.4, -1.0), (0.4, 1.0)]
    for px, py in posts:
        _add_geom(body, name=None,
                  size="0.04 0.04 1.0",
                  pos=f"{_fmt(px)} {_fmt(py)} 1.0",
                  rgba="0.4 0.4 0.4 1")
    # 下层隔板（半尺寸 0.4×1.0×0.02）z=0.6
    _add_geom(body, name=None,
              size="0.4 1.0 0.02",
              pos="0 0 0.6",
              rgba="0.5 0.3 0.1 1")
    # 上层隔板 z=1.2
    _add_geom(body, name=None,
              size="0.4 1.0 0.02",
              pos="0 0 1.2",
              rgba="0.5 0.3 0.1 1")


def _export_obstacle(worldbody, item, idx):
    """矩形障碍物。"""
    name = item.name or f"obstacle_{idx}"
    yaw = _yaw_to_rad(item.yaw)
    # body 放在底面中心，z = H/2
    body = ET.SubElement(worldbody, "body", attrib={
        "name": name,
        "pos": f"{_fmt(item.x)} {_fmt(item.y)} {_fmt(item.z + item.H / 2)}",
        "euler": f"0 0 {_fmt(yaw)}",
    })
    _add_geom(body, name=None,
              size=f"{_fmt(item.L / 2)} {_fmt(item.W / 2)} {_fmt(item.H / 2)}",
              pos="0 0 0",
              rgba="0.7 0.2 0.2 1",
              mass=10, friction="1.5 1 1")


def _export_goods(worldbody, item, idx):
    """货物 + 托盘：两个 body（托盘 body 在前，货物 body 在上）。"""
    pallet_name = f"pallet_{idx}"
    cargo_name = f"cargo_{idx}"
    yaw = _yaw_to_rad(item.yaw)
    # 托盘 body z = bottom_z + 0.075
    pallet_body = ET.SubElement(worldbody, "body", attrib={
        "name": pallet_name,
        "pos": f"{_fmt(item.x)} {_fmt(item.y)} {_fmt(item.pallet_z)}",
        "euler": f"0 0 {_fmt(yaw)}",
    })
    ET.SubElement(pallet_body, "freejoint")
    for gname, size, pos, rgba, mass, friction in PALLET_GEOMS:
        _add_geom(pallet_body, name=gname, size=size, pos=pos,
                  rgba=rgba, mass=mass, friction=friction)
    # 货物 body z = bottom_z + 0.4
    cargo_body = ET.SubElement(worldbody, "body", attrib={
        "name": cargo_name,
        "pos": f"{_fmt(item.x)} {_fmt(item.y)} {_fmt(item.cargo_z)}",
        "euler": f"0 0 {_fmt(yaw)}",
    })
    ET.SubElement(cargo_body, "freejoint")
    _add_geom(cargo_body, name=None,
              size="0.31 0.4 0.15",
              pos="0 0 0",
              rgba="1 0.4 0.7 1",
              mass=50, friction="2.2 1 1")


def _export_forklift(worldbody, item, idx):
    """叉车初始位置框：仅一个 box geom（底盘），无碰撞属性。"""
    name = item.name or f"forklift_init_{idx}"
    yaw = _yaw_to_rad(item.yaw)
    body = ET.SubElement(worldbody, "body", attrib={
        "name": name,
        "pos": f"{_fmt(item.x)} {_fmt(item.y)} {_fmt(item.z + item.H / 2)}",
        "euler": f"0 0 {_fmt(yaw)}",
    })
    # 底盘 box，半尺寸 L/2 × W/2 × H/2，contype=0 conaffinity=0
    _add_geom(body, name="forklift_chassis",
              size=f"{_fmt(item.L / 2)} {_fmt(item.W / 2)} {_fmt(item.H / 2)}",
              pos="0 0 0",
              rgba="0.2 0.6 1 1",
              contype="0", conaffinity="0")


def _export_stack(worldbody, item):
    """货堆：每层一对 body（托盘 + 货物），命名 stack_<ID>_L<i>_pallet/_cargo。

    命名约定是 FSM 与路径规划器的接口：`stack_<ID>_L1_cargo` 就是"最下方货物"，
    目标位姿以它为准。
    """
    prefix = getattr(item, "full_name", None) or f"stack_{item.name}"
    if callable(prefix):
        prefix = prefix()
    yaw = _yaw_to_rad(item.yaw)
    for i in range(1, int(item.layers) + 1):
        p_body = ET.SubElement(worldbody, "body", attrib={
            "name": f"{prefix}_L{i}_pallet",
            "pos": f"{_fmt(item.x)} {_fmt(item.y)} {_fmt(item.pallet_z(i))}",
            "euler": f"0 0 {_fmt(yaw)}",
        })
        ET.SubElement(p_body, "freejoint")
        for gname, size, pos, rgba, mass, friction in PALLET_GEOMS:
            _add_geom(p_body, name=gname, size=size, pos=pos,
                      rgba=rgba, mass=mass, friction=friction)
        c_body = ET.SubElement(worldbody, "body", attrib={
            "name": f"{prefix}_L{i}_cargo",
            "pos": f"{_fmt(item.x)} {_fmt(item.y)} {_fmt(item.cargo_z(i))}",
            "euler": f"0 0 {_fmt(yaw)}",
        })
        ET.SubElement(c_body, "freejoint")
        _add_geom(c_body, name=None,
                  size="0.31 0.4 0.15",
                  pos="0 0 0",
                  rgba="1 0.4 0.7 1",
                  mass=50, friction="2.2 1 1")


def _export_drop_zone(worldbody, item):
    """地面指定放置区：贴地薄板，contype=0 conaffinity=0（不碰撞、不挡雷达）。"""
    prefix = getattr(item, "full_name", None) or f"drop_zone_{item.name}"
    if callable(prefix):
        prefix = prefix()
    yaw = _yaw_to_rad(item.yaw)
    body = ET.SubElement(worldbody, "body", attrib={
        "name": prefix,
        "pos": f"{_fmt(item.x)} {_fmt(item.y)} 0.005",
        "euler": f"0 0 {_fmt(yaw)}",
    })
    _add_geom(body, name=f"{prefix}_plate",
              size=f"{_fmt(item.L / 2)} {_fmt(item.W / 2)} 0.005",
              pos="0 0 0",
              rgba="0.3 0.85 0.3 0.45",
              contype="0", conaffinity="0")


# ----------------------------------------------------------------------
# 导入
# ----------------------------------------------------------------------
def import_xml(filepath):
    """读取环境 XML，返回 BaseItem 列表（不包含路径点）。

    按 body.name 前缀识别类型：
      shelf / shelf_*         -> ShelfItem
      obstacle / obstacle_*   -> ObstacleItem
      pallet_* (配合 cargo_*) -> GoodsItem
      forklift_init_*         -> ForkliftItem
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        return []

    items = []
    # 暂存 pallet body，与 cargo body 配对
    pallets = {}    # (x, y) -> {body, z, yaw, name}
    # 暂存货堆：ID -> {x, y, yaw, layers}
    stacks = {}

    for body in worldbody.findall("body"):
        name = body.attrib.get("name", "")
        pos_str = body.attrib.get("pos", "0 0 0").split()
        x = float(pos_str[0]) if len(pos_str) > 0 else 0.0
        y = float(pos_str[1]) if len(pos_str) > 1 else 0.0
        z = float(pos_str[2]) if len(pos_str) > 2 else 0.0
        euler_str = body.attrib.get("euler", "0 0 0").split()
        yaw_rad = float(euler_str[2]) if len(euler_str) > 2 else 0.0
        yaw_deg = math.degrees(yaw_rad)

        # 判断类型
        if name.startswith("shelf"):
            L, W, H = 0.8, 2.0, 2.0
            # 从首个 geom 推断尺寸（若可能）
            geoms = body.findall("geom")
            if geoms:
                # 立柱：size="0.04 0.04 1.0" -> 推断 L=0.8 W=2.0
                pass
            items.append(ShelfItem(x=x, y=y, z=z, yaw=yaw_deg,
                                    L=L, W=W, H=H, name=name))
        elif name.startswith("obstacle"):
            L, W, H = 0.5, 0.5, 0.5
            geoms = body.findall("geom")
            if geoms:
                sizes = geoms[0].attrib.get("size", "0.25 0.25 0.25").split()
                if len(sizes) >= 3:
                    L = float(sizes[0]) * 2
                    W = float(sizes[1]) * 2
                    H = float(sizes[2]) * 2
            items.append(ObstacleItem(x=x, y=y, z=z - H / 2, yaw=yaw_deg,
                                       L=L, W=W, H=H, name=name))
        elif name.startswith("pallet"):
            pallets[(x, y)] = {"z": z, "yaw": yaw_deg, "name": name}
        elif name.startswith("cargo"):
            # 查找对应的 pallet（同 x,y）
            pal = pallets.get((x, y))
            if pal:
                bottom_z = z - 0.4   # cargo body z = bottom_z + 0.4
                items.append(GoodsItem(x=x, y=y, bottom_z=bottom_z,
                                        yaw=yaw_deg, name=name))
            else:
                # 没有对应托盘，仍创建货物（独立）
                bottom_z = z - 0.4
                items.append(GoodsItem(x=x, y=y, bottom_z=bottom_z,
                                        yaw=yaw_deg, name=name))
        elif name.startswith("drop_zone"):
            L, W = DropZoneItem.DEFAULT_L, DropZoneItem.DEFAULT_W
            geoms = body.findall("geom")
            if geoms:
                sizes = geoms[0].attrib.get("size", "0.6 0.6 0.005").split()
                if len(sizes) >= 2:
                    L = float(sizes[0]) * 2
                    W = float(sizes[1]) * 2
            items.append(DropZoneItem(x=x, y=y, yaw=yaw_deg, L=L, W=W,
                                       name=name))
        elif name.startswith("stack"):
            m = re.match(r"stack_(.+)_L(\d+)_(pallet|cargo)$", name)
            if m:
                sid, layer = m.group(1), int(m.group(2))
                rec = stacks.setdefault(sid, {"x": x, "y": y,
                                              "yaw": yaw_deg, "layers": 0})
                rec["layers"] = max(rec["layers"], layer)
        elif name.startswith("forklift"):
            H = 0.3
            geoms = body.findall("geom")
            if geoms:
                sizes = geoms[0].attrib.get("size", "1.17 0.6 0.15").split()
                if len(sizes) >= 3:
                    H = float(sizes[2]) * 2
            items.append(ForkliftItem(x=x, y=y, z=z - H / 2, yaw=yaw_deg,
                                       name=name))
    for sid, rec in stacks.items():
        items.append(StackItem(x=rec["x"], y=rec["y"], layers=rec["layers"],
                                yaw=rec["yaw"], name=sid))
    return items
