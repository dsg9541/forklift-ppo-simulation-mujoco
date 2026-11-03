"""验证 XML 导出/导入对货物堆叠（3 层）的处理。"""
import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "pylibs"))
sys.path.insert(0, _HERE)

from PyQt5.QtWidgets import QApplication
from items import GoodsItem
import xml_handler


def test_stacking_3layers():
    app = QApplication([])

    # 构造 3 层堆叠货物（同 X,Y，不同 bottom_z）
    items = [
        GoodsItem(x=0.0, y=0.0, bottom_z=0.0, yaw=0.0, name="cargo_L1"),
        GoodsItem(x=0.0, y=0.0, bottom_z=0.55, yaw=0.0, name="cargo_L2"),
        GoodsItem(x=0.0, y=0.0, bottom_z=1.1, yaw=0.0, name="cargo_L3"),
    ]

    print("==== 原始数据 ====")
    for g in items:
        print(f"  {g.name}: bottom_z={g.bottom_z}, layer={g.layer}, "
              f"pallet_z={g.pallet_z}, cargo_z={g.cargo_z}")

    # 导出
    out = os.path.join(_HERE, "..", "test_stack_3layers.xml")
    xml_handler.export_xml(items, out)

    # 打印 XML 中 pallet/cargo body 的 pos
    import xml.etree.ElementTree as ET
    tree = ET.parse(out)
    print("\n==== XML 中的 pallet/cargo body ====")
    for body in tree.iter("body"):
        name = body.attrib.get("name", "")
        if name.startswith("pallet") or name.startswith("cargo"):
            print(f"  {name}: pos={body.attrib.get('pos')}")

    # 导入
    new_items = xml_handler.import_xml(out)
    goods = [it for it in new_items if it.item_type == "goods"]
    print(f"\n==== 导入后 ({len(goods)} 个货物) ====")
    for g in sorted(goods, key=lambda x: x.bottom_z):
        print(f"  {g.name}: bottom_z={g.bottom_z}, layer={g.layer}, "
              f"pallet_z={g.pallet_z}, cargo_z={g.cargo_z}")

    # 校验
    assert len(goods) == 3, f"期望 3 个货物，实际 {len(goods)}"
    layers = sorted(g.layer for g in goods)
    assert layers == [1, 2, 3], f"层数应为 [1,2,3]，实际 {layers}"
    for orig, imp in zip(items, sorted(goods, key=lambda x: x.bottom_z)):
        assert abs(orig.bottom_z - imp.bottom_z) < 0.001, \
            f"bottom_z 不匹配: {orig.bottom_z} vs {imp.bottom_z}"
        assert orig.layer == imp.layer, \
            f"layer 不匹配: {orig.layer} vs {imp.layer}"
    print("\n[OK] 3 层堆叠 XML 导出/导入完全正确！")


if __name__ == "__main__":
    test_stacking_3layers()
