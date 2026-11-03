"""非 GUI 测试：验证 XML 导出/导入往返。"""
import sys
import os

# 解决 Windows GBK 控制台无法打印 Unicode 字符的问题
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "pylibs"))
sys.path.insert(0, _HERE)

from PyQt5.QtWidgets import QApplication
from items import (ShelfItem, ObstacleItem, GoodsItem, ForkliftItem)
import xml_handler
import json_handler


def test_xml_roundtrip():
    app = QApplication([])

    # 构造一组环境物体
    items = [
        ShelfItem(x=6.0, y=0.0, z=0.0, yaw=0.0, name="shelf_1"),
        ObstacleItem(x=2.0, y=3.0, z=0.0, yaw=0.0, name="obstacle_1"),
        GoodsItem(x=0.0, y=0.0, bottom_z=0.0, yaw=0.0, name="cargo_1"),
        GoodsItem(x=0.0, y=0.0, bottom_z=0.55, yaw=0.0, name="cargo_2"),
        ForkliftItem(x=-6.0, y=0.0, z=0.0, yaw=90.0, name="forklift_init_1"),
    ]

    out = os.path.join(_HERE, "..", "test_env.xml")
    xml_handler.export_xml(items, out)
    print(f"已导出: {out}")

    # 打印 XML 内容
    with open(out, "r", encoding="utf-8") as f:
        print("==== XML 内容 ====")
        print(f.read())
        print("==================")

    # 重新导入
    new_items = xml_handler.import_xml(out)
    print(f"重新导入 {len(new_items)} 个物体")
    for it in new_items:
        kind = it.item_type
        print(f"  - {kind}: name={it.name}, "
              f"pos=({it.x:.2f}, {it.y:.2f}, {getattr(it, 'z', 0):.2f}), "
              f"yaw={it.yaw:.2f}")

    # 校验
    assert len(new_items) == 5, f"期望 5 个，实际 {len(new_items)}"
    # 找到堆叠货物
    goods = [it for it in new_items if it.item_type == "goods"]
    print(f"\n货物数: {len(goods)}，层数: {[g.layer for g in goods]}")
    assert len(goods) == 2
    # layer 2 的货物 bottom_z 应为 0.55
    g2 = [g for g in goods if g.layer == 2][0]
    assert abs(g2.bottom_z - 0.55) < 0.01, f"layer 2 bottom_z={g2.bottom_z}"
    print("✓ 堆叠货物层数正确")

    print("\nXML 测试通过！")

    # 测试环境物体信息 JSON 导出
    print("\n==== 测试环境物体信息 JSON 导出 ====")
    info_path = os.path.join(_HERE, "..", "test_env_info.json")
    json_handler.export_environment_info(items, info_path)
    with open(info_path, "r", encoding="utf-8") as f:
        print(f.read())

    import json as _json
    with open(info_path, "r", encoding="utf-8") as f:
        info = _json.load(f)
    env = info["environment"]
    assert len(env) == 5, f"期望 5 个物体，实际 {len(env)}"
    # 校验货物堆叠信息
    goods_info = [e for e in env if e["type"] == "goods"]
    assert len(goods_info) == 2
    g2 = [g for g in goods_info if g["layer"] == 2][0]
    assert abs(g2["bottom_z"] - 0.55) < 0.01
    assert abs(g2["cargo_z"] - 0.95) < 0.01
    print("环境物体信息 JSON 测试通过！")


if __name__ == "__main__":
    test_xml_roundtrip()
