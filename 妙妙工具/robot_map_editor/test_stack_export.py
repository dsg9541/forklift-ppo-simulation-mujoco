"""货堆 / 地面区 的 XML 导出与回读自检（Task 2）。"""
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "pylibs"))
sys.path.insert(0, _HERE)

from PyQt5.QtWidgets import QApplication

from items import StackItem, DropZoneItem
import xml_handler
import json_handler


def main():
    app = QApplication([])
    items = [
        StackItem(x=-6.5, y=7.0, layers=3, yaw=90.0, name="stack_A2"),
        StackItem(x=8.0, y=-6.0, layers=3, yaw=-90.0, name="stack_P1"),
        DropZoneItem(x=8.0, y=2.0, yaw=90.0, name="G1"),
    ]
    out_xml = os.path.join(_HERE, "_t_stack.xml")
    out_json = os.path.join(_HERE, "_t_stack_info.json")
    xml_handler.export_xml(items, out_xml)
    json_handler.export_environment_info(items, out_json)

    s = open(out_xml, encoding="utf-8").read()
    assert 'name="stack_A2_L1_pallet"' in s, "缺少 L1 托盘"
    assert 'name="stack_A2_L3_cargo"' in s, "缺少 L3 货物"
    assert 'name="stack_P1_L1_cargo"' in s, "缺少取货堆最下层货物"
    assert 'name="drop_zone_G1"' in s, "缺少地面区"
    assert 'contype="0" conaffinity="0"' in s, "地面区应为无碰撞"
    # 层高检查：cargo body z = bottom_z + 0.4
    assert 'name="stack_A2_L2_cargo" pos="-6.5 7 0.95"' in s, "L2 货物 z 不对"

    back = xml_handler.import_xml(out_xml)
    stacks = [it for it in back if getattr(it, "item_type", "") == "stack"]
    zones = [it for it in back if getattr(it, "item_type", "") == "drop_zone"]
    assert len(stacks) == 2, f"回读货堆数量 {len(stacks)}"
    assert all(st.layers == 3 for st in stacks), "回读层数不对"
    assert len(zones) == 1, "回读地面区数量不对"

    js = json_handler.load_path
    import json
    info = json.load(open(out_json, encoding="utf-8"))
    st = [e for e in info["environment"] if e["type"] == "stack"][0]
    assert st["layers"] == 3 and len(st["layer_bottoms"]) == 3, "信息 JSON 层数不对"
    assert abs(st["top_z"] - 1.65) < 1e-9, "顶面高度不对"
    print("STACK EXPORT OK  (xml/json 导出+回读 均通过)")


if __name__ == "__main__":
    main()
