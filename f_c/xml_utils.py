import re

def fix_repeated_names(xml_str):
    """自动修复 XML 中重复的 name 属性"""
    name_counts = {}
    def replace_name(match):
        name = match.group(1)
        if name not in name_counts:
            name_counts[name] = 0
            return match.group(0)
        else:
            name_counts[name] += 1
            new_name = f"{name}_{name_counts[name]}"
            return f'name="{new_name}"'
    return re.sub(r'name="([^"]+)"', replace_name, xml_str)

def extract_forklift_init(wh_xml):
    """从仓库环境XML中提取 forklift_init 的 pos 和 euler"""
    match = re.search(r'<body name="forklift_init"([^>]+)>', wh_xml)
    if not match:
        return "0 0 0", "0 0 0"
    attrs = match.group(1)
    pos_match = re.search(r'pos="([^"]+)"', attrs)
    euler_match = re.search(r'euler="([^"]+)"', attrs)
    pos_str = pos_match.group(1) if pos_match else "0 0 0"
    euler_str = euler_match.group(1) if euler_match else "0 0 0"
    return pos_str, euler_str

def _extract_body(xml_str, body_name):
    start_tag = f'<body name="{body_name}"'
    start = xml_str.find(start_tag)
    if start < 0:
        return None
    depth = 0
    i = start
    while i < len(xml_str):
        if xml_str[i:i + 6] == '<body ':
            depth += 1
        elif xml_str[i:i + 7] == '</body>':
            depth -= 1
            if depth == 0:
                return xml_str[start:i + 7]
        i += 1
    return None

def merge_warehouse_and_forklift(wh_xml, fl_xml):
    """合并仓库环境 XML 和叉车 XML，将叉车完美插入 forklift_init 位置"""
    merged = re.sub(r'<body name="forklift_init".*?</body>', '', wh_xml, flags=re.DOTALL)
    fl_compiler = re.search(r'<compiler[^>]*/>', fl_xml)
    fl_option = re.search(r'<option[^>]*/>', fl_xml)
    # <visual><map znear/zfar/></visual>：决定相机近/远裁剪面。
    # 不合并的话会用 MuJoCo 默认值（znear×extent ≈ 2m），相机看不到近处，
    # 训练时之所以能看到 2m 内的托盘/货堆，就是靠这一段把 znear 调到 0.0015。
    fl_visual = re.search(r'<visual>.*?</visual>', fl_xml, re.DOTALL)
    fl_body = _extract_body(fl_xml, "forklift")
    if fl_body is None:
        raise ValueError("叉车 XML 中未找到 forklift body")
    
    init_pos_str, init_euler_str = extract_forklift_init(wh_xml)
    # print(f"[INFO] 提取到叉车初始位姿: pos={init_pos_str}, euler={init_euler_str}")
    
    fl_body = re.sub(
        r'<body name="forklift"[^>]*>',
        f'<body name="forklift" pos="{init_pos_str}" euler="{init_euler_str}">',
        fl_body
    )

    fl_floor = re.search(r'<geom name="floor"[^>]*/>', fl_xml)
    fl_light = re.search(r'<light[^>]*/>', fl_xml)
    fl_act_match = re.search(r'<actuator>(.*?)</actuator>', fl_xml, re.DOTALL)
    fl_actuators = fl_act_match.group(1) if fl_act_match else ""

    insert_parts = []
    if fl_light:
        insert_parts.append(fl_light.group(0))
    if fl_floor:
        floor_str = fl_floor.group(0).replace('size="10 10 0.1"', 'size="40 40 0.1"')
        insert_parts.append(floor_str)
    insert_parts.append(fl_body)
    # 注意：不要在这里把叉车里的 <camera> 抽出来插到世界坐标！
    # forklift_train.xml 里的 mast_cam 定义在 mast body 内部，_extract_body 已经把
    # 它连同门架一起放进 fl_body；再抽一次会把它变成固定在世界坐标 (0.4,0,2.2)
    # 的相机，模型看到的就是完全错误的画面（27 号部署时曾因此表现异常）。
    # 同理 camera 必须留在车体子树里，才能随车运动。

    insert_str = "\n    ".join(insert_parts)
    merged = merged.replace('</worldbody>', f'    {insert_str}\n  </worldbody>')

    if '<actuator>' not in merged:
        merged = merged.replace('</mujoco>', f'  <actuator>\n{fl_actuators}  </actuator>\n</mujoco>')

    header_parts = []
    if fl_compiler:
        header_parts.append(fl_compiler.group(0))
    if fl_option:
        header_parts.append(fl_option.group(0))
    if fl_visual:
        header_parts.append(fl_visual.group(0))
    if header_parts:
        header_str = "\n  ".join(header_parts)
        merged = merged.replace('<worldbody>', f'  {header_str}\n  <worldbody>')

    merged = fix_repeated_names(merged)
    return merged
