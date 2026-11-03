"""确认合并后 mast_cam 仍挂在叉车子树里（随车运动）。"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import mujoco

from f_c import merge_warehouse_and_forklift

fl = open("model/forklift_train.xml", encoding="utf-8").read()
wh = open("model/warehouse_env.xml", encoding="utf-8").read()
merged = merge_warehouse_and_forklift(wh, fl)

head = merged.split("<worldbody>")[1].split('<body name="forklift"')[0]
print("worldbody 顶层是否残留裸 camera:", "<camera" in head)

m = mujoco.MjModel.from_xml_string(merged)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "mast_cam")
fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "forklift")
print("camera 所属 body id =", m.cam_bodyid[cid], " forklift =", fid)
print("camera 世界位置 =", np.round(d.cam_xpos[cid], 3))
