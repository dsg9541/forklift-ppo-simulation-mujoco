# ======================================================================
#  train_mohu_10.py   —— 基于 v8，只改三处（其余与 v8 完全一致）
#
#   [1] 障碍物生成净空：不生成在【货物堆】与【叉车轨迹(起点→真值目标 线段)】2m 内
#       （原有 起点/引导目标 2m 约束保留；引导噪声 0.2m 的裕量预扣；
#        采样上限 200 次，超限则本回合停用该障碍物，避免死循环）
#   [2] INITIAL_LR = 2e-5
#   [3] DEFAULT_TOTAL_STEPS = 4,000,000
#
#  未改动：v8 的全部奖励项（乘积塑形/停稳/0.2m 引导/终局加减分/末段精细控制开关）、
#          判定门限、观测、PPO 超参、货堆真实物理、碰撞惩罚。
# ======================================================================
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
import argparse
import datetime
import glob
import json
import re
from collections import deque


MAX_ANG = 20
Max_steer = 1.0636  
# ===== 赛题式引导输入：位置误差≥0.2m、姿态误差≥10°，奖励/终止用真值 =====
GUIDANCE_POS_ERR_M = 0.2
GUIDANCE_YAW_ERR_DEG = 10.0
GOODS_MAX_LAYERS = 3

# ================= [v11] 随机光照方向 =================
# 为什么加：v10 及以前，环境里光是"世界固定方向"（<light pos="3 5 4" dir="-1 -1 -1"/>），
# 而训练时目标货堆的朝向≈车头朝向≈世界 +x，于是"货堆 + 它在地面的阴影"成了固定搭配。
# 实测换到真实的仓库场景（货堆相对世界转了 90°）后，同一个模型会停在 10~30cm 外，
# 因为阴影这个强线索方向变了。随机化光照方向后，模型必须学会与阴影方向无关的视觉特征。
RANDOMIZE_LIGHT = True            # 关掉即可回到 v10 行为
LIGHT_AZIMUTH_DEG = (0.0, 360.0)  # 方位角均匀随机
LIGHT_ELEVATION_DEG = (35.0, 70.0)  # 仰角随机范围

# ===== 障碍物生成约束（v10 新增轨迹净空）=====
OBSTACLE_CLEAR_M = 2.0        # 距 起点 / 引导目标 / 货物堆中心（原为硬编码 2.0）
OBSTACLE_TRAJ_CLEAR_M = 2.0   # [新增] 距「叉车轨迹」= 起点→真值目标 的线段
OBSTACLE_SAMPLE_TRIES = 200   # 采样上限；超过就本回合停用该障碍物（防死循环）

# ================= 断点续训：每 1,000,000 步保存一次 =================
CKPT_DIRNAME = "train_model"
CKPT_INTERVAL = 1_000_000
DEFAULT_TOTAL_STEPS = 4_000_000   # [v10] 400 万步
INITIAL_LR = 2e-5          # [v10] 微调起始学习率
FINAL_LR = 5e-6            # 学习率衰减下限


# ===== 最终精度判定门限（成功条件；奖励、终止、指标三处共用同一组常量）=====
SUCCESS_POS_TOL = 0.012      # m   车体到真值目标点的距离
SUCCESS_YAW_TOL = 0.02       # rad 姿态误差（≈1.15°）
SUCCESS_LIN_VEL_TOL = 0.1    # m/s 车体真实速度（见下方 _body_speed 说明）
GOAL_BONUS = 150.0            # 三条同时满足时的一次性奖励

# ===== 接近进度塑形：每靠近真值目标 1 m 给 PROGRESS_GAIN 分，后退同额扣分 =====
# 作用：让"最后 0.3m 慢慢蹭进去"不再净亏（原奖励里这一段几乎没有梯度），
# 同时因为按"前进量"给分，原地停在目标附近不会白拿分。
PROGRESS_GAIN = 10.0
PROGRESS_CLIP = 1.0          # 单步上限/下限，防止物理瞬移造成的奖励尖峰

# ===== 停稳奖励：只在成功窗口外一圈内生效的纯正向奖励（无惩罚）=====
# [v4] 距离半径收到 0.020：v3 那次训练（12:32 启动）用的是 0.025，本文件后来
#      被改成 0.030；但 0.030 会让最后 1cm 的梯度掉到 0.056/步·cm，和"提高精度"
#      相反，所以 v4 收回 0.020 ⇒ 1.2~2.0cm 内梯度 0.125/步·cm（v3 是 0.077）。
#      想更温和可改 0.025。覆盖范围用绝对物理量，改 SUCCESS_* 门限时不用动。
# SETTLE_GAIN 保持 0.10 ≤ 每步时间成本：贴门限赖着也最多不亏，永远不会比成功更赚。
SETTLE_GAIN = 0.10

# ===== [v8] 贴近塑形：单一预算的乘积形式 =====
# shape = SHAPE_CAP × 距离因子 × 角度因子 × 速度因子
#  · 距离因子：距真值目标 SUCCESS_POS_TOL~GUIDE_RANGE_M 线性 1→0（覆盖最后 20cm）
#  · 角度因子：(SHAPE_YAW_MIX + (1-MIX)*角度可比度)，可比度在门限~5° 线性 1→0
#  · 速度因子：同理，门限~0.5m/s
# 乘积形式让每一维在任何状态下都有梯度；总额恒 ≤ SHAPE_CAP，避免"封顶压平梯度"。
SHAPE_CAP = 0.095            # 每步塑形总预算（必须 < 每步时间成本 0.1）

# ===== [v8] 末段精细控制（移植自 v5精度，已验证一步 50ms→25ms）=====
CTRL_FINE_DIST_M = 0.10      # m  距真值目标小于它就切精细控制
ACTION_REPEAT_NEAR = 0      # 原 50；设为 0 可关闭
GUIDE_RANGE_M = 0.200        # m   距离因子作用半径（沿用 v7 的 20cm）
SHAPE_YAW_RANGE_RAD = 0.0873 # rad 5°：角度因子作用范围
SHAPE_VEL_RANGE = 0.50       # m/s 速度因子作用范围
SHAPE_YAW_MIX = 0.35         # 完全歪着时仍保留的距离分比例（避免放弃对齐）
SHAPE_VEL_MIX = 0.30         # 速度不达标时仍保留的比例
SETTLE_RANGE_M = 0.020       # m   [v4] 距离覆盖半径 2.0cm（v3 是 2.5cm）
SETTLE_RANGE_RAD = 0.120     # rad 姿态方向的覆盖半径（≈6.9°）
SETTLE_RANGE_VEL = 0.50      # m/s 速度方向的覆盖半径

# ===== 终局精度项（v4 加强）：超时回合的"收尾位置"要有完整梯度 =====
# 只对**跑满步数仍未达标**的回合生效（成功回合拿 GOAL_BONUS，不重复发）。
# 刻意不覆盖"碰撞/翻车提前终止"：否则策略可能学会故意撞停/翻车来领这份分。
# 停在门限处 +25，到 6cm 处线性衰减到 0；超过 6cm 改为扣分。
# 25 ≪ GOAL_BONUS(150)，不会让模型觉得"停在门口"比"进窗口"更划算。
# 设为 0.0 即可关闭这一项。
TERMINAL_CLOSE_GAIN = 25.0      # [v4] 超时回合停在门限处最多 +25（v3 是 +12）
TERMINAL_CLOSE_RANGE_M = 0.060  # m   [v4] 覆盖半径 6cm（v3 是 3cm）
# ---- v4 新增：超时收尾太远的扣分（v3 实测失败回合几乎都在 3cm 以外）----
# 0.06m 处 0 分，0.26m 及以外封顶 -15，用来把"绕远/迷路"和"差一点"拉开。
TERMINAL_FAR_PENALTY = 15.0     # 设为 0.0 可关闭
TERMINAL_FAR_START_M = 0.060    # m  从这里开始扣
TERMINAL_FAR_FULL_M = 0.260     # m  到这里扣满
# ---- v4 新增：翻车/抬轮提前终止的固定扣分 ----
# v3 之前"unsafe 提前终止"没有任何惩罚，等于比跑满 500 步更划算（少付时间成本）；
# 加上远距离扣分后这个漏洞更值钱，所以补上。实测 unsafe_rate 只有 0.04%，影响很小。
UNSAFE_TERMINATION_PENALTY = 10.0   # 设为 0.0 可关闭

# ===== 与门限挂钩的"指标"阈值（改门限时自动跟随，不用手改）=====
# 近失 = 失败且最接近时离门限还差 NEAR_MISS_RATIO 倍以内（1.5 = 差 50% 以内）
NEAR_MISS_RATIO = 1.5

# ===== 门架相机参数 =====
MAST_CAMERA_NAME = "mast_cam"
CAMERA_HEIGHT = 128
CAMERA_WIDTH = 128
LIDAR_DIM = 279
STATE_DIM = 6

# ================= 自定义策略类，限制 std 下限 =================
class MinStdPolicy(ActorCriticPolicy):
    MIN_STD_VALUE = 0
    
    def __init__(self, *args, **kwargs):
        kwargs.pop('min_std', None)
        self.min_log_std = float(torch.log(torch.tensor(self.MIN_STD_VALUE)).item())
        super().__init__(*args, **kwargs)

    def get_distribution(self, obs):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        mean_actions = self.action_net(latent_pi)
        clamped_log_std = torch.clamp(self.log_std, min=self.min_log_std)
        return self.action_dist.proba_distribution(mean_actions, clamped_log_std)
# ==================================================================

# ================= 视觉输入 PPO 策略（MultiInput + 相机图像） =================
class VisionMinStdPolicy(MultiInputActorCriticPolicy):
    # 原为 0（无下限），策略 std 从 0.44 一路塌到 0.13 后成功率不再提升；
    # 0.05 给探索留个下限，避免过早收敛到确定性的坏策略
    MIN_STD_VALUE = 0.05

    def __init__(self, *args, **kwargs):
        kwargs.pop("min_std", None)
        self.min_log_std = float(torch.log(torch.tensor(self.MIN_STD_VALUE)).item())
        super().__init__(*args, **kwargs)

    def get_distribution(self, obs):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        mean_actions = self.action_net(latent_pi)
        clamped_log_std = torch.clamp(self.log_std, min=self.min_log_std)
        return self.action_dist.proba_distribution(mean_actions, clamped_log_std)


class Lidar1DEncoder(nn.Module):
    """保留 279 维雷达测距的 1D 卷积编码器。"""

    def __init__(self, lidar_dim: int = LIDAR_DIM, features_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out_dim = self.cnn(torch.zeros(1, 1, lidar_dim)).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations.unsqueeze(1)))


class VisionCombinedExtractor(BaseFeaturesExtractor):
    """图像 -> NatureCNN；279 维雷达 -> 1D CNN；低维状态 -> MLP。"""

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        state_hidden_dim: int = 128,
        state_output_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim=1)
        self.image_net = NatureCNN(
            observation_space["image"], features_dim=cnn_output_dim
        )
        self.lidar_net = Lidar1DEncoder(
            int(observation_space["lidar"].shape[0]), features_dim=cnn_output_dim
        )
        # 状态量纲固定归一化：误差/角度/速度/转向/mode 都在相近尺度。
        self.register_buffer(
            "_state_scale",
            torch.tensor(
                [10.0, 10.0, float(np.pi), 5.0, 1.1, 1.0],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.state_net = nn.Sequential(
            nn.Linear(int(observation_space["state"].shape[0]), state_hidden_dim),
            nn.ReLU(),
            nn.Linear(state_hidden_dim, state_output_dim),
            nn.ReLU(),
        )
        self._features_dim = 2 * cnn_output_dim + state_output_dim

    def forward(self, observations) -> torch.Tensor:
        state = observations["state"] / self._state_scale
        features = [
            self.image_net(observations["image"]),
            self.lidar_net(observations["lidar"]),
            self.state_net(state),
        ]
        return torch.cat(features, dim=1)


def vision_policy_kwargs():
    return dict(
        features_extractor_class=VisionCombinedExtractor,
        net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]),
    )
# ==================================================================

# ================= 一维卷积雷达特征提取器 =================
class RadarsCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box):
        super().__init__(observation_space, features_dim=256)
        
        self.lidar_dim = 279
        self.other_dim = observation_space.shape[0] - self.lidar_dim
        
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.lidar_dim)
            cnn_out_dim = self.cnn(dummy).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim + self.other_dim, 256),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        lidar_data = observations[:, :self.lidar_dim].unsqueeze(1)
        other_data = observations[:, self.lidar_dim:]
        
        cnn_features = self.cnn(lidar_data)
        combined = torch.cat((cnn_features, other_data), dim=1)
        return self.linear(combined)
# =========================================================

class ForkliftNavEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, xml_path, max_steps=500, action_repeat=50):
        super().__init__()
        self.xml_path = xml_path
        
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_string = f.read()
            
        goods_layer_xml = ""
        for i in range(GOODS_MAX_LAYERS):
            goods_layer_xml += f"""
        <body name="goods_layer_{i}" pos="-100 0 0">
          <freejoint name="goods_layer_joint_{i}"/>
          <geom name="ground_pallet_top_{i}" type="box" size="0.35 0.40 0.015" pos="0 0 0.210" rgba="0.6 0.4 0.2 1" contype="1" conaffinity="1" mass="15" friction="1.5 1 1"/>
          <geom name="ground_pallet_block_center_{i}" type="box" size="0.35 0.035 0.06" pos="0 0 0.135" rgba="0.3 0.2 0.1 1" contype="1" conaffinity="1" mass="2" friction="1.5 0.1 0.1"/>
          <geom name="ground_pallet_block_left_{i}" type="box" size="0.35 0.04 0.08" pos="0 0.35 0.125" rgba="0.3 0.2 0.1 1" contype="1" conaffinity="1" mass="2" friction="1.5 0.1 0.1"/>
          <geom name="ground_pallet_block_right_{i}" type="box" size="0.35 0.04 0.08" pos="0 -0.35 0.125" rgba="0.3 0.2 0.1 1" contype="1" conaffinity="1" mass="2" friction="1.5 0.1 0.1"/>
        </body>
        <body name="goods_cargo_{i}" pos="-100 0 0">
          <freejoint name="goods_cargo_joint_{i}"/>
          <geom name="ground_cargo_{i}" type="box" size="0.31 0.40 0.15" pos="0 0 0" rgba="1 0.4 0.7 1" contype="1" conaffinity="1" mass="50" friction="2.2 1 1"/>
        </body>
        """
            
        obstacle_xml = ""
        for i in range(10):
            obstacle_xml += f'<body name="obstacle_body_{i}" mocap="true" pos="0 0 -10"><geom name="obstacle_{i}" type="box" size="0.3 0.3 0.3" rgba="0.8 0.2 0.2 1" contype="1" conaffinity="1"/></body>\n'
            
        if '</worldbody>' in xml_string:
            xml_string = xml_string.replace('</worldbody>', goods_layer_xml + obstacle_xml + '</worldbody>')
        else:
            raise ValueError("XML中未找到 </worldbody> 标签")
            
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)

        # ===== 门架相机 + 离屏渲染器 =====
        self.mast_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, MAST_CAMERA_NAME
        )
        if self.mast_cam_id < 0:
            raise ValueError(f"模型 XML 中未找到相机 {MAST_CAMERA_NAME}")
        self._renderer = mujoco.Renderer(
            self.model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH
        )

        # ===== 雷达配置 =====
        self.BODY_RADARS_CONFIG = [
            {"name": "lidar_mast",  "alt_name": "sensor_mast",  "yaw": 0.0,        "half_fov": np.pi/4, "rays": 20},
            {"name": "lidar_left",  "alt_name": "sensor_left",  "yaw": np.pi/2,    "half_fov": np.pi/3, "rays": 25},
            {"name": "lidar_right", "alt_name": "sensor_right", "yaw": -np.pi/2,   "half_fov": np.pi/3, "rays": 25},
            {"name": "lidar_rear",  "alt_name": "sensor_rear",  "yaw": np.pi,      "half_fov": np.pi/3, "rays": 25}
        ]
        self.FORK_POINT_RADARS = [
            {"name": "lidar_fork_left",  "dir": [0, 0, 1]},
            {"name": "lidar_fork_right", "dir": [0, 0, 1]},
            {"name": "lidar_fork_bottom_left",  "dir": [0, 0, -1]},
            {"name": "lidar_fork_bottom_right", "dir": [0, 0, -1]}
        ]
        self.MAST_PLANE_RADARS = [
            "lidar_fork_top_left", "lidar_fork_top_right",
            "lidar_fork_mid_left", "lidar_fork_mid_right"
        ]
        self.MAST_PLANE_LAYERS = 3
        self.MAST_PLANE_RAYS = 15
        self.MAST_PLANE_LAYER_ANGLE = np.deg2rad(10.0)

        self.lidar_max_dist = 8.0
        total_rays = 95 + 4 + (4 * self.MAST_PLANE_LAYERS * self.MAST_PLANE_RAYS)

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, CAMERA_HEIGHT, CAMERA_WIDTH),
                    dtype=np.uint8,
                ),
                "lidar": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(LIDAR_DIM,), dtype=np.float32
                ),
                "state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.target_pos = None
        self.target_yaw = 0.0
        self.true_target_pos = None
        self.true_target_yaw = 0.0
        self.progress = 1.0
        #--------------------------------
        self.guidance_noise_scale = 1.0
        #-------------------------------
        self.mode = 0
        self.max_lin_speed = 2.0

        self.step_count = 0
        self.max_episode_steps = max_steps
        self.action_repeat = action_repeat

        self.forklift_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "forklift")
        self.mast_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mast")
        self.forks_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "forks")
        self.pallet_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pallet1")
        self.cargo_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cargo1")

        self._fork_mast_geom_ids = set()
        for gname in ["left_fork", "right_fork", "mast_back"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                self._fork_mast_geom_ids.add(gid)

        self._pallet_geom_ids = set()
        for gname in ["pallet_top", "pallet_block_center", "pallet_block_left", "pallet_block_right"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                self._pallet_geom_ids.add(gid)

        self._cargo_geom_ids = set()
        for i in range(self.model.ngeom):
            if self.model.geom_bodyid[i] == self.cargo_body_id:
                self._cargo_geom_ids.add(i)

        self._tire_geom_ids = []
        for tname in ["fl_tire", "fr_tire", "rl_tire", "rr_tire"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, tname)
            if gid >= 0:
                self._tire_geom_ids.append(gid)

        self.goods_layer_body_ids = []
        self.goods_layer_mocap_ids = []
        # [v6] 货堆层改造成 freejoint 动力学体后，摆放要用 qpos/qvel 地址
        self.goods_layer_qpos_adr = []
        self.goods_layer_dof_adr = []
        # [v6] 每层的货物箱也是独立动力学体
        self.goods_cargo_body_ids = []
        self.goods_cargo_qpos_adr = []
        self.goods_cargo_dof_adr = []
        self._goods_geom_ids = set()
        self.goods_mocap_id = -1
        self.goods_body_id = -1
        for i in range(GOODS_MAX_LAYERS):
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"goods_layer_{i}"
            )
            if bid >= 0:
                self.goods_layer_body_ids.append(bid)
                self.goods_layer_mocap_ids.append(self.model.body_mocapid[bid])
                # [v6] 取该层 freejoint 的 qpos/qvel 起始下标
                _gjid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"goods_layer_joint_{i}"
                )
                self.goods_layer_qpos_adr.append(int(self.model.jnt_qposadr[_gjid]))
                self.goods_layer_dof_adr.append(int(self.model.jnt_dofadr[_gjid]))
                for gid in range(self.model.ngeom):
                    if self.model.geom_bodyid[gid] == bid:
                        self._goods_geom_ids.add(gid)
        # [v6] 货物箱：独立的 freejoint 体，geom 也要算进 _goods_geom_ids 供碰撞检测
        for i in range(GOODS_MAX_LAYERS):
            cbid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"goods_cargo_{i}"
            )
            if cbid >= 0:
                self.goods_cargo_body_ids.append(cbid)
                _cjid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"goods_cargo_joint_{i}"
                )
                self.goods_cargo_qpos_adr.append(int(self.model.jnt_qposadr[_cjid]))
                self.goods_cargo_dof_adr.append(int(self.model.jnt_dofadr[_cjid]))
                for gid in range(self.model.ngeom):
                    if self.model.geom_bodyid[gid] == cbid:
                        self._goods_geom_ids.add(gid)

        if self.goods_layer_mocap_ids:
            self.goods_mocap_id = self.goods_layer_mocap_ids[0]
            self.goods_body_id = self.goods_layer_body_ids[0]

        self.has_goods = False
        self.goods_layer_count = 0
        self.obstacle_tries = []      # [v10] 障碍物采样次数记录
        self.goods_collision_penalty_applied = False
        self.goods_center = None
        self.is_forward = False

        root_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root_joint")
        self.root_qpos_adr = self.model.jnt_qposadr[root_joint_id]
        self.root_dof_adr = self.model.jnt_dofadr[root_joint_id]

        pallet_jnt = self.model.body_jntadr[self.pallet_body_id]
        cargo_jnt = self.model.body_jntadr[self.cargo_body_id]
        self.pallet_qpos_adr = self.model.jnt_qposadr[pallet_jnt]
        self.cargo_qpos_adr = self.model.jnt_qposadr[cargo_jnt]
        self.pallet_dof_adr = self.model.jnt_dofadr[pallet_jnt]
        self.cargo_dof_adr = self.model.jnt_dofadr[cargo_jnt]

        self.fork_left_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar_fork_left")
        self.fork_right_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar_fork_right")

        lift_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lift_joint")
        self.lift_qpos_adr = self.model.jnt_qposadr[lift_joint_id]

        self.has_cargo = False
        self.cargo_collision_penalty_applied = False

        # ===== 力矩控制参数 =====
        self.KP_SPEED = 1000.0; self.KI_SPEED = 50.0; self.KD_SPEED = 20.0
        self.INTEGRAL_MAX = 1000.0; self.MAX_TORQUE = 2000.0
        self.FEEDFORWARD_GAIN = 200.0
        self.WHEEL_RADIUS = 0.2; self.L_AXLE = 1.8; self.W_TRACK = 1.0
        self.MAX_STEER = Max_steer
        self.MAX_STEER_RATE = 4.0  

        self.integral_left = 0.0; self.integral_right = 0.0
        self.last_error_left = 0.0; self.last_error_right = 0.0
        self.current_steer = 0.0

        self.last_dist = None
        self.last_e_theta = None
        self.unsafe_counter = 0
        self.max_unsafe_steps = 500
        self.collision_penalty_applied = False
        
        # ===== 记录上一步转向动作，用于惩罚频率过快 =====
        self.last_steer_action = 0.0
        self.prev_steer_action = 0.0  # 记录上上步转向动作

        # ===== 课程学习进度 (1.0开始，0.0结束) =====
        self.progress = 1.0
        
        # ===== 本回合是否成功 =====
        self.is_success = False

        # ===== 本回合统计量（回合结束时写进 info，供 TensorBoard 指标使用）=====
        self.last_true_dist = None       # 上一步到真值目标的距离（进度塑形用）
        self.ep_min_dist = float("inf")  # 本回合最近到过多少
        self.ep_best_gate_ratio = float("inf")  # 离成功门限最近时差几倍（1.0 = 成功）
        self.ep_final_dist = 0.0
        self.ep_final_yaw = 0.0
        self.ep_final_speed = 0.0
        self.ep_shape_sum = 0.0          # [v8] 本回合累计的贴近塑形奖励
        self.ep_prox_sum = 0.0           # [v8] 本回合每步"距离因子"之和（/步数 = 平均贴近度）
        self.ep_terminal_close_sum = 0.0  # 本回合拿到的终局精度分
        self.ep_terminal_far_penalty = 0.0  # 本回合因收尾太远被扣的分

    def update_progress(self, progress):
        self.progress = progress
        self.guidance_noise_scale = 1.0

    def _apply_guidance_noise(self):
        """把“提供给策略的引导目标”降质，真值只用于奖励/终止。"""
        if self.true_target_pos is None or self.guidance_noise_scale <= 0.0:
            return
        scale = float(self.guidance_noise_scale)
        mag = self.np_random.uniform(
            GUIDANCE_POS_ERR_M, GUIDANCE_POS_ERR_M * 1.0
        ) * scale
        ang = self.np_random.uniform(0.0, 2.0 * np.pi)
        offset = mag * np.array([np.cos(ang), np.sin(ang)])
        yaw_mag = np.deg2rad(self.np_random.uniform(
            GUIDANCE_YAW_ERR_DEG, GUIDANCE_YAW_ERR_DEG * 1.0
        )) * scale
        if self.np_random.random() < 0.5:
            yaw_mag = -yaw_mag
        self.target_pos = self.true_target_pos.copy()
        self.target_pos[:2] = self.true_target_pos[:2] + offset
        self.target_yaw = float(self.true_target_yaw + yaw_mag)

    def _set_goods_layers(self, count, cx, cy, yaw):
        """[v6] 生成 0~3 层货物堆；层体是带 freejoint 的动力学刚体，会自然下坠落地。

        与 v4 的区别只在"怎么摆"：v4 写 mocap_pos/mocap_quat（运动学体，不受重力），
        这里写 freejoint 的 qpos/qvel 并把重力打开（gravcomp=0），于是货堆会自然落稳。
        """
        self.goods_layer_count = int(count)
        self.goods_collision_penalty_applied = False
        if count <= 0 or not self.goods_layer_body_ids:
            self.has_goods = False
            self.goods_center = None
            for i in range(len(self.goods_layer_body_ids)):
                self._park_goods_layer(i)
            return
        self.has_goods = True
        self.goods_center = np.array([float(cx), float(cy)])
        half = float(yaw) / 2.0
        quat = [np.cos(half), 0.0, 0.0, np.sin(half)]
        for i in range(GOODS_MAX_LAYERS):
            if i < count:
                self._place_goods_layer(i, float(cx), float(cy), i * 0.52, quat)
                # 货物箱放在托盘面上方 ~2.5cm，让它自己落到托盘上（不再悬空）
                self._place_goods_cargo(i, float(cx), float(cy), i * 0.52 + 0.40, quat)
            else:
                self._park_goods_layer(i)
                self._park_goods_cargo(i)

    def _place_goods_layer(self, i, cx, cy, z, quat):
        """[v6] 把第 i 层货堆放到指定位置：写 qpos/qvel 并清零速度，之后自然下坠落稳。"""
        a = self.goods_layer_qpos_adr[i]
        d = self.goods_layer_dof_adr[i]
        self.data.qpos[a:a + 3] = [cx, cy, float(z)]
        self.data.qpos[a + 3:a + 7] = quat
        self.data.qvel[d:d + 6] = 0.0

    def _park_goods_layer(self, i):
        """[v6] 本回合用不到的层：挪到很远、且在地面之上，让它安静地落在地板上。

        不能丢到 z=-100：地板是无限平面，动态体在那会被当作深度穿透而弹飞。
        """
        a = self.goods_layer_qpos_adr[i]
        d = self.goods_layer_dof_adr[i]
        self.data.qpos[a:a + 3] = [-100.0, 0.0, 0.05]
        self.data.qpos[a + 3:a + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[d:d + 6] = 0.0

    def _place_goods_cargo(self, i, cx, cy, z, quat):
        """[v6] 第 i 层的货物箱：放到托盘上方一点，让它自然落到托盘面上。"""
        a = self.goods_cargo_qpos_adr[i]
        d = self.goods_cargo_dof_adr[i]
        self.data.qpos[a:a + 3] = [cx, cy, float(z)]
        self.data.qpos[a + 3:a + 7] = quat
        self.data.qvel[d:d + 6] = 0.0

    def _park_goods_cargo(self, i):
        """[v6] 用不到的货物箱：同样挪到很远的地面上方。"""
        a = self.goods_cargo_qpos_adr[i]
        d = self.goods_cargo_dof_adr[i]
        self.data.qpos[a:a + 3] = [-100.0, 0.0, 0.40]
        self.data.qpos[a + 3:a + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[d:d + 6] = 0.0

    def _sample_goods_layers(self):
        """无货物堆 50%，有货堆时 1/2/3 层等概率。"""
        if self.np_random.random() < 0.0:
            return 0
        return int(self.np_random.integers(1, GOODS_MAX_LAYERS + 1))

    def _get_curriculum_max_angle(self):
        training_progress = 1.0 - self.progress
        if training_progress < 0.0:
            return MAX_ANG * (training_progress / 0.0)
        return MAX_ANG

    def _render_camera(self):
        """渲染门架相机 RGB，输出为 CHW uint8。"""
        self._renderer.update_scene(self.data, camera=MAST_CAMERA_NAME)
        rgb = self._renderer.render()
        return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))

    @staticmethod
    def _dist_point_to_segment(px, py, ax, ay, bx, by):
        """点 (px,py) 到线段 AB 的最短距离（轨迹净空用）。"""
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        denom = abx * abx + aby * aby
        if denom <= 1e-12:
            return float(np.hypot(apx, apy))
        tt = float(np.clip((apx * abx + apy * aby) / denom, 0.0, 1.0))
        return float(np.hypot(px - (ax + tt * abx), py - (ay + tt * aby)))

    def _obstacle_pos_ok(self, x, y):
        """障碍物落点是否满足净空：起点 / 引导目标 / 货物堆 > 2m，轨迹 > 2m。"""
        if float(np.hypot(x, y)) <= OBSTACLE_CLEAR_M:
            return False
        # 引导噪声是在障碍物摆好后施加的，所以这里预扣 GUIDANCE_POS_ERR_M 的裕量
        if (self.target_pos is not None and float(np.hypot(
                x - self.target_pos[0], y - self.target_pos[1]))
                <= OBSTACLE_CLEAR_M + GUIDANCE_POS_ERR_M):
            return False
        if (self.goods_center is not None and float(np.hypot(
                x - self.goods_center[0], y - self.goods_center[1])) <= OBSTACLE_CLEAR_M):
            return False
        if self.true_target_pos is not None:
            _d = self._dist_point_to_segment(
                x, y, 0.0, 0.0,
                float(self.true_target_pos[0]), float(self.true_target_pos[1]))
            if _d <= OBSTACLE_TRAJ_CLEAR_M:
                return False
        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # ===== [v11] 每回合随机光照方向（见文件顶部 RANDOMIZE_LIGHT 说明）=====
        if RANDOMIZE_LIGHT:
            az = np.deg2rad(self.np_random.uniform(*LIGHT_AZIMUTH_DEG))
            el = np.deg2rad(self.np_random.uniform(*LIGHT_ELEVATION_DEG))
            d = np.array([-np.cos(el) * np.cos(az),
                          -np.cos(el) * np.sin(az),
                          -np.sin(el)])
            d /= np.linalg.norm(d)
            lid = 0                                   # 环境里只有一个 <light>
            if self.model.nlight > lid:
                self.model.light_dir[lid] = d
                self.model.light_pos[lid] = -5.0 * d  # 与原始 pos(3,5,4)/dir(-1,-1,-1) 同量级

        self.max_lin_speed = 2.0
        self.is_success = False

        self.has_cargo = self.np_random.random() < 0.5
        self.cargo_collision_penalty_applied = False

        self.has_goods = False
        self.goods_collision_penalty_applied = False
        self.goods_center = None
        self.is_forward = False

        max_angle_deg = self._get_curriculum_max_angle()

        if self.np_random.random() < 0.0:
            # ============ 后退场景 ============
            self.is_forward = False
            target_dist = self.np_random.uniform(1.0, 7.0)
            turn_angle = self.np_random.uniform(0, np.deg2rad(max_angle_deg))
            turn_dir = self.np_random.choice([1, -1])
            self.target_yaw = -turn_dir * turn_angle
            half_angle = turn_angle / 2.0
            ideal_x = -target_dist * np.cos(half_angle)
            ideal_y = turn_dir * target_dist * np.sin(half_angle)
            self.target_pos = np.array([
                ideal_x + self.np_random.uniform(-0.2, 0.2),
                ideal_y + self.np_random.uniform(-0.2, 0.2),
                0.0
            ])
            self.true_target_pos = self.target_pos.copy()
            self.true_target_yaw = float(self.target_yaw)
            # 叉车前方 2.85m 的货物堆：50% 无货，有货时 1~3 层等概率
            n_layers = self._sample_goods_layers()
            self._set_goods_layers(n_layers, 2.85, 0.0, 0.0)
            # =======================================================
        else:
            # ============ 前进场景 ============
            self.is_forward = True
            target_dist = self.np_random.uniform(1.0, 7.0)
            turn_dir = self.np_random.choice([1, -1])
            turn_angle = self.np_random.uniform(np.deg2rad(0), np.deg2rad(max_angle_deg))
            half_angle = turn_angle / 2.0
            ideal_x = target_dist * np.cos(half_angle)
            ideal_y = turn_dir * target_dist * np.sin(half_angle)
            ideal_yaw = turn_dir * turn_angle
            self.target_pos = np.array([
                ideal_x + self.np_random.uniform(-0.2, 0.2),
                ideal_y + self.np_random.uniform(-0.2, 0.2),
                0.0
            ])
            self.target_yaw = ideal_yaw + self.np_random.uniform(-0.1, 0.1)
            self.true_target_pos = self.target_pos.copy()
            self.true_target_yaw = float(self.target_yaw)

            # 前进目标点后 2.85m 的货物堆：50% 无货，有货时 1~3 层等概率
            n_layers = self._sample_goods_layers()
            cx = (self.true_target_pos[0]
                  + 2.85 * np.cos(self.true_target_yaw))
            cy = (self.true_target_pos[1]
                  + 2.85 * np.sin(self.true_target_yaw))
            self._set_goods_layers(n_layers, cx, cy, self.true_target_yaw)

        # ===== 模式切换：仅前进且有货物堆时 mode=1，其余情况 mode=0 =====
        self.mode = 1 if (self.is_forward and self.has_goods) else 0
        if self.mode == 0:
            self.target_pos = self.true_target_pos.copy()
            self.target_yaw = float(self.true_target_yaw)

        self.data.qpos[self.root_qpos_adr:self.root_qpos_adr+3] = [0.0, 0.0, 0.0]
        self.data.qpos[self.root_qpos_adr+3:self.root_qpos_adr+7] = [1, 0, 0, 0]
        self.data.qvel[self.root_dof_adr:self.root_dof_adr+6] = 0.0

        lift_target = 0.4 if self.has_cargo else -0.23
        self.data.qpos[self.lift_qpos_adr] = lift_target
        self.data.ctrl[4] = lift_target

        self.collision_penalty_applied = False
        self.obstacle_tries = []      # [v10] 本回合障碍物采样次数（验证用）
        num_obs = self.np_random.integers(1, 6)
        active_indices = self.np_random.choice(10, num_obs, replace=False)
        
        for i in range(10):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_body_{i}")
            mocap_id = self.model.body_mocapid[body_id]
            if i in active_indices:
                # [v10] 采样必须同时满足：起点/引导目标/货物堆 2m + 叉车轨迹 2m
                _placed = False
                for _try in range(OBSTACLE_SAMPLE_TRIES):
                    x = self.np_random.uniform(-7.0, 7.0)
                    y = self.np_random.uniform(-7.0, 7.0)
                    if self._obstacle_pos_ok(x, y):
                        _placed = True
                        break
                self.obstacle_tries.append(_try + 1 if _placed else OBSTACLE_SAMPLE_TRIES)
                if _placed:
                    self.data.mocap_pos[mocap_id] = [x, y, 0.3]
                else:
                    self.data.mocap_pos[mocap_id] = [0.0, 0.0, -10.0]
            else:
                self.data.mocap_pos[mocap_id] = [0.0, 0.0, -10.0]

        if self.mode == 1:
            self._apply_guidance_noise()
        mujoco.mj_forward(self.model, self.data)

        if self.has_cargo:
            site_pos = (self.data.site_xpos[self.fork_left_site_id] +
                        self.data.site_xpos[self.fork_right_site_id]) / 2.0
            tray_pos = site_pos + np.array([0.32, 0.0, -0.11])
            cargo_pos = tray_pos + np.array([0, 0.0, 0.135 + 0.17])

            self.data.qpos[self.pallet_qpos_adr:self.pallet_qpos_adr+3] = tray_pos
            self.data.qpos[self.pallet_qpos_adr+3:self.pallet_qpos_adr+7] = np.array([1, 0, 0, 0])
            self.data.qpos[self.cargo_qpos_adr:self.cargo_qpos_adr+3] = cargo_pos
            self.data.qpos[self.cargo_qpos_adr+3:self.cargo_qpos_adr+7] = np.array([1, 0, 0, 0])

            self.data.qvel[self.pallet_dof_adr:self.pallet_dof_adr+6] = 0.0
            self.data.qvel[self.cargo_dof_adr:self.cargo_dof_adr+6] = 0.0

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.step_count = 0
        self.integral_left = 0.0; self.integral_right = 0.0
        self.last_error_left = 0.0; self.last_error_right = 0.0
        self.current_steer = 0.0
        self.last_steer_action = 0.0
        self.prev_steer_action = 0.0

        obs = self._get_obs()
        state = obs["state"]
        self.last_dist = np.sqrt(state[0] ** 2 + state[1] ** 2)
        self.last_e_theta = float(state[2])
        self.unsafe_counter = 0

        # ===== 回合统计初始化 =====
        e_x0, e_y0, _ = self._true_errors()
        self.last_true_dist = float(np.hypot(e_x0, e_y0))
        self.ep_min_dist = float("inf")
        self.ep_best_gate_ratio = float("inf")
        self.ep_final_dist = 0.0
        self.ep_final_yaw = 0.0
        self.ep_final_speed = 0.0
        self.ep_shape_sum = 0.0          # [v8] 本回合累计的贴近塑形奖励
        self.ep_prox_sum = 0.0           # [v8] 本回合每步"距离因子"之和（/步数 = 平均贴近度）
        self.ep_terminal_close_sum = 0.0  # 本回合拿到的终局精度分
        self.ep_terminal_far_penalty = 0.0  # 本回合因收尾太远被扣的分

        return obs, {}

    def step(self, action):
        v_cmd = float(action[0])
        steer_cmd = float(action[1]) * self.MAX_STEER

        # ===== [v8] 末段精细控制：贴近真值目标时缩小控制步长 =====
        # 同时按比例补偿底层 PID 的 dt/转向速率，保持控制器每秒行为不变。
        _ex0, _ey0, _ = self._true_errors()
        _d_now = float(np.hypot(_ex0, _ey0))
        if 0 < ACTION_REPEAT_NEAR < self.action_repeat and _d_now < CTRL_FINE_DIST_M:
            repeat = ACTION_REPEAT_NEAR
        else:
            repeat = self.action_repeat
        ctrl_scale = repeat / float(self.action_repeat)
        for _ in range(repeat):
            self._apply_control(v_cmd, steer_cmd, dt_scale=ctrl_scale)
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        terminated = self._check_terminated(obs)
        truncated = self.step_count >= self.max_episode_steps

        # ===== 本步统计：末端误差 / 最近距离 / 离成功门限最近的距离 =====
        e_x, e_y, e_theta = self._true_errors()
        dist = float(np.hypot(e_x, e_y))
        yaw_abs = abs(float(e_theta))
        speed_abs = abs(float(obs["state"][3]))
        self.ep_min_dist = min(self.ep_min_dist, dist)
        self.ep_best_gate_ratio = min(
            self.ep_best_gate_ratio, self._gate_ratio(dist, yaw_abs, speed_abs))
        self.ep_final_dist = dist
        self.ep_final_yaw = yaw_abs
        self.ep_final_speed = speed_abs
        self.last_true_dist = dist

        # ===== 终局项（v4）：超时回合的收尾位置按距离给分/扣分 =====
        # 加分：门限处 +TERMINAL_CLOSE_GAIN，到 TERMINAL_CLOSE_RANGE_M 衰减到 0；
        # 扣分：超过 TERMINAL_FAR_START_M 后线性扣，到 TERMINAL_FAR_FULL_M 封顶。
        # 只认 truncated（跑满 max_episode_steps）：碰撞/翻车提前终止不参与，避免被利用。
        if truncated and not self.is_success:
            if TERMINAL_CLOSE_GAIN > 0.0:
                _span = max(TERMINAL_CLOSE_RANGE_M - SUCCESS_POS_TOL, 1e-9)
                _close_frac = float(np.clip(
                    1.0 - (dist - SUCCESS_POS_TOL) / _span, 0.0, 1.0))
                self.ep_terminal_close_sum = TERMINAL_CLOSE_GAIN * _close_frac
                reward += self.ep_terminal_close_sum
            if TERMINAL_FAR_PENALTY > 0.0:
                _fspan = max(TERMINAL_FAR_FULL_M - TERMINAL_FAR_START_M, 1e-9)
                _far_frac = float(np.clip(
                    (dist - TERMINAL_FAR_START_M) / _fspan, 0.0, 1.0))
                self.ep_terminal_far_penalty = TERMINAL_FAR_PENALTY * _far_frac
                reward -= self.ep_terminal_far_penalty
        # 翻车/抬轮导致的提前终止：补固定扣分，避免"提前翻车"比"跑满步数"划算。
        elif (terminated and not self.is_success
                and not self.cargo_collision_penalty_applied):
            reward -= UNSAFE_TERMINATION_PENALTY

        self.prev_steer_action = self.last_steer_action
        self.last_steer_action = float(action[1])
        
        info = {}
        if terminated or truncated:
            info["is_success"] = self.is_success
            # 新增：返回回合类型，用于分类统计
            info["is_forward"] = self.is_forward
            info["mode"] = self.mode

            # ===== 失败模式 / 末端误差，供 TensorBoard 指标使用 =====
            if self.is_success:
                end_reason = "success"
            elif self.cargo_collision_penalty_applied:
                end_reason = "cargo_collision"
            elif terminated:
                end_reason = "unsafe"          # 倾斜/轮子抬起连续超限
            else:
                end_reason = "timeout"         # 跑满 max_episode_steps
            info["end_reason"] = end_reason
            info["obstacle_collision"] = bool(self.collision_penalty_applied)
            info["goods_collision"] = bool(self.goods_collision_penalty_applied)
            info["cargo_collision"] = bool(self.cargo_collision_penalty_applied)
            info["final_dist"] = float(self.ep_final_dist)
            info["final_yaw_deg"] = float(np.rad2deg(self.ep_final_yaw))
            info["final_speed"] = float(self.ep_final_speed)
            info["min_dist"] = float(self.ep_min_dist)
            info["best_gate_ratio"] = float(self.ep_best_gate_ratio)
            info["shape_bonus_sum"] = float(self.ep_shape_sum)
            info["prox_factor_sum"] = float(self.ep_prox_sum)
            info["terminal_close_bonus"] = float(self.ep_terminal_close_sum)
            info["terminal_far_penalty"] = float(self.ep_terminal_far_penalty)
            info["episode_steps"] = int(self.step_count)
        
        return obs, reward, terminated, truncated, info
    
    def _body_speed(self):
        """车体真实前进速度（m/s，前进为正）。

        root 是自由关节，qvel[0:3] 是世界系的线速度；投影到车身 x 轴得到带符号的前进速度。
        为什么不用轮子转速：轮子角速度是 1kHz 级别的瞬时量，单个采样点抖动极大
        （实测与真实车速的相关性只有 0.25 / -0.01），而它同时被用作观测通道和成功判定，
        会让"停稳"既学不到也判不准。轮子转速仍然用于 _apply_control 里的驱动/刹车闭环。
        """
        v_world = self.data.qvel[self.root_dof_adr:self.root_dof_adr + 3]
        x_axis = self.data.xmat[self.forklift_body_id].reshape(3, 3)[:, 0]
        return float(np.dot(v_world, x_axis))

    def _get_obs(self):
        lidar_data = self._get_lidar_scan()
        pos, rot, yaw = self._get_forklift_pose()
        dx = self.target_pos[0] - pos[0]; dy = self.target_pos[1] - pos[1]
        e_x =  dx * np.cos(yaw) + dy * np.sin(yaw)
        e_y = -dx * np.sin(yaw) + dy * np.cos(yaw)
        e_theta = self.target_yaw - yaw
        e_theta = np.arctan2(np.sin(e_theta), np.cos(e_theta))

        lin_vel = self._body_speed()
        steer = self.current_steer

        state = np.array(
            [e_x, e_y, e_theta, lin_vel, steer, float(self.mode)],
            dtype=np.float32,
        )
        return {
            "image": self._render_camera(),
            "lidar": lidar_data,
            "state": state,
        }

    def _get_lidar_scan(self):
        R_forklift = self.data.xmat[self.forklift_body_id].reshape(3, 3)
        R_mast = self.data.xmat[self.mast_body_id].reshape(3, 3) if self.mast_body_id >= 0 else np.eye(3)
        R_forks = self.data.xmat[self.forks_body_id].reshape(3, 3) if self.forks_body_id >= 0 else np.eye(3)
        geomid_out = np.zeros(1, dtype=np.int32)
        lidar_data = []

        # 1. 车体水平雷达 (95) —— 不再过滤托盘/货物
        for cfg in self.BODY_RADARS_CONFIG:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, cfg["name"])
            if site_id < 0:
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, cfg["alt_name"])
            if site_id < 0:
                lidar_data.extend([1.0] * cfg["rays"])
                continue
                
            lidar_pos = self.data.site_xpos[site_id].copy()
            for i in range(cfg["rays"]):
                if cfg["rays"] > 1:
                    angle = -cfg["half_fov"] + 2 * cfg["half_fov"] * (i / (cfg["rays"] - 1))
                else:
                    angle = 0.0
                local_angle = cfg["yaw"] + angle
                local_dir = np.array([np.cos(local_angle), np.sin(local_angle), 0.0])
                ray_dir = R_forklift @ local_dir
                ray_dir /= np.linalg.norm(ray_dir)
                
                dist = mujoco.mj_ray(self.model, self.data, lidar_pos, ray_dir, 
                                     None, False, 
                                     self.forklift_body_id, geomid_out)
                
                if dist < 0 or dist > self.lidar_max_dist:
                    dist = self.lidar_max_dist
                lidar_data.append(dist / self.lidar_max_dist)

        # 2. 货叉上下单束射线 (4) —— 不再过滤托盘/货物
        for cfg in self.FORK_POINT_RADARS:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, cfg["name"])
            if site_id < 0:
                lidar_data.append(1.0)
                continue
            pnt = self.data.site_xpos[site_id].copy()
            local_dir = np.array(cfg["dir"])
            vec = R_forks @ local_dir
            vec /= np.linalg.norm(vec)
            dist = mujoco.mj_ray(self.model, self.data, pnt, vec, 
                                 None, False, 
                                 self.forklift_body_id, geomid_out)
            
            if dist < 0 or dist > self.lidar_max_dist:
                dist = self.lidar_max_dist
            lidar_data.append(dist / self.lidar_max_dist)

        # 3. 门架垂直平面射线 (180) —— 不再过滤托盘/货物
        for radar_name in self.MAST_PLANE_RADARS:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, radar_name)
            if site_id < 0:
                lidar_data.extend([1.0] * (self.MAST_PLANE_LAYERS * self.MAST_PLANE_RAYS))
                continue
            pnt = self.data.site_xpos[site_id].copy()
            
            for layer in range(self.MAST_PLANE_LAYERS):
                layer_yaw = (layer - (self.MAST_PLANE_LAYERS - 1) / 2.0) * self.MAST_PLANE_LAYER_ANGLE
                
                for i in range(self.MAST_PLANE_RAYS):
                    theta = (i / (self.MAST_PLANE_RAYS - 1)) * (np.pi / 2.0)
                    local_dir = np.array([
                        np.cos(theta) * np.cos(layer_yaw),
                        np.cos(theta) * np.sin(layer_yaw),
                        -np.sin(theta)
                    ])
                    
                    vec = R_mast @ local_dir
                    vec /= np.linalg.norm(vec)
                    dist = mujoco.mj_ray(self.model, self.data, pnt, vec, 
                                         None, False, 
                                         self.forklift_body_id, geomid_out)
                    
                    if dist < 0 or dist > self.lidar_max_dist:
                        dist = self.lidar_max_dist
                    lidar_data.append(dist / self.lidar_max_dist)
                
        return np.array(lidar_data, dtype=np.float32)

    def _true_errors(self):
        """奖励/终止统一使用真值，而不是降质后的引导目标。"""
        pos, rot, yaw = self._get_forklift_pose()
        dx = self.true_target_pos[0] - pos[0]
        dy = self.true_target_pos[1] - pos[1]
        e_x = dx * np.cos(yaw) + dy * np.sin(yaw)
        e_y = -dx * np.sin(yaw) + dy * np.cos(yaw)
        e_theta = self.true_target_yaw - yaw
        e_theta = np.arctan2(np.sin(e_theta), np.cos(e_theta))
        return e_x, e_y, e_theta

    def _gate_ratio(self, dist, yaw_abs, speed_abs):
        """离成功窗口还差几倍：三个条件各自除以门限后取最大；1.0 = 刚好达标。

        奖励里的"停稳"项、回合统计、TensorBoard 指标共用这一个口径。
        """
        return max(dist / SUCCESS_POS_TOL,
                   yaw_abs / SUCCESS_YAW_TOL,
                   speed_abs / SUCCESS_LIN_VEL_TOL)

    def _compute_reward(self, obs, action):
        e_x, e_y, e_theta = self._true_errors(); lin_vel = float(obs["state"][3])
        dist = np.sqrt(e_x**2 + e_y**2)

        reward = 0.0

        t_time = -0.10
        reward += t_time

        r_time_dist = -0.20 * (dist / 50.0)
        r_time_angle = -0.40 * (abs(e_theta) / np.pi)
        reward += r_time_dist + r_time_angle

        r_goal = 0.0
        if (dist < SUCCESS_POS_TOL and abs(e_theta) < SUCCESS_YAW_TOL
                and abs(lin_vel) < SUCCESS_LIN_VEL_TOL):
            r_goal = GOAL_BONUS
        reward += r_goal

        # ===== 接近进度塑形：每靠近真值目标 1 m 得 PROGRESS_GAIN 分，后退同额扣分 =====
        # 只按"本步的前进量"给分，所以停在目标附近原地不动不会白拿分。
        if self.last_true_dist is not None:
            progress = self.last_true_dist - dist
            reward += float(np.clip(PROGRESS_GAIN * progress,
                                    -PROGRESS_CLIP, PROGRESS_CLIP))

        # ===== [v8] 贴近塑形（乘积式单预算）=====
        # 距离/角度/速度各自归一化成 [0,1] 的"可比度"，再相乘 × SHAPE_CAP：
        #  · 门限内 = 1；到各自作用范围处 = 0
        #  · 角度/速度再各加一个下限系数(MIX)，避免"歪了就干脆放弃"
        #  · 乘积 ⇒ 每一项在任何状态下都有活梯度；总额恒 ≤ SHAPE_CAP(<0.1)
        _prox = float(np.clip(
            1.0 - (dist - SUCCESS_POS_TOL) / max(GUIDE_RANGE_M - SUCCESS_POS_TOL, 1e-9),
            0.0, 1.0))
        _yaw_ok = float(np.clip(
            1.0 - (abs(float(e_theta)) - SUCCESS_YAW_TOL)
            / max(SHAPE_YAW_RANGE_RAD - SUCCESS_YAW_TOL, 1e-9), 0.0, 1.0))
        _vel_ok = float(np.clip(
            1.0 - (abs(lin_vel) - SUCCESS_LIN_VEL_TOL)
            / max(SHAPE_VEL_RANGE - SUCCESS_LIN_VEL_TOL, 1e-9), 0.0, 1.0))
        _yaw_f = SHAPE_YAW_MIX + (1.0 - SHAPE_YAW_MIX) * _yaw_ok
        _vel_f = SHAPE_VEL_MIX + (1.0 - SHAPE_VEL_MIX) * _vel_ok
        shape_bonus = SHAPE_CAP * _prox * _yaw_f * _vel_f
        reward += shape_bonus
        self.ep_shape_sum += shape_bonus
        self.ep_prox_sum += _prox

        # ===== 障碍物碰撞：扣30分（不再终止本轮） =====
        if not self.collision_penalty_applied and self._check_obstacle_collision():
            reward -= 30.0
            self.collision_penalty_applied = True

        # ===== 货叉上的托盘/货物碰撞（不含地面货物堆）：扣30分，终止 =====
        if self.has_cargo and not self.cargo_collision_penalty_applied and self._check_cargo_collision():
            reward -= 30.0
            self.cargo_collision_penalty_applied = True

        # ===== 地面货物堆碰撞（叉车/托盘/货物碰到托盘和货物）：扣30分，不终止 =====
        if self.has_goods and not self.goods_collision_penalty_applied and self._check_goods_collision():
            reward -= 20.0
            self.goods_collision_penalty_applied = True

        return reward

        
    def _is_forklift_part(self, body_id):
        while body_id > 0:
            if body_id == self.forklift_body_id:
                return True
            body_id = self.model.body_parentid[body_id]
        return False

    def _check_obstacle_collision(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            
            is_obs1 = name1 is not None and name1.startswith("obstacle_")
            is_obs2 = name2 is not None and name2.startswith("obstacle_")
            
            if is_obs1 or is_obs2:
                other_body_id = self.model.geom_bodyid[c.geom2] if is_obs1 else self.model.geom_bodyid[c.geom1]
                if self._is_forklift_part(other_body_id):
                    return True
        return False

    def _check_cargo_collision(self):
        """检测托盘或货物是否碰到非货叉/非门架/非目标货物堆/非自身的物体。"""
        target_geoms = self._pallet_geom_ids | self._cargo_geom_ids
        # ===== 允许接触的：货叉、门架、托盘自身、货物自身、目标货物堆 =====
        # 目标货物堆碰撞由 _check_goods_collision 单独处理，避免双重扣分
        allowed_geoms = self._fork_mast_geom_ids | target_geoms | self._goods_geom_ids

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in target_geoms or g2 in target_geoms:
                other = g2 if g1 in target_geoms else g1
                if other not in allowed_geoms:
                    return True
        return False

    def _check_goods_collision(self):
        """检测叉车 / 托盘 / 货物 是否与目标货物堆发生碰撞。"""
        if not self.has_goods or self.goods_body_id < 0 or len(self._goods_geom_ids) == 0:
            return False

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self._goods_geom_ids or g2 in self._goods_geom_ids:
                other_geom = g2 if g1 in self._goods_geom_ids else g1
                other_body = self.model.geom_bodyid[other_geom]
                # 叉车任意子部件、托盘、货物 碰到目标货物 = 触发
                if (other_body == self.pallet_body_id or
                    other_body == self.cargo_body_id or
                    self._is_forklift_part(other_body)):
                    return True
        return False

    def _check_terminated(self, obs):
        e_x, e_y, e_theta = self._true_errors(); lin_vel = float(obs["state"][3])
        dist = np.sqrt(e_x**2 + e_y**2)
        if (dist < SUCCESS_POS_TOL and abs(e_theta) < SUCCESS_YAW_TOL
                and abs(lin_vel) < SUCCESS_LIN_VEL_TOL):
            self.is_success = True
            return True

        if self.cargo_collision_penalty_applied:
            return True
        
        if self._check_unsafe():
            self.unsafe_counter += 1
            if self.unsafe_counter >= self.max_unsafe_steps:
                return True
        else:
            self.unsafe_counter = 0
        return False

    def _check_unsafe(self):
        body = self.data.body(self.forklift_body_id)
        rot = body.xmat.reshape(3,3)
        if np.dot(rot[:,2], np.array([0,0,1])) < np.cos(np.deg2rad(80)):
            return True
        for gid in self._tire_geom_ids:
            if self.data.geom_xpos[gid][2] > 0.22:
                return True
        return False

    def _apply_control(self, v_cmd, steer_cmd, dt_scale=1.0):
        # dt_scale: [v8] 末段精细控制时 repeat 变小，按比例补偿积分/微分/转向速率
        err = steer_cmd - self.current_steer
        max_chg = self.MAX_STEER_RATE * self.model.opt.timestep * dt_scale
        self.current_steer += np.clip(err, -max_chg, max_chg)

        v_des = v_cmd * self.max_lin_speed
        delta_rear = -self.current_steer
        tan_d = np.tan(delta_rear)
        factor = (self.W_TRACK/2) * tan_d / self.L_AXLE
        left_lin  = v_des * (1.0 - factor)
        right_lin = v_des * (1.0 + factor)

        target_wl = left_lin / self.WHEEL_RADIUS
        target_wr = right_lin / self.WHEEL_RADIUS

        wl = self.data.joint("spin_front_left").qvel[0]
        wr = self.data.joint("spin_front_right").qvel[0]

        el = target_wl - wl; er = target_wr - wr
        if abs(v_des) < 0.001:
            self.integral_left = 0.0; self.integral_right = 0.0

        dt = self.model.opt.timestep * dt_scale
        self.integral_left  += el * dt; self.integral_right += er * dt
        self.integral_left  = np.clip(self.integral_left, -self.INTEGRAL_MAX, self.INTEGRAL_MAX)
        self.integral_right = np.clip(self.integral_right, -self.INTEGRAL_MAX, self.INTEGRAL_MAX)

        dl = (el - self.last_error_left) / dt if dt > 0 else 0.0
        dr = (er - self.last_error_right) / dt if dt > 0 else 0.0

        ff_left  = self.FEEDFORWARD_GAIN * target_wl
        ff_right = self.FEEDFORWARD_GAIN * target_wr

        # ===== 分别计算两轮 PID 力矩 =====
        tl_raw = ff_left  + self.KP_SPEED*el + self.KI_SPEED*self.integral_left  + self.KD_SPEED*dl
        tr_raw = ff_right + self.KP_SPEED*er + self.KI_SPEED*self.integral_right + self.KD_SPEED*dr

        # ===== 两轮力矩取平均，保证一致；转速由物理/阿克曼几何自然不同 =====
        avg_torque = (tl_raw + tr_raw) / 2.0
        tl = np.clip(avg_torque, -self.MAX_TORQUE, self.MAX_TORQUE)
        tr = np.clip(avg_torque, -self.MAX_TORQUE, self.MAX_TORQUE)

        self.last_error_left = el; self.last_error_right = er
        self.data.ctrl[0] = tl; self.data.ctrl[1] = tr

        delta = delta_rear
        if abs(delta) < 1e-6: ls, rs = 0.0, 0.0
        else:
            if delta > 0:
                R = self.L_AXLE / np.tan(delta)
                rs = np.arctan(self.L_AXLE / (R - self.W_TRACK/2))
                ls = np.arctan(self.L_AXLE / (R + self.W_TRACK/2))
            else:
                dp = -delta
                R = self.L_AXLE / np.tan(dp)
                rs = -np.arctan(self.L_AXLE / (R + self.W_TRACK/2))
                ls = -np.arctan(self.L_AXLE / (R - self.W_TRACK/2))
        
        self.data.ctrl[2] = ls
        self.data.ctrl[3] = rs

        if self.has_cargo:
            self.data.ctrl[4] = 0.4
        else:
            self.data.ctrl[4] = -0.23

        lin_vel = (wl + wr) / 2.0 * self.WHEEL_RADIUS
        if abs(v_des) < 0.01:
            brake_torque = -np.sign(lin_vel) * min(800, abs(lin_vel) * 2000)
        else:
            brake_torque = 0.0
            
        self.data.ctrl[5] = brake_torque
        self.data.ctrl[6] = brake_torque

    def _get_forklift_pose(self):
        body = self.data.body(self.forklift_body_id)
        pos = body.xpos.copy(); rot = body.xmat.reshape(3,3)
        yaw = np.arctan2(rot[1,0], rot[0,0])
        return pos, rot, yaw

    def render(self):
        return self._render_camera()

    def close(self):
        renderer = getattr(self, "_renderer", None)
        if renderer is not None:
            renderer.close()
            self._renderer = None


def make_env(xml_path, rank):
    def _init():
        return ForkliftNavEnv(xml_path, max_steps=500, action_repeat=50)
    return _init

def linear_schedule(initial_value: float = 2e-4, final_value: float = 5e-6):
    def schedule(progress_remaining: float):
        return final_value + (initial_value - final_value) * progress_remaining
    return schedule

# ===== PPO 超参（脚本为唯一真源）=====
# 注意：SB3 的 PPO.load() 会用“存档里的超参 + 存档里的策略类”覆盖构造函数
# （见 stable_baselines3/common/base_class.py 的 load()：model.__dict__.update(data)），
# 所以加载已有模型/断点后，必须把这些值再强制写回，否则改脚本不生效。
PPO_N_STEPS = 1024
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 4          # 原 10：单轮数据被反复训练太多，更新过猛
PPO_TARGET_KL = 0.15      # KL 超过 1.5×该值就提前结束本轮更新
PPO_GAMMA = 0.995
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2


def force_hyperparameters(model, min_std=None, verbose=True):
    """把本脚本的 PPO 超参强制写回已加载的模型（覆盖存档值）。"""
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.utils import FloatSchedule

    if min_std is None:
        min_std = VisionMinStdPolicy.MIN_STD_VALUE

    before = (model.n_steps, model.batch_size, model.n_epochs, model.target_kl,
              model.gamma, model.gae_lambda)

    model.n_steps = PPO_N_STEPS
    model.batch_size = PPO_BATCH_SIZE
    model.n_epochs = PPO_N_EPOCHS
    model.target_kl = PPO_TARGET_KL
    model.gamma = PPO_GAMMA
    model.gae_lambda = PPO_GAE_LAMBDA
    model.clip_range = FloatSchedule(PPO_CLIP_RANGE)

    # n_steps / gamma / gae_lambda 在 RolloutBuffer 构造时就固定了，必须重建
    # 注意：Dict 观测空间要用 DictRolloutBuffer（存档里已带 rollout_buffer_class）
    buffer_cls = getattr(model, "rollout_buffer_class", None) or RolloutBuffer
    old_buffer = model.rollout_buffer
    model.rollout_buffer = None
    del old_buffer
    model.rollout_buffer = buffer_cls(
        model.n_steps, model.observation_space, model.action_space,
        device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda,
        n_envs=getattr(model, "n_envs", 1),
        **(getattr(model, "rollout_buffer_kwargs", None) or {}))

    # 策略的 std 下限：存档里的策略类可能是旧版本，这里改实例属性
    if hasattr(model.policy, "min_log_std"):
        model.policy.min_log_std = (float(np.log(min_std)) if min_std > 0
                                    else -np.inf)

    if verbose:
        print(f"[INFO] 强制写回脚本超参: n_steps {before[0]}→{model.n_steps}, "
              f"batch_size {before[1]}→{model.batch_size}, "
              f"n_epochs {before[2]}→{model.n_epochs}, "
              f"target_kl {before[3]}→{model.target_kl}, "
              f"gamma {before[4]}→{model.gamma}, "
              f"gae_lambda {before[5]}→{model.gae_lambda}, "
              f"clip_range {PPO_CLIP_RANGE}, min_std {min_std}", flush=True)
    return model

# ================= 课程学习及成功率统计回调 =================
class CurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_printed_angle = -1.0

        # 与 SB3 rollout/success_rate 相同的近端回合窗口，记录分类后的完成回合
        self._episode_records = deque(maxlen=100)

    def _on_step(self) -> bool:
        training_progress = self.num_timesteps / self.total_timesteps
        self.training_env.env_method("update_progress", 1.0 - training_progress)
        
        if training_progress < 0.0:
            current_max_angle = MAX_ANG * (training_progress / 0.0)
        else:
            current_max_angle = MAX_ANG
            
        if abs(current_max_angle - self.last_printed_angle) >= 1.0 or self.n_calls % 100000 == 0:
            print(f"[Curriculum] Progress: {training_progress*100:.1f}% | Current Max Angle Limit: {current_max_angle:.2f} degrees")
            self.last_printed_angle = current_max_angle
            
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        # 处理每个环境的结束信息；denominator 为对应分类内完成的回合数
        for i, done in enumerate(dones):
            if done:
                if i < len(infos) and isinstance(infos[i], dict):
                    info = infos[i]
                    is_success = bool(info.get("is_success", False))
                    is_forward = bool(info.get("is_forward", False))
                    mode = int(info.get("mode", 0))
                    if is_forward and mode == 1:
                        category = "forward_mode1"
                    elif is_forward:
                        category = "forward_mode0"
                    else:
                        category = "backward"
                    self._episode_records.append({
                        "category": category,
                        "success": is_success,
                        "end_reason": str(info.get("end_reason", "unknown")),
                        "obstacle_collision": bool(info.get("obstacle_collision", False)),
                        "goods_collision": bool(info.get("goods_collision", False)),
                        "cargo_collision": bool(info.get("cargo_collision", False)),
                        "final_dist": float(info.get("final_dist", np.nan)),
                        "final_yaw_deg": float(info.get("final_yaw_deg", np.nan)),
                        "final_speed": float(info.get("final_speed", np.nan)),
                        "min_dist": float(info.get("min_dist", np.nan)),
                        "best_gate_ratio": float(info.get("best_gate_ratio", np.nan)),
                        "shape_bonus_sum": float(info.get("shape_bonus_sum", float("nan"))),
                        "prox_factor_sum": float(info.get("prox_factor_sum", float("nan"))),
                        "terminal_close_bonus": float(info.get("terminal_close_bonus", float("nan"))),
                        "terminal_far_penalty": float(info.get("terminal_far_penalty", float("nan"))),
                        "steps": int(info.get("episode_steps", 0)),
                    })

        return True

    def _on_rollout_end(self) -> None:
        """在每个 SB3 rollout 结束后，把分类成功率、失败模式、末端误差写入 rollout/。"""
        if not self._episode_records:
            return

        recs = list(self._episode_records)
        n = len(recs)

        def _rec(tag, value):
            """跳过 nan，避免把无效点写进 TensorBoard。"""
            if value is None:
                return
            value = float(value)
            if np.isnan(value):
                return
            self.logger.record(tag, value)

        def _rate(pred):
            return sum(1 for r in recs if pred(r)) / n

        def _stat(key):
            vals = sorted(r[key] for r in recs if not np.isnan(r[key]))
            if not vals:
                return float("nan"), float("nan")
            return float(np.mean(vals)), float(np.median(vals))

        # ===== 各分类成功率（保持原样）=====
        for tag, category in (
            ("rollout/forward_mode0_success_rate", "forward_mode0"),
            ("rollout/forward_mode1_success_rate", "forward_mode1"),
            ("rollout/backward_success_rate", "backward"),
        ):
            category_results = [r["success"] for r in recs if r["category"] == category]
            if category_results:
                _rec(tag, float(sum(category_results)) / len(category_results))

        # ===== 失败模式：碰撞率 / 超时率 / 终止原因 =====
        _rec("rollout/collision_rate", _rate(
            lambda r: r["obstacle_collision"] or r["goods_collision"] or r["cargo_collision"]))
        _rec("rollout/obstacle_collision_rate", _rate(lambda r: r["obstacle_collision"]))
        _rec("rollout/goods_collision_rate", _rate(lambda r: r["goods_collision"]))
        _rec("rollout/cargo_collision_rate", _rate(lambda r: r["cargo_collision"]))
        _rec("rollout/timeout_rate", _rate(lambda r: r["end_reason"] == "timeout"))
        _rec("rollout/unsafe_rate", _rate(lambda r: r["end_reason"] == "unsafe"))
        # 近失：失败但最接近时离门限还差 NEAR_MISS_RATIO 倍以内
        # （阈值与门限挂钩，改门限时自动跟随，不会再出现"0.2m 在 1.2cm 门限下毫无区分度"的问题）
        _rec("rollout/near_miss_rate", _rate(
            lambda r: (not r["success"]) and r["best_gate_ratio"] < NEAR_MISS_RATIO))

        # ===== 末端误差（回合结束那一刻；单位 m / 度 / m·s⁻¹）=====
        for tag, key in (("final_dist", "final_dist"),
                         ("final_yaw_deg", "final_yaw_deg"),
                         ("final_speed", "final_speed"),
                         ("min_dist", "min_dist"),
                         ("best_gate_ratio", "best_gate_ratio")):
            mean_v, median_v = _stat(key)
            _rec(f"rollout/{tag}_mean", mean_v)
            _rec(f"rollout/{tag}_median", median_v)

        # ===== 回合长度（完成回合）=====
        mean_steps, median_steps = _stat("steps")
        _rec("rollout/ep_steps_mean", mean_steps)
        _rec("rollout/ep_steps_median", median_steps)

        # ===== 停稳奖励（每回合拿到多少；很大但成功率没涨 = 在赖着不走）=====
        # ===== [v8] 贴近塑形：每回合平均拿到多少；以及"平均贴近度"=====
        shape_mean, _ = _stat("shape_bonus_sum")
        _rec("rollout/shape_bonus_mean", shape_mean)
        _prox_sum, _ = _stat("prox_factor_sum")
        _steps = [r["steps"] for r in recs if r["steps"] > 0]
        if _prox_sum is not None and not np.isnan(_prox_sum) and _steps:
            _rec("rollout/prox_factor_mean",
                 float(_prox_sum) / float(np.mean(_steps)))

        # ===== 终局精度项（v3）：只统计超时回合平均拿到多少（趋于 0 = 根本没靠近门口）=====
        _fails = [r["terminal_close_bonus"] for r in recs
                  if (not r["success"]) and not np.isnan(r["terminal_close_bonus"])]
        _rec("rollout/terminal_close_mean",
             float(np.mean(_fails)) if _fails else float("nan"))
        _farp = [r["terminal_far_penalty"] for r in recs
                if (not r["success"]) and not np.isnan(r["terminal_far_penalty"])]
        _rec("rollout/terminal_far_penalty_mean",
             float(np.mean(_farp)) if _farp else float("nan"))



def _ckpt_dir():
    """断点目录固定在本脚本所在目录的 train_model/ 下（与运行时 cwd 无关）。"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT_DIRNAME)


def current_learning_rate(model, default=INITIAL_LR):
    """取出模型当前（断点处）的学习率。"""
    lr = getattr(model, "learning_rate", None)
    if callable(lr):
        total = int(getattr(model, "_total_timesteps", 0) or 0)
        progress = 1.0 if total <= 0 else max(
            0.0, min(1.0, 1.0 - model.num_timesteps / total))
        try:
            return float(lr(progress))
        except Exception:
            return float(default)
    try:
        return float(lr)
    except (TypeError, ValueError):
        return float(default)


def resume_linear_schedule(start_lr, start_steps, total_steps, final_value=FINAL_LR):
    """断点续训专用：从断点处的学习率按步数线性衰减到 final_value。"""
    span = max(1, int(total_steps) - int(start_steps))

    def schedule(progress_remaining: float):
        done = (1.0 - float(progress_remaining)) * total_steps
        frac = (done - start_steps) / span
        frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)
        return final_value + (start_lr - final_value) * (1.0 - frac)

    return schedule


class CheckpointCallback(BaseCallback):
    """每 interval 步把模型存进 train_model/，文件名带步数，后续可直接接着训练。"""

    def __init__(self, prefix, save_dir=None, interval=CKPT_INTERVAL,
                 run_tag=None, verbose=1):
        super().__init__(verbose)
        self.prefix = prefix
        self.save_dir = save_dir or _ckpt_dir()
        self.interval = int(interval)
        self.run_tag = run_tag      # 本次训练的时间标签，避免不同批次互相覆盖
        self.next_save = self.interval

    def _on_training_start(self) -> None:
        # 续训时 num_timesteps 不为 0，对齐到下一个整数倍，避免一上来就保存
        self.next_save = ((self.num_timesteps // self.interval) + 1) * self.interval

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_save:
            self.save_checkpoint()
            self.next_save = ((self.num_timesteps // self.interval) + 1) * self.interval
        return True

    def save_checkpoint(self, tag=None):
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            steps = int(self.num_timesteps)
            suffix = f"_{self.run_tag}" if self.run_tag else ""
            stem = (f"{self.prefix}_ckpt_{steps}steps" if tag is None
                    else f"{self.prefix}_ckpt_{tag}") + suffix
            path = os.path.join(self.save_dir, stem + ".zip")
            self.model.save(path)
            self.model.save(os.path.join(self.save_dir, f"{self.prefix}_latest.zip"))
            with open(os.path.join(self.save_dir, f"{self.prefix}_latest.json"),
                      "w", encoding="utf-8") as f:
                json.dump({
                    "checkpoint": os.path.basename(path),
                    "num_timesteps": steps,
                    "total_timesteps": int(getattr(self.model, "_total_timesteps", 0)),
                    "learning_rate": current_learning_rate(self.model),
                    "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }, f, ensure_ascii=False, indent=1)
            print(f"[CKPT] 已保存断点: {path}", flush=True)
            return path
        except Exception as exc:   # 保存失败不应中断训练
            print(f"[CKPT][WARN] 保存断点失败: {exc}", flush=True)
            return None


def latest_checkpoint_any(save_dir):
    """[v4] train_model/ 里所有 train_mohu_* 断点中步数最大的一个（不递归子目录）。"""
    best, best_steps = None, -1
    for p in glob.glob(os.path.join(save_dir, "train_mohu_*_ckpt_*steps*.zip")):
        m = re.search(r"_ckpt_(\d+)steps(?:_[^/\\]+)?\.zip$", os.path.basename(p))
        if m and int(m.group(1)) > best_steps:
            best, best_steps = p, int(m.group(1))
    return best


def latest_checkpoint(save_dir, prefix):
    """返回 train_model/ 里步数最大的断点路径，没有则返回 None。"""
    best, best_steps = None, -1
    for p in glob.glob(os.path.join(save_dir, f"{prefix}_ckpt_*.zip")):
        m = re.search(r"_ckpt_(\d+)steps(?:_[^/\\]+)?\.zip$", os.path.basename(p))
        if m and int(m.group(1)) > best_steps:
            best, best_steps = p, int(m.group(1))
    return best

# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视觉导航训练：每 100 万步断点，可接着训练")
    parser.add_argument("--xml", default="model/forklift_train.xml")
    parser.add_argument("--model", default="./robot/ppo_3.zip")


    parser.add_argument("--n-envs", type=int, default=23)
    parser.add_argument("--total-steps", type=int, default=None,
                        help="累计练到多少步（绝对目标）；不填则沿用上次训练记录的目标"
                             f"（没有则用默认 {DEFAULT_TOTAL_STEPS}）")

    
    parser.add_argument("--resume", default="auto",
                        help="auto=优先续训 train_model 最新断点，没有才加载 --model(默认)；"
                             "ckpt=强制用 train_model 最新断点；"
                             "none=从零开始；也可直接写 .zip 路径")
    parser.add_argument("--ckpt-interval", type=int, default=CKPT_INTERVAL)
    parser.add_argument("--lr", type=float, default=None,
                        help="覆盖本阶段的起始学习率，例如 --lr 1e-4；不填则沿用脚本规则："
                             "断点处是多少就从多少继续衰减；上一轮已跑满目标步数时回到 INITIAL_LR")
    args = parser.parse_args()

    xml_path = args.xml
    model_path = args.model
    n_envs = args.n_envs
    total_new_timesteps = args.total_steps
    prefix = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]
    ckpt_dir = _ckpt_dir()
    run_tag = datetime.datetime.now().strftime("%m%d-%H%M")

    resume_path, reset_num_timesteps = None, True
    if args.resume == "none":
        print("[INFO] 从零开始训练新模型")
    elif args.resume == "ckpt":
        resume_path = latest_checkpoint(ckpt_dir, prefix)
        if resume_path is None:
            print(f"[INFO] {ckpt_dir} 中没有断点，从零开始训练新模型")
        else:
            reset_num_timesteps = False
            print(f"[INFO] 从最新断点续训: {resume_path}")
    elif args.resume == "auto":
        # 优先续训 train_model/ 里最新的断点；没有断点才退回 --model
        resume_path = latest_checkpoint(ckpt_dir, prefix)
        if resume_path is not None:
            reset_num_timesteps = False
            print(f"[INFO] 发现本脚本断点，优先续训: {resume_path}")
        else:
            # [v4] 本脚本还没有断点：自动接 train_model 里最新的 train_mohu_* 断点，
            #      避免静默退回 robot/ppo_3.zip 从头再来一遍。
            _prev = latest_checkpoint_any(ckpt_dir)
            if _prev is not None:
                resume_path = _prev
                reset_num_timesteps = False
                print(f"[INFO] 本脚本无断点，自动接上一版最新断点续训: {_prev}")
                print("[INFO] 若不想这样，请用 --resume none 或 --model 指定起点")
    else:
        resume_path = args.resume
        reset_num_timesteps = False
        print(f"[INFO] 从指定模型续训: {resume_path}")

    env = SubprocVecEnv([make_env(xml_path, i) for i in range(n_envs)])
    env = VecMonitor(env)  

    temp_env = ForkliftNavEnv(xml_path, max_steps=500, action_repeat=50)
    check_env(temp_env, warn=True)
    temp_env.close()

    policy_kwargs = vision_policy_kwargs()

    if resume_path is None and args.resume == "auto" and os.path.exists(model_path):
        resume_path = model_path
        print(f"[INFO] 未找到断点，加载已有模型作为新阶段起点: {model_path}")

    if resume_path is not None:
        lr_start = INITIAL_LR if args.lr is None else float(args.lr)
        custom_objects = {
            "policy": VisionMinStdPolicy,
            "policy_kwargs": policy_kwargs,
            "learning_rate": linear_schedule(lr_start, FINAL_LR)
        }
        model = PPO.load(resume_path, env=env, custom_objects=custom_objects)
        print(f"[INFO] 已加载模型 (num_timesteps={model.num_timesteps})")
        force_hyperparameters(model)      # 存档里的超参/策略类会覆盖构造函数，这里按脚本写回
    else:
        print("[INFO] 未找到历史模型，初始化新模型训练")
        lr_start = INITIAL_LR if args.lr is None else float(args.lr)
        model = PPO(
            VisionMinStdPolicy,
            env,
            verbose=1,
            tensorboard_log="./tensorboard_logs_straight/",
            learning_rate=linear_schedule(lr_start, FINAL_LR),
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            target_kl=PPO_TARGET_KL,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            clip_range=PPO_CLIP_RANGE,
            policy_kwargs=policy_kwargs,
            device="cuda"
        )

    # --total-steps 一律表示“累计练到多少步”（绝对目标）：
    #   新阶段（从零/加载 robot 模型）：就是本次要跑的步数
    #   断点续训：本次只再跑 (目标步数 - 断点步数) 步
    if total_new_timesteps is None:
        prev_total = (0 if reset_num_timesteps
                      else int(getattr(model, "_total_timesteps", 0) or 0))
        if prev_total > model.num_timesteps:
            total_new_timesteps = prev_total
            print(f"[INFO] 未指定 --total-steps，沿用上次训练的目标: {prev_total} 步")
        else:
            total_new_timesteps = DEFAULT_TOTAL_STEPS
            print(f"[INFO] 未指定 --total-steps，使用默认目标: {DEFAULT_TOTAL_STEPS} 步")

    absolute_total_timesteps = int(total_new_timesteps)
    start_steps = 0 if reset_num_timesteps else int(model.num_timesteps)
    steps_to_run = absolute_total_timesteps - start_steps
    if steps_to_run <= 0:
        raise SystemExit(
            f"[ERROR] 当前已在 {start_steps} 步，--total-steps={absolute_total_timesteps} "
            f"不比它大，没东西可练。\n"
            f"        想再练 100 万步就写：--total-steps {start_steps + 1_000_000}")

    # 断点续训：学习率也接着断点走
    if not reset_num_timesteps:
        prev_total = int(getattr(model, "_total_timesteps", 0) or 0)
        if args.lr is not None:
            start_lr = float(args.lr)          # 命令行显式指定，优先
        elif prev_total and model.num_timesteps >= prev_total:
            start_lr = INITIAL_LR      # 上一轮已经跑满，视作新阶段重新挂衰减
        else:
            start_lr = current_learning_rate(model)
        resume_schedule = resume_linear_schedule(
            start_lr, model.num_timesteps, absolute_total_timesteps)
        model.learning_rate = resume_schedule
        # SB3 训练时实际调用的是 lr_schedule，必须同步替换才生效
        if hasattr(model, "lr_schedule"):
            model.lr_schedule = resume_schedule
        print(f"[INFO] 续训学习率: {start_lr:.3e} → {FINAL_LR:.1e}"
              f"（步数 {model.num_timesteps} → {absolute_total_timesteps}）"
              f"{'  [命令行 --lr 指定]' if args.lr is not None else ''}", flush=True)

    curriculum_callback = CurriculumCallback(absolute_total_timesteps)
    checkpoint_callback = CheckpointCallback(prefix, ckpt_dir, args.ckpt_interval,
                                             run_tag=run_tag)
    callback = CallbackList([curriculum_callback, checkpoint_callback])

    print(f"[INFO] 累计 {start_steps} 步 → 目标 {absolute_total_timesteps} 步，"
          f"本次实际再训 {steps_to_run} 步；"
          f"每 {args.ckpt_interval} 步保存到 {ckpt_dir}（标签 {run_tag}）", flush=True)

    try:
        model.learn(total_timesteps=steps_to_run, callback=callback,
                    reset_num_timesteps=reset_num_timesteps)
    except KeyboardInterrupt:
        print("[INFO] 训练被手动中断，保存当前断点…", flush=True)
        checkpoint_callback.save_checkpoint(tag="interrupt")
        env.close()
        raise

    model.save(model_path)
    print(f"[INFO] 模型已保存至: {model_path}")
    checkpoint_callback.save_checkpoint()
