"""引导降质模型（27 号）。

赛方规定：提供给算法的目标位姿"位置精度不得优于 20cm、姿态角精度不得优于 10°"。
训练脚本（train_input_104_* / train_mohu_*）里用 `_apply_guidance_noise()` 注入
同样量级的误差，这里在部署侧复刻同一分布，保证训练/部署一致。

约定（与训练一致）：
  - 位置误差：固定幅值 pos_err（m），方向均匀随机；
  - 姿态误差：固定幅值 yaw_err_deg（度），符号随机；
  - 每轮作业重置时采样一次，可设种子复现。
"""
import numpy as np

GUIDANCE_POS_ERR_M = 0.2
GUIDANCE_YAW_ERR_DEG = 10.0


class GuidanceModel:
    """把真值位姿降质成"引导位姿"。"""

    def __init__(self, pos_err=GUIDANCE_POS_ERR_M,
                 yaw_err_deg=GUIDANCE_YAW_ERR_DEG, seed=None):
        self.pos_err = float(pos_err)
        self.yaw_err = float(np.deg2rad(yaw_err_deg))
        self.rng = np.random.default_rng(seed)
        self._last = (0.0, 0.0)

    def sample(self, true_pose):
        """true_pose=(x, y, yaw) -> 带噪引导位姿 np.ndarray(3,)。

        位置误差幅值固定为 pos_err、方向随机；姿态误差幅值固定、符号随机，
        与训练脚本 `uniform(lo, hi)` 且 lo==hi 的行为一致。
        """
        true_pose = np.asarray(true_pose, dtype=float)
        ang = self.rng.uniform(0.0, 2.0 * np.pi)
        # 【用户 2026-09-20】幅值在 0~pos_err 内均匀随机（原来固定成最坏的 pos_err）
        _mag = self.rng.uniform(0.0, self.pos_err)
        offset = _mag * np.array([np.cos(ang), np.sin(ang)])
        yaw_mag = self.rng.uniform(-self.yaw_err, self.yaw_err)
        guide = true_pose.copy()
        guide[0:2] = true_pose[0:2] + offset
        guide[2] = true_pose[2] + yaw_mag
        guide[2] = np.arctan2(np.sin(guide[2]), np.cos(guide[2]))
        self._last = (float(np.hypot(offset[0], offset[1])), float(abs(yaw_mag)))
        return guide

    def last_error(self):
        """上一次采样的 (位置误差 m, 姿态误差 rad)，用于日志与可视化。"""
        return self._last

    def reset_seed(self, seed):
        self.rng = np.random.default_rng(seed)
