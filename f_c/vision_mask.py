"""门架相机图像处理（对齐 train_mohu_12.CameraMasker 的 "uniform" 模式）。

训练环境里只有一垛货，所以 v12 直接"保留所有 pallet_/cargo_ 像素、其余涂同色"。
27 号仓库里有多垛货，必须**只保留本轮目标那一垛**、其它货堆/墙/地面/叉车自身
全部涂成同一个颜色，画面分布才和训练一致（用户实测：这样处理后模型在
0.02m / 0.03rad / 0.1m/s 门限下成功率约 0.98）。

做法与训练一致：用 MuJoCo 的**语义分割**渲染（每个像素给 geom id）判断"这个
像素属于谁"，纯几何、与光照无关；不用颜色阈值猜。
"""
import numpy as np
import mujoco

UNIFORM_COLOR = (200, 200, 200)      # 与 train_mohu_12.VISION_UNIFORM_COLOR 一致


class StackMasker:
    """只保留目标货堆（按 body 名前缀，如 "stack_P1_"）的像素。"""

    def __init__(self, model, stack_prefix=None, uniform_color=UNIFORM_COLOR,
                 enabled=True):
        self.model = model
        self.enabled = bool(enabled)
        self.uniform_color = np.array(uniform_color, dtype=np.uint8).reshape(1, 1, 3)
        # geom_id -> 是否保留；索引 +1 是因为分割图里背景是 -1
        self.lut = np.zeros(model.ngeom + 2, dtype=np.uint8)
        self.last_stats = {}
        self.set_prefix(stack_prefix)

    def set_prefix(self, stack_prefix):
        """切换目标货堆（每轮开始调一次）。None = 全部涂掉（该轮不用视觉）。"""
        self.prefix = stack_prefix
        self.lut[:] = 0
        self.keep_geoms = []
        if not stack_prefix:
            return
        keep_bodies = set()
        for bid in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if name.startswith(stack_prefix):
                keep_bodies.add(bid)
        for gid in range(self.model.ngeom):
            if int(self.model.geom_bodyid[gid]) in keep_bodies:
                self.lut[gid + 1] = 1
                self.keep_geoms.append(gid)

    def apply(self, rgb_hwc, seg_geom):
        """rgb_hwc: (H,W,3) uint8；seg_geom: (H,W) int32（分割图第 0 通道）。
        返回新的 (H,W,3) uint8。"""
        if not self.enabled or not self.keep_geoms:
            return rgb_hwc
        keep = self.lut[seg_geom + 1] > 0
        out = np.empty_like(rgb_hwc)
        out[...] = self.uniform_color
        if keep.any():
            out[keep] = rgb_hwc[keep]
        self.last_stats = {"keep": float(keep.mean())}
        return out


class MaskedCamera:
    """门架相机的 RGB + 语义分割双渲染，输出"处理过"的 CHW uint8 图像。"""

    def __init__(self, model, height=128, width=128, camera="mast_cam",
                 uniform_color=UNIFORM_COLOR):
        self.camera = camera
        self.height = int(height)
        self.width = int(width)
        self.rgb_renderer = mujoco.Renderer(model, height=height, width=width)
        self.seg_renderer = mujoco.Renderer(model, height=height, width=width)
        self.seg_renderer.enable_segmentation_rendering()
        self.masker = StackMasker(model, uniform_color=uniform_color)

    def set_target(self, stack_prefix):
        self.masker.set_prefix(stack_prefix)

    def render(self, data, masked=True):
        """返回 CHW uint8 图像。masked=False 时只渲染原图（ppo_2 用）。"""
        self.rgb_renderer.update_scene(data, camera=self.camera)
        rgb = self.rgb_renderer.render()
        if masked and self.masker.keep_geoms:
            self.seg_renderer.update_scene(data, camera=self.camera)
            seg = self.seg_renderer.render()[:, :, 0]
            rgb = self.masker.apply(rgb, seg)
        return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))
