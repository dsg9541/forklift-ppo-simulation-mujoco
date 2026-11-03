import numpy as np
import mujoco

class ForkHeightController:
    """独立的货叉高度控制器，基于物理高度闭环跟踪"""
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.forks_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "forks")
        self.lift_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lift_joint")
        self.lift_joint_range = [-0.23, 1.45]
        
        self.use_raw_target = False
        self.raw_target = 0.0
        self.target_height = 0.0

    def set_target_height(self, target_height):
        """设置货叉下表面的目标全局高度"""
        self.use_raw_target = False
        self.target_height = target_height

    def set_raw_target(self, raw_val):
        """直接下发 ctrl[4] 数值"""
        self.use_raw_target = True
        self.raw_target = raw_val

    def update(self):
        """每步仿真前调用，更新 ctrl[4]"""
        if self.use_raw_target:
            self.data.ctrl[4] = np.clip(self.raw_target, self.lift_joint_range[0], self.lift_joint_range[1])
            return

        current_height = self.data.body(self.forks_body_id).xpos[2]
        error = self.target_height - current_height
        
        if abs(error) > 0.2:
            speed = 0.25  # m/s
        else:
            speed = 0.175 # m/s
            
        dt = self.model.opt.timestep
        delta_z = speed * dt * np.sign(error)
        
        if abs(delta_z) > abs(error):
            delta_z = error
            
        new_ctrl = self.data.ctrl[4] + delta_z
        new_ctrl = np.clip(new_ctrl, self.lift_joint_range[0], self.lift_joint_range[1])
        self.data.ctrl[4] = new_ctrl
