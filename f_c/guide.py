import numpy as np


def body_frame_goal(robot_pos, robot_yaw, goal_pos, goal_yaw):
    dx = goal_pos[0] - robot_pos[0]
    dy = goal_pos[1] - robot_pos[1]
    e_x = dx*np.cos(robot_yaw) + dy*np.sin(robot_yaw)
    e_y = -dx*np.sin(robot_yaw) + dy*np.cos(robot_yaw)
    e_theta = goal_yaw - robot_yaw
    e_theta = np.arctan2(np.sin(e_theta), np.cos(e_theta))
    return float(e_x), float(e_y), float(e_theta)


class GoalProvider:
    def __init__(self, target_pos, target_yaw):
        self.target_pos = np.asarray(target_pos, dtype=float)
        self.target_yaw = float(target_yaw)

    def update(self, robot_pos, robot_yaw):
        e = body_frame_goal(robot_pos, robot_yaw, self.target_pos, self.target_yaw)
        slot = np.array([self.target_pos[0], self.target_pos[1],
                         self.target_yaw, 0.0], dtype=np.float32)
        return e, slot
