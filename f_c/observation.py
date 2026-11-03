import numpy as np
import mujoco

RADARS_CONFIG = [
    {"name": "lidar_mast",  "alt_name": "sensor_mast",  "yaw": 0.0,        "half_fov": np.pi/4, "rays": 20},
    {"name": "lidar_left",  "alt_name": "sensor_left",  "yaw": np.pi/2,    "half_fov": np.pi/3, "rays": 25},
    {"name": "lidar_right", "alt_name": "sensor_right", "yaw": -np.pi/2,   "half_fov": np.pi/3, "rays": 25},
    {"name": "lidar_rear",  "alt_name": "sensor_rear",  "yaw": np.pi,      "half_fov": np.pi/3, "rays": 25}
]
LIDAR_MAX_DIST = 8.0

def get_lidar_scan(model, data, forklift_body_id, pallet_body_id, cargo_body_id):
    R = data.xmat[forklift_body_id].reshape(3, 3)
    geomid_out = np.zeros(1, dtype=np.int32)
    lidar_data = []
    
    for cfg in RADARS_CONFIG:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg["name"])
        if site_id < 0:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg["alt_name"])
        if site_id < 0:
            lidar_data.extend([1.0] * cfg["rays"])
            continue
            
        lidar_pos = data.site_xpos[site_id].copy()
        
        for i in range(cfg["rays"]):
            if cfg["rays"] > 1:
                angle = -cfg["half_fov"] + 2 * cfg["half_fov"] * (i / (cfg["rays"] - 1))
            else:
                angle = 0.0
            local_angle = cfg["yaw"] + angle
            
            local_dir = np.array([np.cos(local_angle), np.sin(local_angle), 0.0])
            ray_dir = R @ local_dir
            ray_dir /= np.linalg.norm(ray_dir)
            
            dist = mujoco.mj_ray(model, data, lidar_pos, ray_dir, 
                                 None, False, 
                                 forklift_body_id, geomid_out)
            
            if dist >= 0 and geomid_out[0] >= 0:
                hit_body = model.geom_bodyid[geomid_out[0]]
                if hit_body == pallet_body_id or hit_body == cargo_body_id:
                    dist = LIDAR_MAX_DIST
            
            if dist < 0 or dist > LIDAR_MAX_DIST:
                dist = LIDAR_MAX_DIST
                
            lidar_data.append(dist / LIDAR_MAX_DIST)
            
    return np.array(lidar_data, dtype=np.float32)

def get_obs(model, data, target_pos, target_yaw, has_cargo, forklift_body_id, controller, pallet_body_id, cargo_body_id, init_pos, init_yaw):
    lidar_data = get_lidar_scan(model, data, forklift_body_id, pallet_body_id, cargo_body_id)
    
    body = data.body(forklift_body_id)
    pos = body.xpos.copy()
    rot = body.xmat.reshape(3, 3)
    yaw = np.arctan2(rot[1, 0], rot[0, 0])

    # 坐标系相对化转换
    dx_world = pos[0] - init_pos[0]
    dy_world = pos[1] - init_pos[1]
    rel_x =  dx_world * np.cos(init_yaw) + dy_world * np.sin(init_yaw)
    rel_y = -dx_world * np.sin(init_yaw) + dy_world * np.cos(init_yaw)
    rel_yaw = yaw - init_yaw
    rel_yaw = np.arctan2(np.sin(rel_yaw), np.cos(rel_yaw))

    t_dx_world = target_pos[0] - init_pos[0]
    t_dy_world = target_pos[1] - init_pos[1]
    rel_t_x =  t_dx_world * np.cos(init_yaw) + t_dy_world * np.sin(init_yaw)
    rel_t_y = -t_dx_world * np.sin(init_yaw) + t_dy_world * np.cos(init_yaw)
    rel_t_yaw = target_yaw - init_yaw
    rel_t_yaw = np.arctan2(np.sin(rel_t_yaw), np.cos(rel_t_yaw))

    dx = rel_t_x - rel_x
    dy = rel_t_y - rel_y
    e_x =  dx * np.cos(rel_yaw) + dy * np.sin(rel_yaw)
    e_y = -dx * np.sin(rel_yaw) + dy * np.cos(rel_yaw)
    e_theta = rel_t_yaw - rel_yaw
    e_theta = np.arctan2(np.sin(e_theta), np.cos(e_theta))

    wl = data.joint("spin_front_left").qvel[0]
    wr = data.joint("spin_front_right").qvel[0]
    lin_vel = (wl + wr) / 2.0 * 0.2  # WHEEL_RADIUS
    steer = controller.current_steer
    
    target_info = np.array([rel_t_x, rel_t_y, rel_t_yaw, float(has_cargo)])
    
    return np.concatenate([lidar_data,
                           np.array([e_x, e_y, e_theta]),
                           np.array([lin_vel, steer]),
                           target_info]).astype(np.float32)


# ======================================================================
# 27 号：与两个视觉模型训练时完全一致的观测构造
#   观测 = Dict{ image: (3,128,128) uint8, lidar: (279,), state: (6,) }
#   lidar = 车体水平雷达 95 + 货叉上下 4 + 门架垂直平面 180
#   state = [e_x, e_y, e_theta, lin_vel(真实车体速度), steer, mode]
#   其中 e_* 一律相对"引导位姿"计算（模型看到的就是降质后的引导）
# ======================================================================
CAMERA_NAME = "mast_cam"
CAMERA_HEIGHT = 128
CAMERA_WIDTH = 128

BODY_RADARS_CONFIG = [
    {"name": "lidar_mast",  "alt_name": "sensor_mast",  "yaw": 0.0,      "half_fov": np.pi / 4, "rays": 20},
    {"name": "lidar_left",  "alt_name": "sensor_left",  "yaw": np.pi / 2,  "half_fov": np.pi / 3, "rays": 25},
    {"name": "lidar_right", "alt_name": "sensor_right", "yaw": -np.pi / 2, "half_fov": np.pi / 3, "rays": 25},
    {"name": "lidar_rear",  "alt_name": "sensor_rear",  "yaw": np.pi,    "half_fov": np.pi / 3, "rays": 25},
]
FORK_POINT_RADARS = [
    {"name": "lidar_fork_left", "dir": [0, 0, 1]},
    {"name": "lidar_fork_right", "dir": [0, 0, 1]},
    {"name": "lidar_fork_bottom_left", "dir": [0, 0, -1]},
    {"name": "lidar_fork_bottom_right", "dir": [0, 0, -1]},
]
MAST_PLANE_RADARS = ["lidar_fork_top_left", "lidar_fork_top_right",
                     "lidar_fork_mid_left", "lidar_fork_mid_right"]
MAST_PLANE_LAYERS = 3
MAST_PLANE_RAYS = 15
MAST_PLANE_LAYER_ANGLE = np.deg2rad(10.0)


def _site_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def _dist_or_max(model, data, pnt, vec, exclude_body, max_dist):
    geomid_out = np.zeros(1, dtype=np.int32)
    d = mujoco.mj_ray(model, data, pnt, vec, None, False, exclude_body,
                      geomid_out)
    if d < 0 or d > max_dist:
        d = max_dist
    return d / max_dist


def build_lidar_scan(model, data, forklift_body_id, mast_body_id, forks_body_id,
                     max_dist=8.0):
    """279 维雷达（与训练脚本 _get_lidar_scan 逐行一致）。"""
    R_forklift = data.xmat[forklift_body_id].reshape(3, 3)
    R_mast = (data.xmat[mast_body_id].reshape(3, 3)
              if mast_body_id >= 0 else np.eye(3))
    R_forks = (data.xmat[forks_body_id].reshape(3, 3)
               if forks_body_id >= 0 else np.eye(3))
    lidar = []

    # 1) 车体水平雷达 95
    for cfg in BODY_RADARS_CONFIG:
        sid = _site_id(model, cfg["name"])
        if sid < 0:
            sid = _site_id(model, cfg["alt_name"])
        if sid < 0:
            lidar.extend([1.0] * cfg["rays"])
            continue
        pnt = data.site_xpos[sid].copy()
        for i in range(cfg["rays"]):
            angle = (-cfg["half_fov"] + 2 * cfg["half_fov"]
                     * (i / (cfg["rays"] - 1))) if cfg["rays"] > 1 else 0.0
            local_angle = cfg["yaw"] + angle
            local_dir = np.array([np.cos(local_angle), np.sin(local_angle), 0.0])
            vec = R_forklift @ local_dir
            vec /= np.linalg.norm(vec)
            lidar.append(_dist_or_max(model, data, pnt, vec,
                                      forklift_body_id, max_dist))

    # 2) 货叉上下单束 4
    for cfg in FORK_POINT_RADARS:
        sid = _site_id(model, cfg["name"])
        if sid < 0:
            lidar.append(1.0)
            continue
        pnt = data.site_xpos[sid].copy()
        vec = R_forks @ np.array(cfg["dir"], dtype=float)
        vec /= np.linalg.norm(vec)
        lidar.append(_dist_or_max(model, data, pnt, vec,
                                  forklift_body_id, max_dist))

    # 3) 门架垂直平面 180
    for name in MAST_PLANE_RADARS:
        sid = _site_id(model, name)
        if sid < 0:
            lidar.extend([1.0] * (MAST_PLANE_LAYERS * MAST_PLANE_RAYS))
            continue
        pnt = data.site_xpos[sid].copy()
        for layer in range(MAST_PLANE_LAYERS):
            layer_yaw = ((layer - (MAST_PLANE_LAYERS - 1) / 2.0)
                         * MAST_PLANE_LAYER_ANGLE)
            for i in range(MAST_PLANE_RAYS):
                theta = (i / (MAST_PLANE_RAYS - 1)) * (np.pi / 2.0)
                local_dir = np.array([np.cos(theta) * np.cos(layer_yaw),
                                      np.cos(theta) * np.sin(layer_yaw),
                                      -np.sin(theta)])
                vec = R_mast @ local_dir
                vec /= np.linalg.norm(vec)
                lidar.append(_dist_or_max(model, data, pnt, vec,
                                          forklift_body_id, max_dist))
    return np.asarray(lidar, dtype=np.float32)


def body_speed(model, data, forklift_body_id, root_dof_adr):
    """真实车体前进速度（沿车身 x 轴投影，与训练一致）。"""
    v_world = data.qvel[root_dof_adr:root_dof_adr + 3]
    x_axis = data.xmat[forklift_body_id].reshape(3, 3)[:, 0]
    return float(np.dot(v_world, x_axis))


def render_camera(renderer, data, camera_name=CAMERA_NAME):
    """门架相机 RGB -> CHW uint8。"""
    # 策略相机的裁剪面必须与训练一致（训练里 <visual> 是 znear 0.0015 / zfar 0.5，
    # 即远裁剪面 ≈ 0.5×extent）。而 vis.map 是全局的、窗口相机也用它，
    # 所以这里在"策略取图"时临时切到训练值，取完再切回（默认值给窗口用）。
    m = renderer.model
    old = float(m.vis.map.zfar)
    model_vis_zfar = 0.5
    try:
        m.vis.map.zfar = model_vis_zfar
        renderer.update_scene(data, camera=camera_name)
        rgb = renderer.render()
    finally:
        m.vis.map.zfar = old
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))


def build_obs(model, data, guidance_pose, forklift_body_id, controller,
              root_dof_adr, mast_body_id=-1, forks_body_id=-1,
              mode=0, renderer=None, camera_name=CAMERA_NAME,
              max_dist=8.0, image=None):
    """构造与训练完全一致的 Dict 观测。

    guidance_pose: (x, y, yaw) —— 降质后的引导位姿（不是真值）
    mode: 0 = 无货堆/地面区（精确分支），1 = 前方有货堆需视觉对准
    renderer: mujoco.Renderer 实例；为 None 时 image 返回全零（跳过视觉模型时用）
    image: 直接给定 CHW uint8 图像（图像处理后的门架相机图）；给了就不再渲染，
           用于 ppo_3 需要"只留目标货堆"的图像、ppo_2 用原图的情况。
    """
    body = data.body(forklift_body_id)
    pos = body.xpos.copy()
    rot = body.xmat.reshape(3, 3)
    yaw = np.arctan2(rot[1, 0], rot[0, 0])

    dx = guidance_pose[0] - pos[0]
    dy = guidance_pose[1] - pos[1]
    e_x = dx * np.cos(yaw) + dy * np.sin(yaw)
    e_y = -dx * np.sin(yaw) + dy * np.cos(yaw)
    e_theta = guidance_pose[2] - yaw
    e_theta = np.arctan2(np.sin(e_theta), np.cos(e_theta))

    state = np.array([e_x, e_y, e_theta,
                      body_speed(model, data, forklift_body_id, root_dof_adr),
                      controller.current_steer, float(mode)], dtype=np.float32)

    if image is not None:
        pass
    elif renderer is not None:
        image = render_camera(renderer, data, camera_name)
    else:
        image = np.zeros((3, CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.uint8)

    return {
        "image": image,
        "lidar": build_lidar_scan(model, data, forklift_body_id,
                                  mast_body_id, forks_body_id, max_dist),
        "state": state,
    }
