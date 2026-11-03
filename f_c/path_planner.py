import numpy as np
import heapq
import mujoco


class FineAlignController:
    """精细对齐控制器：前进纠偏为主，必要时后退创建空间"""

    def __init__(self, model, data, max_lin_speed=0.2):
        self.model = model
        self.data = data

        # 控制增益（提高纠偏力度）
        self.Kp_y = 4.0
        self.Kp_theta = 5.0

        # 精度阈值（适度放宽，保证可达成）
        self.TOL_Y = 0.015      # 2cm
        self.TOL_THETA = 0.02  # ~1.7度
        self.TOL_X = 0.05      # 5cm

        # 轮胎转角限制（15度，提升纠偏能力）
        self.MAX_STEER_RAD = np.deg2rad(15.0)

        # 状态机
        self.state = "FORWARD"
        self.step_count = 0
        self.max_steps = 5000
        self.backup_count = 0
        self.max_backups = 15
        self.backup_target_e_x = 0.0

    def reset(self):
        self.state = "FORWARD"
        self.step_count = 0
        self.backup_count = 0
        self.backup_target_e_x = 0.0

    def compute_action(self, e_x, e_y, e_theta, current_pos):
        self.step_count += 1

        if self.step_count > self.max_steps:
            print(f"  -> [对齐超时] 步数: {self.step_count}，强制结束对齐")
            return 0.0, 0.0, True

        y_ok = abs(e_y) < self.TOL_Y
        theta_ok = abs(e_theta) < self.TOL_THETA
        x_ok = abs(e_x) < self.TOL_X

        if y_ok and theta_ok and x_ok:
            return 0.0, 0.0, True

        if (abs(e_y) < self.TOL_Y * 2 and
                abs(e_theta) < self.TOL_THETA * 2 and
                abs(e_x) < self.TOL_X * 2):
            is_aligned = y_ok and theta_ok and x_ok
            return 0.0, 0.0, is_aligned

        if self.state == "BACKUP":
            if e_x >= self.backup_target_e_x:
                self.state = "FORWARD"
                print(f"  -> [对齐] 后退完成(e_x={e_x:.3f})，切换到前进纠偏")
            else:
                v_cmd = -0.1
                steer_angle = - (self.Kp_y * e_y + self.Kp_theta * e_theta)
                steer_cmd = np.clip(steer_angle, -self.MAX_STEER_RAD, self.MAX_STEER_RAD)
                return v_cmd, steer_cmd, False

        need_backup = False

        if e_x < -0.03:
            need_backup = True
        elif abs(e_x) < 0.15 and (abs(e_y) > self.TOL_Y or abs(e_theta) > self.TOL_THETA):
            if self.backup_count < self.max_backups:
                need_backup = True

        if need_backup:
            if self.backup_count >= self.max_backups:
                print(f"  -> [对齐] 达到最大后退次数({self.max_backups})，强制结束")
                return 0.0, 0.0, True

            self.state = "BACKUP"
            self.backup_target_e_x = max(0.4, 2.0 * abs(e_y) + 0.3)
            self.backup_count += 1
            print(f"  -> [对齐] 开始后退 #{self.backup_count} "
                  f"(e_x={e_x:.3f}, e_y={e_y:.3f}, e_θ={e_theta:.3f}), "
                  f"目标e_x={self.backup_target_e_x:.2f}")

            v_cmd = -0.1
            steer_angle = - (self.Kp_y * e_y + self.Kp_theta * e_theta)
            steer_cmd = np.clip(steer_angle, -self.MAX_STEER_RAD, self.MAX_STEER_RAD)
            return v_cmd, steer_cmd, False

        if e_x > 0.05:
            v_cmd = min(0.15, max(0.04, 0.2 * e_x))
        elif e_x > 0.0:
            v_cmd = 0.04
        else:
            v_cmd = 0.0

        steer_angle = (self.Kp_y * e_y + self.Kp_theta * e_theta)
        steer_cmd = np.clip(steer_angle, -self.MAX_STEER_RAD, self.MAX_STEER_RAD)

        return v_cmd, steer_cmd, False


class PathPlanner:
    # 膨胀距离改为 1.9m
    EROSION_DIST = 1.9

    def __init__(self, graph_data, model, data, forklift_body_id, root_qpos_adr):
        self.points = {p["id"]: p for p in graph_data["points"]}
        self.edges = graph_data["edges"]
        self.model = model
        self.data = data
        self.forklift_body_id = forklift_body_id
        self.root_qpos_adr = root_qpos_adr
        self.smoothed_path = []
        self.current_idx = 0
        self.just_reached = False
        self.last_reached_is_arc = False
        self.static_obstacles_aabb = []

        # 预计算所有有连边的有效节点
        self.connected_nodes = set()
        for e in self.edges:
            self.connected_nodes.add(e["from"])
            self.connected_nodes.add(e["to"])

    def _dijkstra(self, start_id, end_id):
        adj = {pid: [] for pid in self.points}
        for e in self.edges:
            adj[e["from"]].append(e["to"])
            adj[e["to"]].append(e["from"])
        dist = {pid: float('inf') for pid in self.points}
        prev = {pid: None for pid in self.points}
        dist[start_id] = 0
        pq = [(0, start_id)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == end_id: break
            if d > dist[u]: continue
            for v in adj[u]:
                w = np.hypot(self.points[u]["x"] - self.points[v]["x"], self.points[u]["y"] - self.points[v]["y"])
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        path = []
        cur = end_id
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        if path and path[0] == start_id:
            return path
        return []

    def _find_nearest_connected_node(self, node_id):
        """如果节点悬空(无连边)，寻找距离它最近的有效主干道节点"""
        if node_id in self.connected_nodes:
            return node_id
        if node_id not in self.points:
            return -1
        p = self.points[node_id]
        min_d = float('inf')
        best = -1
        for pid, other in self.points.items():
            if pid not in self.connected_nodes:
                continue
            d = (p['x'] - other['x'])**2 + (p['y'] - other['y'])**2
            if d < min_d:
                min_d = d
                best = pid
        return best

    def _get_point_yaw(self, p_node, next_node=None, prev_node=None):
        kind = p_node.get('kind', '')
        
        # 特殊节点：直接返回固定角度，无需计算
        if kind == 'shelf':
            return np.pi / 2.0
        elif kind == 'goods':
            return -np.pi / 2.0
        
        # 普通节点：12 号的纯路径点语义，中间点直接朝向下一段，
        # 让 PPO 在接近路径点时就开始提前转向，不必在点上“顿准”。
        if next_node:
            # 起点只有离开方向：朝向下一节点
            return np.arctan2(next_node['y'] - p_node['y'],
                              next_node['x'] - p_node['x'])
                              
        elif prev_node:
            # 普通终点(无特殊标记)：沿用进入方向
            return np.arctan2(p_node['y'] - prev_node['y'],
                              p_node['x'] - prev_node['x'])
                              
        else:
            return 0.0

    def _preprocess_static_obstacles(self):
        """预处理环境中的静态障碍物，提取并按 1.9m 侵蚀其轴对齐包围盒(AABB)"""
        self.static_obstacles_aabb = []
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not name:
                continue
            
            if name.startswith("shelf_") or name.startswith("obstacle_wall_") or name.startswith("obstacle_body_"):
                min_x, max_x = float('inf'), float('-inf')
                min_y, max_y = float('inf'), float('-inf')
                
                geom_start = self.model.body_geomadr[i]
                geom_end = geom_start + self.model.body_geomnum[i]
                
                for gid in range(geom_start, geom_end):
                    gpos = self.data.geom_xpos[gid]
                    gmat = self.data.geom_xmat[gid].reshape(3,3)
                    gsize = self.model.geom_size[gid]
                    
                    if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                        local_verts = np.array([
                            [gsize[0], gsize[1], gsize[2]], [-gsize[0], gsize[1], gsize[2]],
                            [gsize[0], -gsize[1], gsize[2]], [-gsize[0], -gsize[1], gsize[2]],
                            [gsize[0], gsize[1], -gsize[2]], [-gsize[0], gsize[1], -gsize[2]],
                            [gsize[0], -gsize[1], -gsize[2]], [-gsize[0], -gsize[1], -gsize[2]],
                        ])
                        world_verts = (gmat @ local_verts.T).T + gpos
                        min_x = min(min_x, world_verts[:,0].min())
                        max_x = max(max_x, world_verts[:,0].max())
                        min_y = min(min_y, world_verts[:,1].min())
                        max_y = max(max_y, world_verts[:,1].max())
                    else:
                        r = self.model.geom_rbound[gid]
                        min_x = min(min_x, gpos[0] - r)
                        max_x = max(max_x, gpos[0] + r)
                        min_y = min(min_y, gpos[1] - r)
                        max_y = max(max_y, gpos[1] + r)
                        
                if min_x != float('inf'):
                    self.static_obstacles_aabb.append({
                        'min_x': min_x - self.EROSION_DIST, 'max_x': max_x + self.EROSION_DIST,
                        'min_y': min_y - self.EROSION_DIST, 'max_y': max_y + self.EROSION_DIST
                    })

    def _check_point_in_aabb(self, point, aabb):
        return (aabb['min_x'] <= point[0] <= aabb['max_x'] and 
                aabb['min_y'] <= point[1] <= aabb['max_y'])

    def _smooth_path(self, raw_path_ids):
        """纯路径点：不做贝塞尔/圆弧插值，只按图节点生成折线路径。"""
        smoothed = []
        raw_nodes = [self.points[pid] for pid in raw_path_ids]

        for i, p_curr in enumerate(raw_nodes):
            p_next = raw_nodes[i+1] if i+1 < len(raw_nodes) else None
            p_prev = raw_nodes[i-1] if i > 0 else None

            # 目标航向：保留货架/货物朝向，普通点朝向下一段
            yaw_curr_target = self._get_point_yaw(p_curr, p_next, p_prev)

            # 精度阈值只用于路径规划器自身的 get_current_target 接口；
            # 22 号整机不再用“对准每个路径点”的方式，因此普通点给宽松阈值。
            kind = p_curr.get('kind', '')
            if kind in ('goods', 'shelf'):
                pos_tol = 0.05
                yaw_tol = 0.06
            else:
                pos_tol = 0.35
                yaw_tol = 0.5

            smoothed.append({'x': p_curr['x'], 'y': p_curr['y'],
                             'yaw': yaw_curr_target, 'pos_tol': pos_tol,
                             'yaw_tol': yaw_tol, 'is_arc': False})
        
        return smoothed

    def plan(self, start_id, end_id):
        s_actual = self._find_nearest_connected_node(start_id)
        e_actual = self._find_nearest_connected_node(end_id)
        
        if s_actual == -1 or e_actual == -1 or s_actual == e_actual:
            raw_path_ids = [start_id, end_id]
        else:
            dijkstra_path = self._dijkstra(s_actual, e_actual)
            if not dijkstra_path:
                raw_path_ids = [start_id, end_id]
            else:
                raw_path_ids = [start_id] + dijkstra_path + [end_id]
                deduped = []
                for pid in raw_path_ids:
                    if not deduped or deduped[-1] != pid:
                        deduped.append(pid)
                raw_path_ids = deduped
                
        self.smoothed_path = self._smooth_path(raw_path_ids)
        self.current_idx = 0
        return self.smoothed_path

    def get_current_target(self, pos, yaw):
        self.just_reached = False
        
        while True:
            if self.current_idx >= len(self.smoothed_path):
                return None, None, True, False
            
            wp = self.smoothed_path[self.current_idx]
            target_pos = np.array([wp['x'], wp['y'], 0.0])
            target_yaw = wp['yaw']
            
            dx = wp['x'] - pos[0]
            dy = wp['y'] - pos[1]
            dist = np.hypot(dx, dy)
            
            yaw_err = target_yaw - yaw
            yaw_err = np.arctan2(np.sin(yaw_err), np.cos(yaw_err))
            
            if dist < wp['pos_tol'] and abs(yaw_err) < wp['yaw_tol']:
                self.last_reached_is_arc = wp.get('is_arc', False)
                self.current_idx += 1
                self.just_reached = True
                if self.current_idx >= len(self.smoothed_path):
                    return None, None, True, False
                continue
            else:
                return target_pos, target_yaw, False, False
