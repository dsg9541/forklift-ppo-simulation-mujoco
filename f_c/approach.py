"""进走廊（货堆正前方）的路径点生成器——**纯几何、普适、与模型无关**。

为什么需要（2026-09-15 实测 tools/turn_radius.py）：
  · 本车（舵角伺服 / ppo_2）极限转弯半径只有 **1.4~1.6m**；
  · A/B 排的"车道"（路径图 y≈+2.3 / -5.7）正好穿过走廊口，锚点在车道
    **外侧 1.85m**，所以"沿车道直接打 90° 弯进走廊"必然甩出走廊 ~1.5m，
    模型只能一边绕圈一边蹭（用户否掉的失败模式）；
  · 正解：**先把车从车道平移 R 米到"副车道"**，再从副车道打 90° 圆弧，
    终点正好落在"锚点正后方 2m、车头朝货堆"的预交接位姿上。全程半径 = R，
    引导点每次只偏十几度，ppo_2 自己就能开（不需要程序接管）。

两段路径：
  ① 平移弧（S 形两段 ±θ）：从车道点 B 走到副车道点 A，净平移 R、净前进
     L=2R·sinθ；取 θ=60° ⇒ 平移 = R、前进 = 1.73R。
  ② 进走廊弧：半径 R、90°，从 A 走到预交接点 P，车头由 h 转到 u。
离散成 step 米一个点（默认 0.7m），拼到折线里即可。
"""
import numpy as np


def _rot(v, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _arc(center, radius, start_pt, sweep_deg, step):
    """从 start_pt 绕 center 扫 sweep_deg（正=逆时针），返回不含起点的点列。"""
    a0 = np.arctan2(start_pt[1] - center[1], start_pt[0] - center[0])
    sweep = np.deg2rad(sweep_deg)
    n = max(2, int(abs(sweep) * radius / max(step, 0.1)))
    return [center + radius * np.array([np.cos(a), np.sin(a)])
            for a in np.linspace(a0, a0 + sweep, n + 1)[1:]]


def corridor_approach(g, from_dir, r=1.60, via_back=2.00, step=0.70):
    """生成"车道 → 预交接点"的路径点。

    g        : (x, y, yaw) 引导锚点；yaw = 进货方向 u
    from_dir : 车沿车道的前进方向（二维单位向量，必须近似垂直于 u；
               例如 A2 从东侧车道来 ⇒ (-1, 0) 朝西）
    r        : 转弯半径（实测极限 1.4~1.6m）
    返回     : (路径点列表, 说明)；路径点从车道上的 B 到预交接点 P（含两者）
    """
    g = np.asarray(g, dtype=float)
    u = np.array([np.cos(g[2]), np.sin(g[2])])
    h = np.asarray(from_dir, dtype=float)
    h = h / max(np.linalg.norm(h), 1e-9)
    P = g[:2] - via_back * u                    # 预交接点（车头朝 u）
    right = np.array([u[1], -u[0]])             # u 的右法向
    cross = h[0] * u[1] - h[1] * u[0]
    C = P + r * right if cross < 0 else P - r * right   # 圆弧圆心
    # 弧②入口 A：车头 = h。右转（cross<0）时圆心在车的右前方 ⇒
    # 半径方向 = rot(h, +90°)（指向车后侧）；左转反之。
    A = C + r * _rot(h, 90.0 if cross < 0 else -90.0)
    arc2 = _arc(C, r, A, -90.0 if cross < 0 else 90.0, step)
    # 引道段：把"车道上的点 B"接到弧②入口 A。
    # 车道与 h 平行，A 到车道的垂直距离 = lead_lat；沿 h 方向回退 lead_along。
    n_perp = _rot(h, 90.0)
    lead_lat, lead_along = 1.75, 2.50
    # 车道在 A 的"另一侧"：右转时车道在 rot(h,-90°) 方向
    B = A + lead_lat * _rot(h, -90.0 if cross < 0 else 90.0) - lead_along * h
    pts = [B, A] + arc2
    return pts, (f"R={r:.2f} 车道点({B[0]:+.2f},{B[1]:+.2f}) "
                 f"预交接点({P[0]:+.2f},{P[1]:+.2f})")
