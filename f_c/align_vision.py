"""经典（非学习）末段对准：只用门架相机 + 雷达，不读取货物真值位置。

对准标准（用户指定）：
  1) 雷达左右对称：车体前向那一组射线（±45°，20 束）左右镜像差的均值 → 0；
  2) 货堆在图片正中：粉色货物像素在图像水平方向的中心 → 图像中心。

另加一个"站位距离"判据：前向射线的最小回波 = 货堆正面距离，
把它对到标定值（在名义站位下离线测一次得到，运行时不读真值）。
"""
import numpy as np

# ---- 颜色阈值：货物是粉色 (rgba 1,0.4,0.7)，受光后大致 R>G、B>G ----
DET_R_MIN = 110
DET_RG = 40
DET_BG = 5
DET_MIN_PIXELS = 40

FWD_RAYS = (0, 20)          # 车体前向雷达在 279 维里的切片（±45°）
LIDAR_MAX = 8.0

DIST_TOL = 0.03             # 站位距离容差 (m)
LAT_TOL = 0.03              # 图像横向归一化容差
MAX_STEPS = 400


def stack_mask(image):
    """粉色货物掩码。image: (3,H,W) uint8。"""
    r = image[0].astype(np.int16)
    g = image[1].astype(np.int16)
    b = image[2].astype(np.int16)
    return ((r > DET_R_MIN) & ((r - g) > DET_RG) & ((b - g) > DET_BG))


def image_lateral_error(image):
    """货物在图像水平方向的偏移，归一化到 [-1, 1]（0 = 正中）。"""
    m = stack_mask(image)
    n = int(m.sum())
    if n < DET_MIN_PIXELS:
        return None, n
    cols = np.where(m.any(axis=0))[0]
    center = 0.5 * (cols.min() + cols.max())
    half = (image.shape[2] - 1) / 2.0
    return float((center - half) / half), n


def image_lateral_error_ex(image, edge_margin=1):
    """同上，另返回 (像素列范围, 是否贴边)。

    货物一旦有一部分出画面，掩码中心就会被"拉回"画面中心，横向读数不可信；
    调用方看到 edge=True 时应改用雷达单独收偏航，不要用这个横向值。
    """
    m = stack_mask(image)
    n = int(m.sum())
    if n < DET_MIN_PIXELS:
        return None, n, None, True
    cols = np.where(m.any(axis=0))[0]
    c0, c1 = int(cols.min()), int(cols.max())
    w = image.shape[2]
    edge = (c0 <= edge_margin) or (c1 >= w - 1 - edge_margin)
    half = (w - 1) / 2.0
    center = 0.5 * (c0 + c1)
    return float((center - half) / half), n, (c0, c1), edge


CARGO_WIDTH_M = 0.80         # 货物正面宽度（27 号场景实测）
CAM_TAN_HALF = np.tan(np.deg2rad(30.0))   # mast_cam fovy=60、画面为正方形


def stack_lateral_error(image, x_dist, cargo_w=CARGO_WIDTH_M):
    """贴边也不失真的横向误差：用"已知物宽 + 已知距离"预测像素宽度，
    贴边时只用没被裁掉的那一侧边缘推中心。

    x_dist: 相机到货堆中心的前向距离 (m)
    返回 (lat, npix, cols, clipped)；lat 的归一化定义同 image_lateral_error。
    """
    m = stack_mask(image)
    n = int(m.sum())
    if n < DET_MIN_PIXELS:
        return None, n, None, True
    cols = np.where(m.any(axis=0))[0]
    c0, c1 = int(cols.min()), int(cols.max())
    w = image.shape[2]
    half = (w - 1) / 2.0
    pred_w = cargo_w / (2.0 * max(x_dist, 0.3) * CAM_TAN_HALF) * (w - 1)
    left_clip = c0 <= 1
    right_clip = c1 >= w - 2
    if right_clip and not left_clip:
        center = c0 + pred_w / 2.0
        clipped = True
    elif left_clip and not right_clip:
        center = c1 - pred_w / 2.0
        clipped = True
    else:
        center = 0.5 * (c0 + c1)
        clipped = left_clip
    return float((center - half) / half), n, (c0, c1), clipped


MIN_VALID = 0.40            # 小于 0.4m 的回波是"车体自身结构"（门架/货叉），必须剔除


def lidar_metrics(lidar):
    """返回 (货堆正面距离 m, 左右对称误差 m, 有效回波数)。

    注意：前向那 20 束里，靠中间的几束会打到自己的门架（~0.15m），
    直接取 min 会得到"自遮挡"距离，所以这里先把 <0.4m 的回波剔除。
    """
    seg = np.asarray(lidar[FWD_RAYS[0]:FWD_RAYS[1]], dtype=float) * LIDAR_MAX
    valid = seg >= MIN_VALID
    if not valid.any():
        return float("inf"), float("inf"), 0
    d = np.where(valid, seg, LIDAR_MAX)
    dist = float(d.min())
    sym = float(np.mean(np.abs(d - d[::-1])))
    return dist, sym, int(valid.sum())


def compute_action(lat_err, dist_err, k_lat=1.2, k_dist=0.8, base=0.10):
    """由两个误差给出 (v_cmd, steer_cmd)。

    距离误差决定进退；横向误差决定转向；base 是"蠕行"速度——横向没对正时
    必须有前进速度，否则原地打方向盘是纠正不了图像偏移的。
    """
    v = base if abs(lat_err) > LAT_TOL else 0.0
    v = float(np.clip(k_dist * dist_err + v, -0.45, 0.45))
    steer = float(np.clip(-k_lat * lat_err, -0.35, 0.35))
    return v, steer


# ======================================================================
# 货堆正面平面拟合：把前向 20 束的命中点拟合成一条直线，一次拿到两个量
#   · 法向偏角 psi_car —— 车头相对货堆正面法向的偏角（判"左右对称"的本质量）
#   · 垂直距离 d_perp —— 门架雷达小球到货堆正面的垂距（判"站位"的本质量）
# 只有货堆正面是一块近似平面（托盘/货物正面）时这个模型才成立；
# 命中点少于 FACE_MIN_PTS 时返回 None，调用方走"看不到货堆"的分支。
# ======================================================================
FACE_ANGLES = np.deg2rad(-45.0 + 90.0 * np.arange(20) / 19.0)
# 只用中间 ±25° 的那 10 束拟合货堆正面：大角度那几束在大偏角时打到货堆的
# 侧面，混进来会让"平面"拟合失真（实测大偏角时残差判据会一直拒绝）。
FACE_USE = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
FACE_MIN_PTS = 2
FACE_MAX_RESID_M = 0.06      # 拟合残差超过这个值 → 命中点不共面，读数不可信


def face_fit(fwd_rays, angles=FACE_ANGLES, min_valid=MIN_VALID, max_dist=LIDAR_MAX):
    """返回 (psi_car_rad, d_perp_m, n_pts)；命中点不足或退化时返回 (None, None, n)。

    psi_car = 0 表示车头正对货堆正面（左右完全对称）；>0 表示车头相对
    理想朝向逆时针偏了 psi_car。
    """
    d = np.asarray(fwd_rays, dtype=float)
    globals()["LAST_FIT_INFO"] = {}
    a = np.asarray(angles, dtype=float)
    if d.shape[-1] == a.shape[-1] and d.shape[-1] > FACE_USE.max():
        # 先用中间 ±25° 的窄扇面拟合；窄扇面看不到货堆正面时（比如车头偏得
        # 比较多、正面落在 25°~45° 之间），再退回整条 ±45° 扇面。
        r = _fit_subset(d[FACE_USE], a[FACE_USE])
        if r[0] is not None:
            globals()["LAST_FIT_INFO"]["band"] = "narrow"
            return r
        r = _fit_subset(d, a)
        if r[0] is not None:
            globals()["LAST_FIT_INFO"]["band"] = "wide"
        return r
    return _fit_subset(d, a)


def _fit_subset(d, a, min_valid=MIN_VALID, max_dist=LIDAR_MAX):
    ok = (d >= min_valid) & (d <= max_dist - 0.01)
    n = int(ok.sum())
    if n < FACE_MIN_PTS:
        globals()["LAST_FIT_INFO"] = {"why": "pts", "n": n}
        return None, None, n
    px = d[ok] * np.cos(a[ok])
    py = d[ok] * np.sin(a[ok])
    pts = np.stack([px, py], axis=1)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    if not np.all(np.isfinite(cov)):
        globals()["LAST_FIT_INFO"] = {"why": "bad"}
        return None, None, n
    w, v = np.linalg.eigh(cov)
    normal = v[:, 0]                     # 最小特征值方向 = 平面法向
    if normal[0] > 0.0:                  # 法向必须指向车体（-x 一侧）
        normal = -normal
    c = float(normal @ mean)
    resid = float(np.sqrt(np.mean((pts @ normal - c) ** 2)))
    if resid > FACE_MAX_RESID_M:
        globals()["LAST_FIT_INFO"] = {"why": "resid", "n": n,
                                      "resid": round(resid, 3)}
        return None, None, n
    d_perp = -c                          # 平面到雷达小球的垂距
    if d_perp <= 0.0:
        globals()["LAST_FIT_INFO"] = {"why": "dperp", "n": n}
        return None, None, n
    ang = float(np.arctan2(normal[1], normal[0])) - np.pi
    ang = float(np.arctan2(np.sin(ang), np.cos(ang)))
    globals()["LAST_FIT_INFO"] = {"why": "ok", "n": n,
                                  "resid": round(resid, 3),
                                  "psi": round(np.rad2deg(ang), 1)}
    return ang, d_perp, n


# ======================================================================
# 程序控制末段对准（DockAligner）
# ----------------------------------------------------------------------
# 两个观测量由上面的 face_fit + 相机得到，且实测互不耦合：
#   psi  [rad]  货堆正面法向在车体系里的偏角（= -车身偏航误差）→ 判"左右对称"
#   lat  [-]    粉色货物在门架图像里的横向归一化误差（0=正中）
#   d_perp [m]  门架雷达小球到货堆正面的垂距（= 站位）
# 实测标定（tools/dock_probe.py，第 1/2/3 轮一致）：
#   psi = -Δyaw（纯偏航时精确成立，与横向偏移无关）
#   lat = 1.28·L - 3.52·psi   →   L = (lat + 3.52·psi) / 1.28
#   d_perp 与横向偏移、偏航都无关，只随前后站位变化
# 其中 L [m] 是车身参考点相对"理想入库中线"的横向偏移（车偏左为正）。
# 控制用纯追踪：瞄准中线上、车前方 lookahead 处的点；因为货堆正面是一块
# 平面，横向偏移不会改变垂距，所以站位只用 d_perp 闭环，不会互相打架。
# ======================================================================
# 相机几何：门架相机在车体前 1.60m、高 2.35m、俯角 42°，水平半视场 30°。
# 实测横向标定：lat = -CAM_K · θ_cam，θ_cam 是"货堆中心"在相机系里的水平方位角，
# CAM_K = 1.577（用 tools/dock_probe.py 的纯横向/纯偏航两组数据标定）。
CAM_AHEAD_M = 1.60
CAM_K = 1.577
FACE_TO_CENTER_M = 1.25      # 相机到货堆中心的标称前向距离（= 2.85 - 1.60）

LAT_PER_L = 1.28             # 小角度近似下的等效值，仅用于日志/对照
LAT_PER_PSI = 3.52
DOCK_STANDOFF_M = 1.270      # 到位时 d_perp（tools/align_signature.py 实测）


def lateral_from_image(lat, psi, gap=0.0):
    """由图像横向误差 + 雷达法向角还原车身横向偏移 L (m)。

    精确几何反解（不再用小角度线性近似）：
        θ_cam = -lat / CAM_K                  # 货堆中心在相机系里的方位角
        θ_w   = θ_cam - psi_car               # 换到"理想入库系"
        L     = -X·tan(θ_w) - CAM_AHEAD·sin(psi_car)
    其中 psi_car = -psi，X = 相机到货堆中心的前向距离。
    lat 或 psi 缺失时返回 None（调用方退回"只看 psi"）。
    """
    if lat is None or psi is None:
        return None
    psi_car = -float(psi)
    theta_cam = -float(lat) / CAM_K
    theta_w = theta_cam + psi_car
    x = FACE_TO_CENTER_M + float(gap)
    return float(-x * np.tan(theta_w) - CAM_AHEAD_M * np.sin(psi_car))


# ----------------------------------------------------------------------
# 程序对准用的二自由度车辆参数（tools/dyn_id.py + tools/law_test.py 实测）：
#     dψ_car/ds = KAPPA · steer            （KAPPA ≈ 0.53 /m，v≈0.12~0.14m/s）
#     d L   /ds = ψ_car - BETA · steer
# 于是 z = L + (BETA/KAPPA)·ψ_car 满足 z' = ψ_car、ψ_car' = KAPPA·steer ——
# 一个双积分器，用 z、ψ_car 的 PD 就能同时把它们收到 0（纯比例反馈做不到，
# 因为该线性化对任何静态状态反馈都是鞍点）。
# BETA/KAPPA 是关键参数：它错一点点闭环就会出现"几乎不收敛"的慢模态。
# tools/law_test.py 扫 Z_RATIO∈[0.3,1.9] 实测收敛残差，0.5 附近最优
# （0.44m 初始横向差、6m 行程：残差 0.4cm / 0.25°，叉尖 0.6cm）。
# ----------------------------------------------------------------------
KAPPA = 0.53
Z_RATIO = 0.70


class DockAligner:
    """程序控制末段对准：转向对准 → 纯追踪进站 → 回退重试，直到三个判据同时满足。

    update() 每控制周期调用一次，输入当周期的观测量与引导方位，返回
    (v_cmd, steer_cmd, status)；status ∈ {"search","turn","dock","retreat","done"}。
    全程只用车体里程 + 门架相机 + 雷达，不读货物真值。
    """

    def __init__(self, standoff=DOCK_STANDOFF_M, standoff_tol=0.030,
                 lat_tol=0.030, yaw_tol_deg=2.5, omega=3.0,
                 steer_max=0.50, v_creep=0.20, v_fast=0.20,
                 v_back=0.20, min_start_gap=2.80, turn_tol_deg=25.0,
                 k_turn=1.0, prior_gain=0.35, yaw_cap_deg=16.0):
        self.standoff = float(standoff)
        self.standoff_tol = float(standoff_tol)
        self.lat_tol = float(lat_tol)
        self.yaw_tol = float(np.deg2rad(yaw_tol_deg))
        self.omega = float(omega)
        self.steer_max = float(steer_max)
        self.v_creep = float(v_creep)
        self.v_fast = float(v_fast)
        self.v_back = float(v_back)
        self.min_start_gap = float(min_start_gap)
        self.turn_tol = float(np.deg2rad(turn_tol_deg))
        self.k_turn = float(k_turn)
        self.prior_gain = float(prior_gain)
        self.l_gain = 1.0
        self.yaw_cap = float(np.deg2rad(yaw_cap_deg))
        # 双积分器 PD：z'' = KAPPA·steer，取二重极点 -omega
        self.k_z = self.omega ** 2 / KAPPA
        self.k_psi = 2.0 * self.omega / KAPPA
        self.reset()

    def reset(self):
        self.mode = "approach"
        self.entered = False
        self.retries = 0
        self.last = {}

    # ---------------------------------------------------------------- 内
    def _aligned(self, L, psi, gap):
        if psi is None:
            return False
        if abs(psi) > self.yaw_tol or abs(gap) > self.standoff_tol:
            return False
        return L is None or abs(L) <= self.lat_tol

    def update(self, lat, fwd_rays, bearing=None, lat_clipped=False,
               l_prior=None, heading_err=None):
        """lat 建议传 stack_lateral_error() 的抗贴边读数；
        l_prior 只在相机完全看不到货物时兜底（引导 + 里程，0.2m 量级精度），
        而且只用 1/3 增益——引导先验误差 0.2m 时直接按满增益打舵会把车甩飞
        （实测：真值横向 +0.28m 时先验给出 -0.02m，符号是反的）。"""
        # ---- 姿态粗对准：先把车头摆到"接近货堆正面"的方向 ----
        # 本车车身转得快（≈0.53rad/m 行程）、轨迹弯得慢，所以 30° 以上的
        # 大偏角靠双积分律收不住，得先用（雷达优先的）车头偏差转到位。
        if (not self.entered and self.mode == "approach"
                and heading_err is not None
                and abs(heading_err) > self.turn_tol):
            self.mode = "turn"
        if self.mode == "turn":
            if heading_err is not None and abs(heading_err) <= self.turn_tol:
                self.mode = "approach"
            else:
                e = 0.0 if heading_err is None else heading_err
                self.last = dict(psi=None, d_perp=None, L=None, turn=True)
                return (self.v_creep,
                        float(np.clip(self.k_turn * e, -self.steer_max,
                                      self.steer_max)), "turn")

        psi, d_perp, npts = face_fit(fwd_rays)
        # 读数可信性：命中点太少 / 法向角离谱（打到别的面）都按"看不到"处理，
        # 否则会拿着错误的 psi 把车转飞。
        if (psi is None or d_perp is None or npts < 2
                or abs(psi) > np.deg2rad(40.0)):
            self.last = dict(psi=psi, d_perp=d_perp, L=None, npts=npts)
            # 看不到货堆：先把"退到起跑线"这件事做完，别和它对着干
            fr = np.asarray(fwd_rays, dtype=float)
            valid = fr[(fr > MIN_VALID) & (fr < LIDAR_MAX - 0.01)]
            near = float(valid.min()) if valid.size else float("inf")
            gap_est = near - self.standoff if np.isfinite(near) else None
            if self.mode == "retreat":
                if gap_est is not None and gap_est >= self.retreat_until:
                    self.mode = "approach"
                else:
                    return -self.v_back, 0.0, "retreat"
            # 前面还很空才缓慢前探，绝不冲进货物
            if near < 2.0:
                return 0.0, 0.0, "search"
            return self.v_creep, 0.0, "search"

        L = None if lat_clipped else lateral_from_image(lat, psi)
        if L is not None:
            self.l_gain = 1.0
        elif l_prior is not None:
            L = float(l_prior)
            self.l_gain = self.prior_gain
        else:
            self.l_gain = 0.0
        gap = d_perp - self.standoff
        self.last = dict(psi=psi, d_perp=d_perp, L=L, npts=npts, lat=lat,
                         gap=gap, l_prior=l_prior, clipped=lat_clipped,
                         l_gain=self.l_gain)

        z = None if L is None else float(L) + Z_RATIO * (-psi)

        # ---- 到位判定 ----
        if self._aligned(L, psi, gap):
            return 0.0, 0.0, "done"

        # ---- 一次完整的"退到够远 -> 对准着进站"循环 ----
        # 双积分律每米行程衰减 e^-omega，需要留够行程；一次进站没对正就
        # 直接退到起跑线重来，绝不在"到站线"上来回蹭。
        if self.mode == "retreat":
            if gap >= self.retreat_until:
                self.mode = "approach"
            else:
                return -self.v_back, 0.0, "retreat"

        if gap <= 0.03 or (not self.entered and gap < self.min_start_gap):
            # 进站前先在起跑线上站好（第一帧太近也要先退够）
            self.entered = True
            self.retries += 1
            self.retreat_until = self.min_start_gap
            self.mode = "retreat"
            return -self.v_back, 0.0, "retreat"
        self.entered = True

        steer = self._law(L, psi, sign=+1.0)
        side_ok = (z is None or abs(z) <= self.lat_tol) \
            and abs(psi) <= self.yaw_tol
        if side_ok and gap < 0.20:
            # 已经对正：最后几厘米降速收尾（这一段太短，不影响标定）
            return min(self.v_creep, 1.0 * gap + 0.03), steer, "dock"
        # 全程用同一档速度：β/κ 这个关键比值是随速度变的，变速进站会把标定
        # 打乱（实测变速时残差反而大一个量级）。
        return self.v_creep, steer, "dock"

    def _law(self, L, psi, sign=1.0):
        """双积分器 PD：steer = -sign·(k_z·z + k_psi·ψ_car) / KAPPA。"""
        psi_car = -float(psi)
        if L is None:
            z = None
        else:
            z = self.l_gain * float(L) + Z_RATIO * psi_car
        # 车头偏角超过上限时先只收偏航：车头一旦歪太多，货堆就出了
        # 门架雷达 ±25° 拟合窗口，读数直接消失，后面就没法收敛了。
        if abs(psi_car) > self.yaw_cap:
            u = self.k_psi * psi_car
        else:
            u = self.k_psi * psi_car
            if z is not None:
                u += self.k_z * z
        return float(np.clip(-sign * u / KAPPA, -self.steer_max,
                             self.steer_max))
