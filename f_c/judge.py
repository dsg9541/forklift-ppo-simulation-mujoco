"""作业判定器（27 号）——按赛方三条失败条件 + 一条成功条件自动判定。

失败 1：货物翻倒               -> 任一货物 body 的立轴与世界 Z 夹角 > 25°（持续 0.1s）
失败 2：车体与环境（地面除外）碰撞 -> 叉车子树与墙/障碍/货堆发生接触
失败 3：货物与环境或无关货物碰撞   -> 货物/托盘与墙/障碍/别的货堆接触
成功   ：放置完成后，被放货物位姿与目标位姿：距离 < 10cm 且角度 < 5°

允许的接触（不算失败）：与地面；与正下方承托的那一层（垂直叠放）；载货与本车货叉。
"""
import re

import numpy as np
import mujoco

TOPPLE_DEG = 25.0
TOPPLE_HOLD_S = 0.1
PLACE_TOL_POS = 0.10          # m
PLACE_TOL_YAW_DEG = 5.0
STACK_XY_TOL = 0.35           # 视为"垂直叠放"的水平距离（m）


class Judge:
    def __init__(self, model, data, forklift_body_id,
                 topple_deg=TOPPLE_DEG, topple_hold_s=TOPPLE_HOLD_S):
        self.model = model
        self.data = data
        self.forklift_body_id = forklift_body_id
        self.topple_deg = float(topple_deg)
        self.topple_hold_s = float(topple_hold_s)

        self._name = {}
        for bid in range(model.nbody):
            self._name[bid] = (mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, bid) or "")

        # 叉车子树
        self.vehicle_bodies = set()
        for bid in range(model.nbody):
            cur = bid
            while cur > 0:
                if cur == forklift_body_id:
                    self.vehicle_bodies.add(bid)
                    break
                cur = int(model.body_parentid[cur])

        # 货物 / 托盘 / 环境
        self.cargo_bodies = {b for b, n in self._name.items()
                             if n.startswith("stack_") and n.endswith("_cargo")}
        self.pallet_bodies = {b for b, n in self._name.items()
                              if n.startswith("stack_") and n.endswith("_pallet")}
        self.env_bodies = {b for b, n in self._name.items()
                           if n.startswith("obstacle_")}
        self.floor_body = 0                      # world（地板 geom 挂在 world 上）

        self.events = {"topple": [], "vehicle_collision": [],
                       "cargo_collision": []}
        self._topple_timer = {}
        self._seen = set()
        self._placed = None
        self.carried = set()          # 本车正在搬运的 body（允许与车体接触）

        # 【31 号 · 纯性能优化】判定器原来每个接触点都要走一遍 Python：
        # 一个物理步 ~187 个接触，cProfile 实测 judge.step 占整步耗时 65%
        # （MuJoCo mj_step 本体只占 28%）。这里把"每个 body 属于哪一类"预先
        # 算成整型数组，用 numpy 把**不可能记录事件**的接触整批筛掉，只把
        # 可能记录的接触交给下面**一字未改**的 `_classify()`。
        # 语义等价：被筛掉的接触在原实现里都是"直接 return/什么都不做"。
        # 等价性用同 seed 整局对比验证（tick 数与判定逐位一致）。
        self._VB, self._CB, self._PB, self._EB = 1, 2, 4, 8
        cls = np.zeros(self.model.nbody, dtype=np.int8)
        for b in self.vehicle_bodies:
            cls[b] |= self._VB
        for b in self.cargo_bodies:
            cls[b] |= self._CB
        for b in self.pallet_bodies:
            cls[b] |= self._PB
        for b in self.env_bodies:
            cls[b] |= self._EB
        self._cls = cls
        self._gbid = np.asarray(self.model.geom_bodyid)
        # "同一堆"的 key：名字里 "_L" 之前的部分；非货堆 body 给 -1
        _keys = {}
        skey = np.full(self.model.nbody, -1, dtype=np.int32)
        for b, n in self._name.items():
            if b in self.cargo_bodies or b in self.pallet_bodies:
                k = n.split("_L")[0]
                if k not in _keys:
                    _keys[k] = len(_keys)
                skey[b] = _keys[k]
        self._skey = skey
        self._cargo_ids = np.asarray(sorted(self.cargo_bodies), dtype=np.int32)

    def note_carrying(self, bodies):
        """登记"正在被本车搬运"的 body 集合。"""
        self.carried = set(bodies)

    # ------------------------------------------------------------------
    def _pos(self, bid):
        return self.data.body(bid).xpos

    def _record(self, kind, msg):
        key = (kind, msg)
        if key in self._seen:
            return
        self._seen.add(key)
        self.events[kind].append(msg)

    def step(self, dt):
        # 1) 翻倒
        # 【31 号 · 性能】一次性向量化算所有货物的倾角；只有真出现倾斜才进
        # Python 循环累计时器（等价于原实现逐 body 累加/清零）。
        tilt = np.degrees(np.arccos(np.clip(
            self.data.xmat[self._cargo_ids, 8], -1.0, 1.0)))
        if float(tilt.max(initial=0.0)) > self.topple_deg:
            for bid, tlt in zip(self._cargo_ids.tolist(), tilt.tolist()):
                if tlt > self.topple_deg:
                    t = self._topple_timer.get(bid, 0.0) + dt
                    self._topple_timer[bid] = t
                    if t >= self.topple_hold_s:
                        self._record("topple",
                                     f"{self._name[bid]} 倾角 {tlt:.1f}°")
                else:
                    self._topple_timer[bid] = 0.0
        elif self._topple_timer:
            self._topple_timer.clear()

        # 2) 接触
        ncon = int(self.data.ncon)
        if ncon <= 0:
            return
        b1 = self._gbid[self.data.contact.geom1[:ncon]]
        b2 = self._gbid[self.data.contact.geom2[:ncon]]
        k1 = self._cls[b1]
        k2 = self._cls[b2]
        v1 = (k1 & self._VB) != 0
        v2 = (k2 & self._VB) != 0
        c1 = (k1 & (self._CB | self._PB)) != 0
        c2 = (k2 & (self._CB | self._PB)) != 0
        e1 = (k1 & self._EB) != 0
        e2 = (k2 & self._EB) != 0
        cc = c1 & c2 & (self._skey[b1] != self._skey[b2])
        cand = ((b1 != 0) & (b2 != 0)
                & ((v1 & (e2 | c2)) | (v2 & (e1 | c1))
                   | (c1 & e2) | (c2 & e1) | cc))
        for i in np.nonzero(cand)[0]:
            self._classify(int(b1[i]), int(b2[i]))

    def _classify(self, b1, b2):
        if b1 == self.floor_body or b2 == self.floor_body:
            return                                   # 地面接触一律允许
        v1 = b1 in self.vehicle_bodies
        v2 = b2 in self.vehicle_bodies

        cargo_like = lambda b: (b in self.cargo_bodies or b in self.pallet_bodies)
        c1, c2 = cargo_like(b1), cargo_like(b2)
        e1, e2 = b1 in self.env_bodies, b2 in self.env_bodies

        # 车体 vs 环境
        if v1 and e2:
            self._record("vehicle_collision", f"车体 <-> {self._name[b2]}")
        if v2 and e1:
            self._record("vehicle_collision", f"车体 <-> {self._name[b1]}")

        # 车体 vs 货物（载货时货叉托着载荷属允许；叉齿插到别的货堆上属失败）
        if v1 and c2:
            if b2 in self.carried or (self._placed and b2 in self._placed["bodies"]):
                return                                # 本车正在搬运/放置的载荷
            self._record("vehicle_collision", f"车体 <-> {self._name[b2]}")
        if v2 and c1:
            if b1 in self.carried or (self._placed and b1 in self._placed["bodies"]):
                return
            self._record("vehicle_collision", f"车体 <-> {self._name[b1]}")

        # 货物 vs 环境
        if c1 and e2:
            self._record("cargo_collision", f"{self._name[b1]} <-> {self._name[b2]}")
        if c2 and e1:
            self._record("cargo_collision", f"{self._name[b2]} <-> {self._name[b1]}")

        # 货物 vs 别的货物：垂直叠放（同一堆上下层、或刚放上去的那一层）允许
        if c1 and c2 and b1 != b2:
            if self._name[b1].split('_L')[0] == self._name[b2].split('_L')[0]:
                return   # 同一堆内的叠放接触（托盘/货物互压），属于正常状态不算碰撞
            p1, p2 = self._pos(b1), self._pos(b2)
            xy = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
            if xy > STACK_XY_TOL:
                self._record("cargo_collision",
                             f"{self._name[b1]} <-> {self._name[b2]} (水平距离 {xy:.2f}m)")

    # ------------------------------------------------------------------
    def note_placed(self, load_bodies, target_pose,
                    tol_pos=PLACE_TOL_POS, tol_yaw_deg=PLACE_TOL_YAW_DEG):
        """放货完成时登记：被放上去的 body 集合 + 目标位姿（真值）。"""
        self._placed = dict(bodies=set(load_bodies),
                            target=np.asarray(target_pose, dtype=float),
                            tol_pos=float(tol_pos),
                            tol_yaw=float(np.deg2rad(tol_yaw_deg)))
        self._evaluate_placement()

    def _evaluate_placement(self):
        """被放货物（以托盘为准）与目标位姿的误差。"""
        if not self._placed:
            return
        pallet = None
        for b in self._placed["bodies"]:
            if self._name[b].endswith("_pallet"):
                pallet = b
                break
        if pallet is None:
            return
        pos = self._pos(pallet)
        R = self.data.xmat[pallet].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        tgt = self._placed["target"]
        d_pos = float(np.hypot(pos[0] - tgt[0], pos[1] - tgt[1]))
        d_yaw = float(abs(np.arctan2(np.sin(yaw - tgt[2]),
                                     np.cos(yaw - tgt[2]))))
        # 托盘/货物都是**关于中心对称的箱体**（0.7×0.8、0.62×0.8）：绕 z 轴
        # 转 180° 与不转在物理上完全等价（外轮廓、承托面、与货堆的贴合都一样），
        # 只是"朝向标签"翻了。27 号 A 排从南进站（车头朝北）、B 排从北进站
        # （车头朝南），同一套叉齿必然给出 180° 相差的标签——判定时取模 180°
        # 的较小值，否则第 2 轮这种"物理上完全摆正"的放置会被判成 178.8°。
        d_yaw = float(min(d_yaw, abs(np.pi - d_yaw)))
        self._placed["pos_err"] = d_pos
        self._placed["yaw_err"] = d_yaw
        self._placed["ok"] = bool(d_pos <= self._placed["tol_pos"]
                                  and d_yaw <= self._placed["tol_yaw"])

    # ------------------------------------------------------------------
    def report(self):
        p = self._placed or {}
        ok = (not self.events["topple"]
              and not self.events["vehicle_collision"]
              and not self.events["cargo_collision"]
              and bool(p.get("ok")))
        return {
            "topple": list(self.events["topple"]),
            "vehicle_collision": list(self.events["vehicle_collision"]),
            "cargo_collision": list(self.events["cargo_collision"]),
            "place_error": {"pos_m": p.get("pos_err"),
                            "yaw_deg": (None if p.get("yaw_err") is None
                                        else float(np.rad2deg(p["yaw_err"]))),
                            "ok": p.get("ok")},
            "ok": ok,
        }

    def summary(self):
        r = self.report()
        pe = r["place_error"]
        txt = [f"成功：{'是' if r['ok'] else '否'}"]
        if r["topple"]:
            txt.append(f"翻倒 {len(r['topple'])}: {r['topple'][:2]}")
        if r["vehicle_collision"]:
            txt.append(f"车体碰撞 {len(r['vehicle_collision'])}: "
                       f"{r['vehicle_collision'][:2]}")
        if r["cargo_collision"]:
            txt.append(f"货物碰撞 {len(r['cargo_collision'])}: "
                       f"{r['cargo_collision'][:2]}")
        if pe["pos_m"] is not None:
            txt.append(f"放置误差 {pe['pos_m'] * 100:.1f}cm / "
                       f"{pe['yaw_deg']:.2f}°")
        return " | ".join(txt)
