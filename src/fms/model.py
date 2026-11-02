from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Callable

from .elements import BLOCKED, FREE, INF, Channel, Create, Exit, Part, QS, Robot, Route
from .params import DEDICATED, Params
from .rng import RandomStreams

SRC = {1: "A", 2: "M1", 3: "M2"}     # точка завантаження шляху S
DST = {1: "M1", 2: "M2", 3: "W"}     # точка вивантаження шляху S


class Model:
    def __init__(self, params: Params, seed: int = 1, trace: bool = False, trace_until: float = INF):
        self.params = params
        self.streams = RandomStreams(seed)
        self.seed = seed
        self.tcurr = 0.0
        self.trace = False
        self.trace_until = trace_until
        self.trace_lines: list[str] = []
        self.calendar: list = []          # календар подій: (момент, номер запису, канал)
        self._seq = 0
        self.completed: list[Part] = []
        self.completed_total = 0
        self.t_stats_start = 0.0
        self._created_at_reset = 0
        self.dedicated = params.strategy == DEDICATED
        p = params
        self.travel = {S: p.distance(SRC[S], DST[S]) for S in (1, 2, 3)}

        # --- фрагмент деталей
        self.create = Create(self, "Create", p.arrival, self.streams.arrivals)
        self.A = QS(self, "A", None, service=lambda part, h: 0.0, release=lambda part, h: h,
                    on_done=lambda part, t: self._ready(part, 1, t))
        self.M1 = QS(self, "M1", p.m1_capacity, service=lambda part, h: h + p.m1_time.sample(self.streams.m1),
                     release=lambda part, h: h, on_done=lambda part, t: self._ready(part, 2, t))
        self.M2 = QS(self, "M2", p.m2_capacity, service=lambda part, h: h + p.m2_time.sample(self.streams.m2),
                     release=lambda part, h: h, on_done=lambda part, t: self._ready(part, 3, t))
        self.W = QS(self, "W", None, service=lambda part, h: h)
        self.exit = Exit(self)
        self.M = {1: self.M1, 2: self.M2}

        # --- СМО перевезень і роботи
        self.T: dict[int, QS] = {}
        self.robot_channels: dict[int, list[Channel]] = defaultdict(list)   # робот -> його пристрої
        self.robots: list[Robot] = []
        routes_of = p.robot_routes()
        if self.dedicated:
            # варіант 1: СМО Rs, пристрої – закріплені за шляхом роботи; після вивільнення робот
            # повертається до точки завантаження, тому пристрій звільняється із затримкою t_вив + d_S
            for S in (1, 2, 3):
                ids = [k + 1 for k, r in enumerate(routes_of) if r[0] == S]
                qs = QS(self, f"R{S}", len(ids), service=lambda part, h, S=S: h + self.travel[S],
                        release=lambda part, h, S=S: h + self.travel[S])
                for ch, k in zip(qs.channels, ids):
                    self.robot_channels[k].append(ch)
                self.T[S] = qs
        else:
            # варіант 2: на кожному шляху СМО Ts з N_R пристроїв; пристрій r – робот r
            n = p.n_robots
            self.robots = [Robot(id=k + 1, pos=("A", "M1", "M2")[k % 3]) for k in range(n)]
            for S in (1, 2, 3):
                qs = QS(self, f"T{S}", n, service=lambda part, h, S=S: h + self.travel[S],
                        release=lambda part, h: h, on_free=lambda ch, t, S=S: self._robot_at(ch, DST[S], t),
                        names=[f"T{S}:R{k + 1}" for k in range(n)])
                for ch in qs.channels:
                    self.robot_channels[ch.index + 1].append(ch)
                self.T[S] = qs
            # фрагмент руху роботів: П – роботи в пунктах, Х – холостий хід
            self.P = QS(self, "П", n, service=lambda robot, h: 0.0, names=[f"П:R{k + 1}" for k in range(n)])
            self.X = QS(self, "Х", n, service=lambda robot, h: p.distance(robot.pos, robot.target),
                        names=[f"Х:R{k + 1}" for k in range(n)])
            for ch in self.X.channels:
                self.robot_channels[ch.index + 1].append(ch)

        # --- маршрути і блокування
        Route(self, self.create, self.A)
        B = self._blockings()
        self.routes: list[Route] = [
            Route(self, self.T[1], self.M1, B[1], "B1", handling="вивільнення"),
            Route(self, self.T[2], self.M2, B[3], "B3", handling="вивільнення"),
            Route(self, self.M1, self.T[2], B[2], "B2", handling="захоплення", select=self._robot_for(2),
                  on_transfer=self._picked(2)),
            Route(self, self.M2, self.T[3], B[4], "B4", handling="захоплення", select=self._robot_for(3),
                  on_transfer=self._picked(3)),
            Route(self, self.A, self.T[1], B[0], "B0", handling="захоплення", select=self._robot_for(1),
                  on_transfer=self._picked(1)),
        ]
        Route(self, self.T[3], self.W, handling="вивільнення")
        Route(self, self.W, self.exit)
        self.stat_qs = [self.M1, self.M2, *self.T.values()]
        self.robot_control: Callable[[float], bool] | None = None
        if not self.dedicated:
            # маршрут П -> Х відкриває лише керування роботами (формули (2.4), (2.5))
            self.route_PX = Route(self, self.P, self.X, lambda: True, "BХ",
                                  select=lambda robot: self.X.channels[robot.id - 1])
            Route(self, self.X, self.P, select=lambda robot: self.P.channels[robot.id - 1],
                  on_transfer=lambda robot, ch, dst, t: self._arrived(robot, t))
            for r in self.robots:
                self.P.accept(r, 0.0, 0.0, self.P.channels[r.id - 1])
            self.stat_qs.append(self.X)
            self.robot_control = self.control
        self.trace = trace
        self.reset_stats()

    def _blockings(self) -> dict[int, Callable[[], bool]]:
        if self.dedicated:
            return {0: self.B0, 1: self.B1, 2: self.B2, 3: self.B3, 4: self.B4}
        return {0: self.B0_n, 1: self.B1, 2: self.B2_n, 3: self.B3, 4: self.B4_n}

    # варіант 1, (2.1): маршрут до робота закритий, якщо робот не вільний
    def B0(self) -> bool:
        return self.T[1].n[FREE] == 0

    def B2(self) -> bool:
        return self.T[2].n[FREE] == 0

    def B4(self) -> bool:
        return self.T[3].n[FREE] == 0

    # обидва варіанти, (2.1): на верстаті немає вільного місця
    def B1(self) -> bool:
        return self.M1.n[FREE] == 0

    def B3(self) -> bool:
        return self.M2.n[FREE] == 0

    # варіант 2, (2.3): деталь переходить до робота, якщо в точці src_S стоїть вільний робот
    # і до верстата-приймача везуть деталі не більше роботів, ніж там вільних місць (D_S = 0)
    def B0_n(self) -> bool:
        return not self.free_at("A") or self.D(1)

    def B2_n(self) -> bool:
        return not self.free_at("M1") or self.D(2)

    def B4_n(self) -> bool:
        return not self.free_at("M2")

    def is_free(self, robot: Robot) -> bool:
        """F_r(t): робот не їде порожнім і жоден його пристрій у СМО T1–T3 не зайнятий."""
        return all(ch.K == FREE for ch in self.robot_channels[robot.id])

    def free_at(self, point: str) -> list[Robot]:
        return [r for r in self.robots if r.pos == point and self.is_free(r)]

    def heading(self, point: str) -> int:
        """Кількість роботів, що їдуть порожнім до точки point."""
        return sum(1 for ch in self.X.channels if ch.K != FREE and ch.request.target == point)

    def u(self, S: int) -> int:
        """u_S(t): роботи, які везуть деталь до верстата шляху S або їдуть по неї."""
        return self.T[S].requests + self.heading(SRC[S])

    def D(self, S: int) -> bool:
        """D_S(t) = 1{u_S(t) > f_S(t)}: до верстата вже везуть на одну деталь більше, ніж там вільних місць."""
        return S != 3 and self.u(S) > self.M[S].n[FREE]

    def waiting(self, S: int) -> int:
        """w_S(t): деталі, що чекають на робота в точці src_S (заблоковані пристрої)."""
        return len(self.route_from(SRC[S]).waiting)

    def route_from(self, name: str) -> Route:
        return {"A": self.A, "M1": self.M1, "M2": self.M2}[name].route

    def control(self, t: float) -> bool:
        """Керування роботами К, формули (2.4) і (2.5): найближчий вільний робот рушає до деталі.

        Робот, який уже стоїть у точці src_S, забирає деталь маршрутом src_S -> T_S (блокування B0, B2, B4).
        """
        moved = False
        while True:
            best = None
            for r in self.robots:
                if not self.is_free(r):
                    continue
                for S in (1, 2, 3):
                    src = SRC[S]
                    q = self.waiting(S) - self.heading(src) - len(self.free_at(src))
                    if r.pos == src or q <= 0 or self.D(S):
                        continue
                    key = (self.params.distance(r.pos, src), self.streams.ties.random())
                    if best is None or key < best[0]:
                        best = (key, r, S)
            if best is None:
                return moved
            (d, _), r, S = best
            r.target = SRC[S]
            ch = self.P.channels[r.id - 1]
            self.route_PX.waiting.remove(ch)
            self.empty_travel[S] += d
            self.log(t, ch.name, f"холостий хід {r.pos} -> {r.target} ({d:g} с) до деталі шляху {S}")
            self.route_PX.transfer(ch, t)
            moved = True

    def _robot_for(self, S: int) -> Callable:
        """Правило вибору пристрою СМО перевезення: вільний робот у точці завантаження."""
        if self.dedicated:
            return lambda part: self.T[S].free_channel()

        def select(part):
            r = self.streams.ties.choice(self.free_at(SRC[S]))
            return self.T[S].channels[r.id - 1]
        return select

    def _robot_at(self, ch: Channel, point: str, t: float) -> None:
        robot = self.robots[ch.index]
        robot.pos = point
        self.log(t, ch.name, f"робот вільний у {point}")

    def _arrived(self, robot: Robot, t: float) -> None:
        robot.pos = robot.target
        self.log(t, f"Х:{robot}", f"кінець холостого ходу, робот у {robot.pos}")

    def _ready(self, part: Part, S: int, t: float) -> None:
        part.t_ready[S] = t

    def _picked(self, S: int) -> Callable:
        def hook(part, ch, dst_ch, t):
            part.t_grab[S] = t
            robot = next(k for k, chs in self.robot_channels.items() if dst_ch in chs)
            self.jobs[(robot, S)] += 1
        return hook

    def handling_time(self) -> float:
        return self.params.handling.sample(self.streams.handling)

    def record_completion(self, part: Part, t: float) -> None:
        part.t_done = t
        self.completed_total += 1
        self.log(t, "вихід", f"{part} на складі, час проходження {t - part.t_arrival:.3f}")
        if t >= self.t_stats_start:
            self.completed.append(part)

    def wip(self) -> int:
        return self.create.created - self.completed_total

    def schedule(self, ch: Channel, t: float) -> None:
        ch.tnext = t
        self._seq += 1
        ch.seq = self._seq
        heapq.heappush(self.calendar, (t, self._seq, ch))

    def _next_event_time(self) -> float:
        cal = self.calendar
        while cal and (cal[0][2].seq != cal[0][1] or cal[0][2].tnext != cal[0][0]):
            heapq.heappop(cal)                     # застарілий запис календаря
        return min(self.create.tnext, cal[0][0] if cal else INF)

    def transition(self, t: float) -> None:
        """Алгоритм 3.2: перетворення стану мережі в момент t."""
        due = []
        while self.calendar and self.calendar[0][0] == t:
            _, seq, ch = heapq.heappop(self.calendar)
            if ch.seq == seq and ch.tnext == t:
                due.append(ch)
        self.streams.ties.shuffle(due)             # одночасні події – у випадковому порядку
        for ch in due:
            ch.qs.end_of_service(ch, t)            # завершення обслуговування в каналі
        if self.create.tnext == t:
            self.create.arrival(t)                 # надходження – після завершень обслуговування
        self.update_blockings(t)

    def update_blockings(self, t: float) -> None:
        changed = True
        while changed:
            changed = False
            for route in self.routes:
                changed |= route.unblock(t)
            if self.robot_control:
                changed |= self.robot_control(t)

    def run(self, t_end: float, t_warmup: float = 0.0,
            observe_every: float | None = None,
            observer: Callable[["Model"], None] | None = None) -> "Model":
        self.t_end = t_end
        pending_reset = t_warmup > 0
        next_obs = observe_every if observe_every else INF
        self.update_blockings(0.0)
        self.log_state(0.0)
        while True:
            t_next = self._next_event_time()
            while True:
                t_service = min(t_warmup if pending_reset else INF, next_obs)
                if t_service > min(t_next, t_end):
                    break
                self._advance_stats(t_service)
                if pending_reset and t_service == t_warmup:
                    self.reset_stats()
                    pending_reset = False
                if t_service == next_obs:
                    observer(self)
                    next_obs += observe_every
            if t_next > t_end:
                self._advance_stats(t_end)
                break
            self._advance_stats(t_next)
            self.transition(self.tcurr)
            self.log_state(self.tcurr)
        return self

    def reset_stats(self) -> None:
        self.t_stats_start = self.tcurr
        self.completed = []
        for qs in self.stat_qs:
            qs.reset_stats()
        self.sum_LA = 0.0
        self.hist_LA: dict[int, float] = {}
        self.max_LA = self.A.occupied()
        self.robot_wait = {1: 0.0, 2: 0.0}
        self.jobs: dict[tuple[int, int], int] = defaultdict(int)
        self.empty_travel: dict[int, float] = defaultdict(float)
        self._created_at_reset = self.create.created

    def _advance_stats(self, t_to: float) -> None:
        dt = t_to - self.tcurr
        if dt > 0:
            for qs in self.stat_qs:
                qs.update_stats(self.tcurr, t_to)
            LA = self.A.occupied()
            self.sum_LA += LA * dt
            self.hist_LA[LA] = self.hist_LA.get(LA, 0.0) + dt
            self.max_LA = max(self.max_LA, LA)
            for i in (1, 2):
                if self.T[i].n[BLOCKED]:
                    self.robot_wait[i] += dt
        self.tcurr = t_to

    @property
    def period(self) -> float:
        return self.tcurr - self.t_stats_start

    def sojourn_times(self) -> list[float]:
        return [p.t_done - p.t_arrival for p in self.completed]

    def results(self) -> dict:
        return collect_results(self)

    def robot_time(self, k: int) -> dict[str, float]:
        """Тривалість станів пристроїв робота k: обслуговування деталі, блокування, затримка звільнення."""
        out = dict.fromkeys(("grab", "loaded_move", "wait_slot", "release", "empty_move"), 0.0)
        for ch in self.robot_channels[k]:
            acc = ch.qs.time[ch.index]
            if ch.qs is getattr(self, "X", None):
                out["empty_move"] += acc["busy_h"] + acc["busy"]
                continue
            out["grab"] += acc["busy_h"]
            out["loaded_move"] += acc["busy"]
            out["wait_slot"] += acc["blocked"]
            out["release"] += acc["post_h"]
            out["empty_move"] += acc["post"]          # варіант 1: повернення до точки завантаження
        return out

    def log(self, t: float, who: str, what: str) -> None:
        if self.trace and t <= self.trace_until:
            self.trace_lines.append(f"{t:10.3f}  {who:<7} {what}")

    def log_state(self, t: float) -> None:
        if self.trace and t <= self.trace_until:
            parts = [f"A={self.A.occupied()}", self.M1.state_str(), self.M2.state_str(),
                     " ".join(self.T[S].state_str() for S in (1, 2, 3))]
            if not self.dedicated:
                parts.append(self.X.state_str())
                parts.append(" ".join(f"{r}@{r.pos}" for r in self.robots))
            self.trace_lines.append(f"{'':10}  {'стан':<7} " + " ".join(parts))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else math.nan


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return math.nan
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _quantile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return math.nan
    pos = q * (len(sorted_xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def _time_quantile(dist: dict[int, float], q: float) -> float:
    acc = 0.0
    for value, share in dist.items():
        acc += share
        if acc >= q - 1e-12:
            return value
    return max(dist) if dist else math.nan


def collect_results(m: Model) -> dict:
    """Вихідні змінні одного прогону (за період після перехідного)."""
    T = m.period
    soj = sorted(m.sojourn_times())
    parts = m.completed
    hist = {k: v / T for k, v in sorted(m.hist_LA.items())} if T > 0 else {}
    res: dict = {
        "strategy": m.params.strategy,
        "seed": m.seed,
        "t_end": m.tcurr,
        "t_warmup": m.t_stats_start,
        "arrived": m.create.created - m._created_at_reset,
        "completed": len(parts),
        "wip_end": m.wip(),
        "throughput_per_hour": len(parts) / T * 3600 if T > 0 else math.nan,
        "sojourn_mean": _mean(soj),
        "sojourn_std": _std(soj),
        "sojourn_min": soj[0] if soj else math.nan,
        "sojourn_p50": _quantile(soj, 0.5),
        "sojourn_p90": _quantile(soj, 0.9),
        "sojourn_p95": _quantile(soj, 0.95),
        "sojourn_max": soj[-1] if soj else math.nan,
        "wait_route1_mean": _mean([p.t_grab[1] - p.t_ready[1] for p in parts]),
        "wait_route2_mean": _mean([p.t_grab[2] - p.t_ready[2] for p in parts]),
        "wait_route3_mean": _mean([p.t_grab[3] - p.t_ready[3] for p in parts]),
        "buffer_max": m.max_LA,
        "buffer_mean": m.sum_LA / T if T > 0 else math.nan,
        "buffer_end": m.A.occupied(),
        "buffer_q95": _time_quantile(hist, 0.95),
        "buffer_q99": _time_quantile(hist, 0.99),
    }
    for i, mach in ((1, m.M1), (2, m.M2)):
        res[f"M{i}_occupied"] = 1.0 - mach.share(T, ("free",))
        res[f"M{i}_processing"] = mach.share(T, ("busy",))
        res[f"M{i}_blocked"] = mach.share(T, ("blocked",))
        res[f"M{i}_handling"] = mach.share(T, ("busy_h", "post_h", "post"))
        res[f"M{i}_robot_waiting"] = m.robot_wait[i] / T if T > 0 else math.nan
    for S in (1, 2, 3):
        # холостий хід на одне перевезення: у варіанті 1 – повернення робота своїм шляхом
        n = sum(v for (k, s), v in m.jobs.items() if s == S)
        travel = m.travel[S] * n if m.dedicated else m.empty_travel[S]
        res[f"empty_route{S}_mean"] = travel / n if n else math.nan
    utils = []
    for k in sorted(m.robot_channels):
        phases = m.robot_time(k)
        u = sum(phases.values()) / T if T > 0 else math.nan
        utils.append(u)
        res[f"robot{k}_util"] = u
        for name, v in phases.items():
            res[f"robot{k}_{name}"] = v / T if T > 0 else math.nan
        for S in (1, 2, 3):
            res[f"robot{k}_jobs_route{S}"] = m.jobs.get((k, S), 0)
    res["robots_util_mean"] = _mean(utils)
    return res


def simulate(params: Params, seed: int, t_end: float, t_warmup: float = 0.0) -> tuple[dict, list[float]]:
    """Один прогін: (вихідні змінні, масив часів проходження деталей)."""
    m = Model(params, seed=seed).run(t_end, t_warmup)
    return m.results(), m.sojourn_times()
