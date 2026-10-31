"""Модель роботизованої ГВС та алгоритм імітації (алгоритм 3.1/3.2 методички)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable

from .elements import (
    INF, Create, Element, Machine, Part, Queue, Robot, Task, Transport, Warehouse,
)
from .params import DEDICATED, Params
from .rng import RandomStreams


@dataclass
class Route:
    index: int
    pickup: str
    drop: str
    source: Element
    dest: Element

    def blocked(self) -> bool:
        return isinstance(self.dest, Machine) and not self.dest.can_dispatch_to()


class Model:
    def __init__(self, params: Params, seed: int = 1, trace: bool = False,
                 trace_until: float = INF):
        self.params = params
        self.streams = RandomStreams(seed)
        self.seed = seed
        self.tcurr = 0.0
        self.trace = trace
        self.trace_until = trace_until
        self.trace_lines: list[str] = []

        self.tasks: dict[int, deque[Task]] = {1: deque(), 2: deque(), 3: deque()}
        self.completed: list[Part] = []
        self.completed_total = 0
        self.t_stats_start = 0.0

        p = params
        self.buffer = Queue("Arrival", self, route_out=1)
        self.create = Create("Create", self, self.buffer)
        self.m1 = Machine("M1", self, p.m1_capacity, p.m1_time, self.streams.m1, route_out=2)
        self.m2 = Machine("M2", self, p.m2_capacity, p.m2_time, self.streams.m2, route_out=3)
        self.warehouse = Warehouse("Warehouse", self)
        self.routes = {
            1: Route(1, "A", "M1", self.buffer, self.m1),
            2: Route(2, "M1", "M2", self.m1, self.m2),
            3: Route(3, "M2", "W", self.m2, self.warehouse),
        }

        self.robots: list[Robot] = []
        for k, allowed in enumerate(p.robot_routes()):
            home = self.routes[allowed[0]].pickup if len(allowed) == 1 else ("A", "M1", "M2")[k % 3]
            self.robots.append(Robot(id=k + 1, location=home, allowed_routes=allowed))

        if p.strategy == DEDICATED:
            self.transports = [Transport(f"T{r.id}", self, [r]) for r in self.robots]
        else:
            self.transports = [Transport("T", self, self.robots)]

        self.event_elements: list[Element] = [self.m1, self.m2, *self.transports]
        self.stat_elements: list[Element] = [self.buffer, self.m1, self.m2, *self.transports]

    def log(self, t: float, who: str, what: str) -> None:
        if self.trace and t <= self.trace_until:
            self.trace_lines.append(f"{t:11.3f}  {who:<9} {what}")

    def log_state(self, t: float) -> None:
        if self.trace and t <= self.trace_until:
            parts = [self.buffer.state_str(), self.m1.state_str(), self.m2.state_str(),
                     " ".join(tr.state_str() for tr in self.transports)]
            self.trace_lines.append(f"{'':11}  {'стан':<9} " + " | ".join(parts))

    def add_task(self, task: Task) -> None:
        self.tasks[task.route].append(task)

    def record_completion(self, part: Part) -> None:
        self.completed_total += 1
        if self.tcurr >= self.t_stats_start:
            self.completed.append(part)

    def wip(self) -> int:
        return self.create.created - self.completed_total

    def dispatch(self, t: float) -> None:
        rank = self.params.route_rank
        while True:
            best = None
            for tr in self.transports:
                for robot in tr.free_robots():
                    for r in robot.allowed_routes:
                        queue = self.tasks[r]
                        if not queue:
                            continue
                        route = self.routes[r]
                        if route.blocked():
                            continue
                        key = (self.params.distance(robot.location, route.pickup), rank[r - 1],
                               queue[0].t_ready, self.streams.ties.random())
                        if best is None or key < best[0]:
                            best = (key, tr, robot, r)
            if best is None:
                return
            _, tr, robot, r = best
            tr.assign(robot, self.tasks[r].popleft(), t)

    def _next_event_time(self) -> float:
        return min(self.create.next_time(), *(e.next_time() for e in self.event_elements))

    def _advance_stats(self, t_to: float) -> None:
        dt = t_to - self.tcurr
        if dt > 0:
            for e in self.stat_elements:
                e.update_stats(dt)
        self.tcurr = t_to

    def reset_stats(self) -> None:
        self.t_stats_start = self.tcurr
        self.completed = []
        for e in self.stat_elements:
            e.reset_stats()
        self._created_at_reset = self.create.created

    def run(self, t_end: float, t_warmup: float = 0.0,
            observe_every: float | None = None,
            observer: Callable[["Model"], None] | None = None) -> "Model":
        self.t_end = t_end
        self._created_at_reset = 0
        pending_reset = t_warmup > 0
        next_obs = observe_every if observe_every else INF
        self.dispatch(0.0)
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
            t = self.tcurr
            due = [e for e in self.event_elements if e.next_time() == t]
            self.streams.ties.shuffle(due)
            for e in due:
                e.out_act(t)
            if self.create.next_time() == t:
                self.create.out_act(t)
            self.dispatch(t)
            self.log_state(t)
        return self

    @property
    def period(self) -> float:
        return self.tcurr - self.t_stats_start

    def sojourn_times(self) -> list[float]:
        return [p.t_done - p.t_arrival for p in self.completed]

    def results(self) -> dict:
        return collect_results(self)


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
    T = m.period
    soj = sorted(m.sojourn_times())
    parts = m.completed
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
        # очікування робота на початку кожного маршруту
        "wait_route1_mean": _mean([p.t_grab[1] - p.t_ready[1] for p in parts]),
        "wait_route2_mean": _mean([p.t_grab[2] - p.t_ready[2] for p in parts]),
        "wait_route3_mean": _mean([p.t_grab[3] - p.t_ready[3] for p in parts]),
        "buffer_max": m.buffer.max_length,
        "buffer_mean": m.buffer.mean_length(T),
        "buffer_end": m.buffer.length,
        # місткість, якої достатньо у 95% / 99% часу (за розподілом довжини черги)
        "buffer_q95": _time_quantile(m.buffer.length_distribution(T), 0.95),
        "buffer_q99": _time_quantile(m.buffer.length_distribution(T), 0.99),
    }
    for mach in (m.m1, m.m2):
        for k, v in mach.utilization(T).items():
            res[f"{mach.name}_{k}"] = v
        res[f"{mach.name}_robot_waiting"] = mach.robot_wait_time / T if T > 0 else math.nan
    # середній холостий хід робота на одне перевезення кожного маршруту
    for route in (1, 2, 3):
        n = sum(r.jobs.get(route, 0) for r in m.robots)
        travel = sum(r.empty_travel.get(route, 0.0) for r in m.robots)
        res[f"empty_route{route}_mean"] = travel / n if n else math.nan
    utils = []
    for r in m.robots:
        busy = sum(v for s, v in r.state_time.items() if s != "free")
        u = busy / T if T > 0 else math.nan
        utils.append(u)
        res[f"robot{r.id}_util"] = u
        for s, v in r.state_time.items():
            if s != "free":
                res[f"robot{r.id}_{s}"] = v / T if T > 0 else math.nan
        for route in (1, 2, 3):
            res[f"robot{r.id}_jobs_route{route}"] = r.jobs.get(route, 0)
    res["robots_util_mean"] = _mean(utils)
    return res


def simulate(params: Params, seed: int, t_end: float, t_warmup: float = 0.0) -> tuple[dict, list[float]]:
    """Один прогін: (вихідні змінні, масив часів проходження деталей)."""
    m = Model(params, seed=seed).run(t_end, t_warmup)
    return m.results(), m.sojourn_times()
