"""Елементи мережі масового обслуговування роботизованої ГВС."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model, Route

INF = math.inf


@dataclass
class Part:
    """Вимога мережі – деталь. Зберігає моменти подій для статистики часу проходження."""

    id: int
    t_arrival: float
    t_grab: dict[int, float] = field(default_factory=dict)   # маршрут -> початок захоплення
    t_ready: dict[int, float] = field(default_factory=dict)  # маршрут -> момент готовності до перевезення
    t_done: float = INF


@dataclass
class Task:
    """Заявка на перевезення: деталь чекає робота на початку маршруту."""

    route: int
    part: Part
    t_ready: float
    channel: "MachineChannel | None" = None  # канал верстата-джерела (для маршрутів 2 і 3)


class Element:
    def __init__(self, name: str, model: "Model"):
        self.name = name
        self.model = model

    def next_time(self) -> float:
        return INF

    def out_act(self, t: float) -> None:
        pass

    def update_stats(self, dt: float) -> None:
        pass

    def reset_stats(self) -> None:
        pass

    def state_str(self) -> str:
        return ""


class Create(Element):
    """Зовнішнє надходження деталей на пункт прибуття."""

    def __init__(self, name: str, model: "Model", target: "Queue"):
        super().__init__(name, model)
        self.target = target
        self.tnext = model.params.arrival.sample(model.streams.arrivals)
        self.created = 0

    def next_time(self) -> float:
        return self.tnext

    def out_act(self, t: float) -> None:
        self.created += 1
        part = Part(id=self.created, t_arrival=t)
        self.model.log(t, self.name, f"надходження деталі #{part.id}")
        self.target.put(part, t)
        self.tnext = t + self.model.params.arrival.sample(self.model.streams.arrivals)


class Queue(Element):
    """Місце зберігання деталей на ділянці прибуття."""

    def __init__(self, name: str, model: "Model", route_out: int):
        super().__init__(name, model)
        self.route_out = route_out
        self.length = 0
        self.max_length = 0
        self._sum_length = 0.0
        self._hist: dict[int, float] = {}  # довжина -> сумарний час перебування в цьому стані

    def put(self, part: Part, t: float) -> None:
        self.length += 1
        self.max_length = max(self.max_length, self.length)
        part.t_ready[self.route_out] = t
        self.model.add_task(Task(route=self.route_out, part=part, t_ready=t))

    def grab_done(self, t: float) -> None:
        self.length -= 1

    def update_stats(self, dt: float) -> None:
        self._sum_length += self.length * dt
        self._hist[self.length] = self._hist.get(self.length, 0.0) + dt

    def reset_stats(self) -> None:
        self._sum_length = 0.0
        self._hist = {}
        self.max_length = self.length

    def mean_length(self, period: float) -> float:
        return self._sum_length / period if period > 0 else 0.0

    def length_distribution(self, period: float) -> dict[int, float]:
        return {k: v / period for k, v in sorted(self._hist.items())} if period > 0 else {}

    def state_str(self) -> str:
        return f"A={self.length}"


FREE, LOADING, BUSY, BLOCKED, UNLOADING = (
    "free", "loading", "busy", "blocked", "unloading",
)
MACHINE_STATES = (FREE, LOADING, BUSY, BLOCKED, UNLOADING)
_STATE_CHAR = {FREE: ".", LOADING: "l", BUSY: "B", BLOCKED: "x", UNLOADING: "u"}


@dataclass
class MachineChannel:
    index: int
    state: str = FREE
    part: Part | None = None
    tnext: float = INF


class Machine(Element):
    def __init__(self, name: str, model: "Model", capacity: int, dist, stream, route_out: int):
        super().__init__(name, model)
        self.channels = [MachineChannel(i) for i in range(capacity)]
        self.dist = dist
        self.stream = stream
        self.route_out = route_out
        self.incoming = 0  # роботи, призначені на доставку сюди, але ще без місця (u_i)
        self.waiting_robots: deque = deque()  # роботи з деталлю, що чекають вільного місця
        self.processed = 0
        self._state_time = {s: 0.0 for s in MACHINE_STATES}
        self.robot_wait_time = 0.0  # час, коли біля верстата чекає робот з деталлю

    @property
    def capacity(self) -> int:
        return len(self.channels)

    def count(self, state: str) -> int:
        return sum(1 for ch in self.channels if ch.state == state)

    def _first(self, state: str) -> MachineChannel | None:
        return next((ch for ch in self.channels if ch.state == state), None)

    def next_time(self) -> float:
        return min(ch.tnext for ch in self.channels)

    # --- події обробки
    def out_act(self, t: float) -> None:
        due = [ch for ch in self.channels if ch.tnext == t]
        self.model.streams.ties.shuffle(due)
        for ch in due:
            ch.state = BLOCKED
            ch.tnext = INF
            self.processed += 1
            ch.part.t_ready[self.route_out] = t
            self.model.log(t, self.name, f"кінець обробки деталі #{ch.part.id} (місце {ch.index}) -> блокування")
            self.model.add_task(Task(route=self.route_out, part=ch.part, t_ready=t, channel=ch))

    # --- умова призначення робота на доставку сюди (частина умов B0, B2)
    def can_dispatch_to(self) -> bool:
        return self.incoming <= self.count(FREE)

    def on_dispatch(self) -> None:
        self.incoming += 1

    def take_for_release(self) -> MachineChannel | None:
        """Робот підвіз деталь: повертає місце для вивільнення або None, якщо треба чекати."""
        ch = self._first(FREE)
        if ch is None:
            return None
        self.incoming -= 1
        ch.state = LOADING
        return ch

    def release_done(self, ch: MachineChannel, part: Part, t: float) -> None:
        ch.state = BUSY
        ch.part = part
        ch.tnext = t + self.dist.sample(self.stream)
        self.model.log(t, self.name, f"початок обробки деталі #{part.id} (місце {ch.index}), кінець о {ch.tnext:.3f}")

    def start_grab(self, ch: MachineChannel) -> None:
        ch.state = UNLOADING

    def grab_done(self, ch: MachineChannel, t: float) -> None:
        ch.state = FREE
        ch.part = None
        if self.waiting_robots:
            robot = self.waiting_robots.popleft()
            slot = self.take_for_release()
            robot.transport.start_release(robot, slot, t)

    def update_stats(self, dt: float) -> None:
        for ch in self.channels:
            self._state_time[ch.state] += dt
        if self.waiting_robots:
            self.robot_wait_time += dt

    def reset_stats(self) -> None:
        self._state_time = {s: 0.0 for s in MACHINE_STATES}
        self.processed = 0
        self.robot_wait_time = 0.0

    def utilization(self, period: float) -> dict[str, float]:
        denom = period * self.capacity
        share = {s: v / denom for s, v in self._state_time.items()} if denom > 0 else {}
        occupied = 1.0 - share.get(FREE, 1.0)
        return {
            "occupied": occupied,
            "processing": share.get(BUSY, 0.0),
            "blocked": share.get(BLOCKED, 0.0),
            "handling": share.get(LOADING, 0.0) + share.get(UNLOADING, 0.0),
        }

    def state_str(self) -> str:
        return f"{self.name}[{''.join(_STATE_CHAR[ch.state] for ch in self.channels)}]"


class Warehouse(Element):

    def accept(self, part: Part, t: float) -> None:
        part.t_done = t
        self.model.log(t, self.name, f"деталь #{part.id} на складі, час проходження {t - part.t_arrival:.3f}")
        self.model.record_completion(part)


R_FREE, R_TO_PICKUP, R_GRAB, R_CARRY, R_WAIT, R_RELEASE = (
    "free", "empty_move", "grab", "loaded_move", "wait_slot", "release",
)
ROBOT_STATES = (R_FREE, R_TO_PICKUP, R_GRAB, R_CARRY, R_WAIT, R_RELEASE)


@dataclass(eq=False)
class Robot:
    id: int
    location: str
    allowed_routes: tuple[int, ...]
    transport: "Transport" = None
    state: str = R_FREE
    tnext: float = INF
    task: Task | None = None
    route: "Route | None" = None
    dest_slot: MachineChannel | None = None
    jobs: dict[int, int] = field(default_factory=dict)
    empty_travel: dict[int, float] = field(default_factory=dict)  # сумарний холостий хід за маршрутами
    state_time: dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in ROBOT_STATES})


class Transport(Element):
    def __init__(self, name: str, model: "Model", robots: list[Robot]):
        super().__init__(name, model)
        self.robots = robots
        for r in robots:
            r.transport = self

    def next_time(self) -> float:
        return min(r.tnext for r in self.robots)

    def _handling(self) -> float:
        return self.model.params.handling.sample(self.model.streams.handling)

    def assign(self, robot: Robot, task: Task, t: float) -> None:
        route = self.model.routes[task.route]
        robot.task = task
        robot.route = route
        robot.jobs[route.index] = robot.jobs.get(route.index, 0) + 1
        if isinstance(route.dest, Machine):
            route.dest.on_dispatch()
        travel = self.model.params.distance(robot.location, route.pickup)
        robot.empty_travel[route.index] = robot.empty_travel.get(route.index, 0.0) + travel
        self.model.log(
            t, f"Robot{robot.id}",
            f"призначено на маршрут {route.index} ({route.pickup}->{route.drop}) для деталі #{task.part.id}, "
            f"холостий хід {robot.location}->{route.pickup} {travel:g}",
        )
        if travel > 0:
            robot.state = R_TO_PICKUP
            robot.tnext = t + travel
        else:
            self._start_grab(robot, t)

    def _start_grab(self, robot: Robot, t: float) -> None:
        robot.location = robot.route.pickup
        robot.state = R_GRAB
        robot.tnext = t + self._handling()
        robot.task.part.t_grab[robot.route.index] = t
        if robot.task.channel is not None:
            robot.route.source.start_grab(robot.task.channel)
        self.model.log(t, f"Robot{robot.id}", f"початок захоплення деталі #{robot.task.part.id} у {robot.location}")

    def _start_carry(self, robot: Robot, t: float) -> None:
        route = robot.route
        # деталь знята з місця-джерела
        if robot.task.channel is not None:
            route.source.grab_done(robot.task.channel, t)
        else:
            route.source.grab_done(t)
        robot.state = R_CARRY
        robot.tnext = t + self.model.params.distance(route.pickup, route.drop)
        self.model.log(t, f"Robot{robot.id}", f"деталь #{robot.task.part.id} захоплено, хід {route.pickup}->{route.drop}")

    def _arrive(self, robot: Robot, t: float) -> None:
        route = robot.route
        robot.location = route.drop
        if isinstance(route.dest, Machine):
            slot = route.dest.take_for_release()
            if slot is None:
                # маршрут заблоковано: робот тримає деталь і чекає вільного місця
                robot.state = R_WAIT
                robot.tnext = INF
                route.dest.waiting_robots.append(robot)
                self.model.log(t, f"Robot{robot.id}", f"на {route.drop} немає місця, очікування з деталлю #{robot.task.part.id}")
                return
            self.start_release(robot, slot, t)
        else:
            self.start_release(robot, None, t)

    def start_release(self, robot: Robot, slot: MachineChannel | None, t: float) -> None:
        robot.dest_slot = slot
        robot.state = R_RELEASE
        robot.tnext = t + self._handling()
        self.model.log(t, f"Robot{robot.id}", f"початок вивільнення деталі #{robot.task.part.id} у {robot.location}")

    def _finish(self, robot: Robot, t: float) -> None:
        route, part = robot.route, robot.task.part
        if isinstance(route.dest, Machine):
            route.dest.release_done(robot.dest_slot, part, t)
        else:
            route.dest.accept(part, t)
        self.model.log(t, f"Robot{robot.id}", f"вивільнив деталь #{part.id}, вільний у {robot.location}")
        robot.state = R_FREE
        robot.tnext = INF
        robot.task = robot.route = robot.dest_slot = None

    def out_act(self, t: float) -> None:
        due = [r for r in self.robots if r.tnext == t]
        self.model.streams.ties.shuffle(due)
        for robot in due:
            if robot.state == R_TO_PICKUP:
                self._start_grab(robot, t)
            elif robot.state == R_GRAB:
                self._start_carry(robot, t)
            elif robot.state == R_CARRY:
                self._arrive(robot, t)
            elif robot.state == R_RELEASE:
                self._finish(robot, t)

    def free_robots(self) -> list[Robot]:
        return [r for r in self.robots if r.state == R_FREE]

    def update_stats(self, dt: float) -> None:
        for r in self.robots:
            r.state_time[r.state] += dt

    def reset_stats(self) -> None:
        for r in self.robots:
            r.state_time = {s: 0.0 for s in ROBOT_STATES}
            r.jobs = {}
            r.empty_travel = {}

    def state_str(self) -> str:
        return " ".join(f"R{r.id}:{r.state}@{r.location}" for r in self.robots)
