from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .model import Model

INF = math.inf
FREE, BUSY, BLOCKED = 0, 1, 2


@dataclass(eq=False)
class Part:

    id: int
    t_arrival: float
    t_ready: dict[int, float] = field(default_factory=dict)  # шлях -> момент готовності до перевезення
    t_grab: dict[int, float] = field(default_factory=dict)   # шлях -> початок захоплення
    t_done: float = INF

    def __str__(self) -> str:
        return f"деталь #{self.id}"


@dataclass(eq=False)
class Robot:

    id: int
    pos: str
    target: str = ""

    def __str__(self) -> str:
        return f"R{self.id}"


class Channel:
    __slots__ = ("qs", "index", "K", "tnext", "request", "t_from", "t_split", "seq")

    def __init__(self, qs: "QS", index: int):
        self.qs = qs
        self.index = index
        self.K = FREE
        self.tnext = INF
        self.request = None
        self.t_from = 0.0     # початок поточного стану
        self.t_split = 0.0    # кінець захоплення або вивільнення в поточному стані
        self.seq = 0          # номер запису в календарі подій

    @property
    def name(self) -> str:
        qs = self.qs
        if qs.names:
            return qs.names[self.index]
        return qs.name if len(qs.channels) == 1 and not qs.unlimited else f"{qs.name}.{self.index + 1}"


# категорії часу каналу для статистики
CATS = ("free", "busy_h", "busy", "post_h", "post", "blocked")


class QS:
    def __init__(self, model: "Model", name: str, m: int | None,
                 service: Callable, release: Callable | None = None,
                 on_done: Callable | None = None, on_free: Callable | None = None,
                 names: list[str] | None = None):
        self.model = model
        self.name = name
        self.names = names
        self.unlimited = m is None
        self.channels: list[Channel] = [] if m is None else [Channel(self, j) for j in range(m)]
        self._spare: list[Channel] = []
        self.service = service
        self.release = release or (lambda req, h: 0.0)
        self.on_done = on_done
        self.on_free = on_free
        self.route: Route | None = None
        self.n = [len(self.channels), 0, 0]   # кількість каналів у станах 0, 1, 2
        self.requests = 0                     # кількість запитів у СМО
        self.reset_stats()

    # --- стан
    def set_state(self, ch: Channel, K: int, t: float) -> None:
        self.n[ch.K] -= 1
        self.n[K] += 1
        ch.K = K
        ch.t_from = t

    def free_channel(self) -> Channel | None:
        if self.unlimited:
            if self._spare:
                return self._spare.pop()
            ch = Channel(self, len(self.channels))
            self.channels.append(ch)
            self.n[FREE] += 1
            return ch
        return next((ch for ch in self.channels if ch.K == FREE), None)

    def occupied(self) -> int:
        return self.n[BUSY] + self.n[BLOCKED]

    # --- надходження запиту в пристрій
    def accept(self, req, t: float, h: float = 0.0, ch: Channel | None = None) -> Channel:
        ch = ch or self.free_channel()
        ch.request = req
        self.requests += 1
        self.set_state(ch, BUSY, t)
        d = self.service(req, h)
        ch.t_split = t + h
        if d > 0:
            self.model.schedule(ch, t + d)
            if isinstance(req, Part):
                self.model.log(t, ch.name, f"{req}: обслуговування до {t + d:.3f}")
        else:                              # нульова затримка: запит одразу намагається вийти
            ch.tnext = t
            self.end_of_service(ch, t)
        return ch

    # --- подія «завершення обслуговування в каналі»
    def end_of_service(self, ch: Channel, t: float) -> None:
        ch.tnext = INF
        if ch.request is None:             # кінець затримки звільнення: пристрій вільний
            self._vacated(ch, t, delayed=True)
            return
        if self.on_done:
            self.on_done(ch.request, t)
        route = self.route
        if route.blocked():                # B(S, t) = 1: запит лишається в пристрої
            self.set_state(ch, BLOCKED, t)
            route.waiting.append(ch)
            if isinstance(ch.request, Part):
                self.model.log(t, ch.name, f"{ch.request}: маршрут {route.name} заблоковано ({route.label})")
        else:
            route.transfer(ch, t)

    def vacate(self, ch: Channel, t: float, h: float) -> None:
        req = ch.request
        ch.request = None
        self.requests -= 1
        delay = self.release(req, h)
        if delay > 0:
            self.set_state(ch, BUSY, t)
            ch.t_split = t + h
            self.model.schedule(ch, t + delay)
            self.model.log(t, ch.name, f"затримка звільнення до {t + delay:.3f}")
        else:
            self._vacated(ch, t)

    def _vacated(self, ch: Channel, t: float, delayed: bool = False) -> None:
        if delayed and not self.on_free:
            self.model.log(t, ch.name, "пристрій вільний")
        self.set_state(ch, FREE, t)
        if self.unlimited:
            self._spare.append(ch)
        if self.on_free:
            self.on_free(ch, t)

    # --- статистика: тривалість станів кожного каналу на [t1; t2]
    def reset_stats(self) -> None:
        self.time = {ch.index: dict.fromkeys(CATS, 0.0) for ch in self.channels} if not self.unlimited else {}

    def update_stats(self, t1: float, t2: float) -> None:
        if self.unlimited:
            return
        for ch in self.channels:
            acc = self.time[ch.index]
            dt = t2 - t1
            if ch.K == FREE:
                acc["free"] += dt
            elif ch.K == BLOCKED:
                acc["blocked"] += dt
            else:
                h = max(0.0, min(t2, ch.t_split) - t1)     # частина інтервалу до кінця захоплення/вивільнення
                if ch.request is not None:
                    acc["busy_h"] += h
                    acc["busy"] += dt - h
                else:
                    acc["post_h"] += h
                    acc["post"] += dt - h

    def share(self, period: float, cats, channels=None) -> float:
        """Частка часу (на один канал), яку канали перебували в категоріях cats."""
        chs = channels if channels is not None else [ch.index for ch in self.channels]
        if period <= 0 or not chs:
            return math.nan
        return sum(self.time[j][c] for j in chs for c in cats) / (period * len(chs))

    def state_str(self) -> str:
        return f"{self.name}[{''.join(str(ch.K) for ch in self.channels)}]"


class Route:
    def __init__(self, model: "Model", src, dst, blocked: Callable[[], bool] | None = None,
                 label: str = "", handling: str = "", select: Callable | None = None,
                 on_transfer: Callable | None = None):
        self.model = model
        self.src, self.dst = src, dst
        self.blocked = blocked or (lambda: False)
        self.label = label
        self.handling = handling      # «захоплення» або «вивільнення», якщо перехід їх супроводжує
        self.select = select
        self.on_transfer = on_transfer
        self.waiting: deque[Channel] = deque()   # заблоковані канали src у порядку блокування
        self.name = f"{src.name} -> {dst.name}"
        src.route = self

    def transfer(self, ch: Channel, t: float) -> None:
        req = ch.request
        h = self.model.handling_time() if self.handling else 0.0
        dst_ch = self.select(req) if self.select else None
        if isinstance(req, Part) and self.dst is not self.model.exit:
            self.model.log(t, ch.name, f"{req}: перехід {self.name}" + (f", {self.handling} {h:.3f}" if h else ""))
        if self.on_transfer:
            self.on_transfer(req, ch, dst_ch, t)
        self.src.vacate(ch, t, h)
        self.dst.accept(req, t, h, dst_ch)

    def unblock(self, t: float) -> bool:
        moved = False
        while self.waiting and not self.blocked():
            ch = self.waiting.popleft()
            self.model.log(t, ch.name, f"{ch.request}: маршрут {self.name} відкрито ({self.label} = 0)")
            self.transfer(ch, t)
            moved = True
        return moved


class Create:
    def __init__(self, model: "Model", name: str, dist, stream):
        self.model = model
        self.name = name
        self.dist = dist
        self.stream = stream
        self.tnext = dist.sample(stream)
        self.created = 0
        self.route: Route | None = None

    def arrival(self, t: float) -> None:
        self.created += 1
        part = Part(id=self.created, t_arrival=t)
        self.model.log(t, self.name, f"надходження деталі #{part.id}")
        self.route.dst.accept(part, t)
        self.tnext = t + self.dist.sample(self.stream)


class Exit:
    name = "вихід"

    def __init__(self, model: "Model"):
        self.model = model

    def accept(self, part: Part, t: float, h: float = 0.0, ch=None) -> None:
        self.model.record_completion(part, t)
