from __future__ import annotations

import math
import random
from dataclasses import dataclass


class Stream:

    def __init__(self, seed: int | str):
        self._rnd = random.Random(seed)
        self._spare_normal: float | None = None

    def zeta(self) -> float:
        """Рівномірна величина на (0, 1]."""
        return 1.0 - self._rnd.random()

    def exponential(self, mean: float) -> float:
        return -mean * math.log(self.zeta())

    def uniform(self, center: float, half_width: float) -> float:
        return center - half_width + 2.0 * half_width * self._rnd.random()

    def standard_normal(self) -> float:
        """Полярний метод Марсальї–Брея: генерує пару, другу зберігає."""
        if self._spare_normal is not None:
            value, self._spare_normal = self._spare_normal, None
            return value
        while True:
            a1 = 2.0 * self._rnd.random() - 1.0
            a2 = 2.0 * self._rnd.random() - 1.0
            s = a1 * a1 + a2 * a2
            if 0.0 < s <= 1.0:
                break
        beta = math.sqrt(-2.0 * math.log(s) / s)
        self._spare_normal = a2 * beta
        return a1 * beta

    def normal(self, mean: float, std: float) -> float:
        return mean + std * self.standard_normal()

    def shuffle(self, items: list) -> None:
        self._rnd.shuffle(items)

    def random(self) -> float:
        return self._rnd.random()

    def choice(self, items: list):
        return items[0] if len(items) == 1 else self._rnd.choice(items)


@dataclass(frozen=True)
class Dist:
    kind: str
    a: float
    b: float = 0.0

    def sample(self, stream: Stream) -> float:
        if self.kind == "const":
            return self.a
        if self.kind == "exp":
            return stream.exponential(self.a)
        if self.kind == "unif":
            return stream.uniform(self.a, self.b)
        if self.kind == "norm":
            # усічений нормальний закон: час обробки не може бути від'ємним
            while True:
                value = stream.normal(self.a, self.b)
                if value > 0.0:
                    return value
        raise ValueError(f"Невідомий закон розподілу: {self.kind}")

    @property
    def mean(self) -> float:
        return self.a

    def __str__(self) -> str:
        if self.kind == "const":
            return f"{self.a:g}"
        if self.kind == "exp":
            return f"exp({self.a:g})"
        return f"{self.kind}({self.a:g},{self.b:g})"


def const(d: float) -> Dist:
    return Dist("const", d)


def exp(mean: float) -> Dist:
    return Dist("exp", mean)


def unif(center: float, half_width: float) -> Dist:
    return Dist("unif", center, half_width)


def norm(mean: float, std: float) -> Dist:
    return Dist("norm", mean, std)


STREAM_NAMES = ("arrivals", "m1", "m2", "handling", "ties")


class RandomStreams:
    def __init__(self, seed: int):
        self.seed = seed
        for name in STREAM_NAMES:
            setattr(self, name, Stream(f"{seed}:{name}"))

    arrivals: Stream
    m1: Stream
    m2: Stream
    handling: Stream
    ties: Stream
