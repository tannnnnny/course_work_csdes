"""Верифікація моделі: варіювання вхідних змінних і перевірка логічності реакції виходів."""

from __future__ import annotations

from ..params import Params
from ..rng import const, exp, norm, unif
from .common import replicate, summarize

# (назва, зміни параметрів, очікувана реакція)
SCENARIOS = [
    ("базовий", {}, "—"),
    ("рідше надходження exp(80)", {"arrival": exp(80.0)},
     "пропускна ≈ 45/год, черга A мала, завантаження ↓"),
    ("частіше надходження exp(30)", {"arrival": exp(30.0)},
     "перевантаження: черга A росте, пропускна обмежена M2"),
    ("швидший M2 exp(50)", {"m2_time": exp(50.0)},
     "M2 не вузьке місце: пропускна ≈ 90/год, час проходження ↓"),
    ("повільніший M1 norm(120,10)", {"m1_time": norm(120.0, 10.0)},
     "M1 стає вузьким місцем, зайнятість M1 ≈ 1"),
    ("M2 на 6 місць", {"m2_capacity": 6},
     "пропускна ≈ 90/год, блокування M1 зникає"),
    ("швидке захоплення unif(2,0.5)", {"handling": unif(2.0, 0.5)},
     "завантаження роботів ↓, пропускна ↑"),
    ("переїзди ×3", "travel3",
     "завантаження роботів ↑, час проходження ↑"),
    ("детерміновані часи, exp→const(300)", {"arrival": const(300.0), "m1_time": const(60.0),
                                            "m2_time": const(100.0), "handling": const(8.0)},
     "перша деталь 226 с, решта 244 с (226 + холості ходи 6+7+5)"),
]

COLUMNS = ["throughput_per_hour", "sojourn_mean", "buffer_max", "buffer_mean",
           "M1_occupied", "M2_occupied", "M2_processing", "robots_util_mean", "wip_end"]
INPUT_COLUMNS = ["in_arrival", "in_m1", "in_m2", "in_handling", "in_m2_capacity", "in_travel"]


def inputs_of(p: Params, travel: float) -> dict:
    """Значення вхідних змінних прогону для таблиці верифікації."""
    return {"in_arrival": str(p.arrival), "in_m1": str(p.m1_time), "in_m2": str(p.m2_time),
            "in_handling": str(p.handling), "in_m2_capacity": p.m2_capacity, "in_travel": f"×{travel:g}"}


def run(strategy: str, reps: int, t_end: float, t_warmup: float, jobs: int) -> list[dict]:
    rows = []
    for name, change, expected in SCENARIOS:
        p = Params(strategy=strategy)
        travel = 3.0 if change == "travel3" else 1.0
        p = p.scaled_travel(travel) if change == "travel3" else p.with_(**change)
        results, _ = replicate(p, reps, t_end, t_warmup, jobs=jobs)
        row = {"scenario": name, "strategy": strategy, **inputs_of(p, travel), "expected": expected}
        for c in COLUMNS:
            row[c] = summarize([r[c] for r in results])["mean"]
        rows.append(row)
    return rows
