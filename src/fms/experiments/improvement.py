from __future__ import annotations

from ..params import DEDICATED, NEAREST, Params
from .common import replicate, summarize

SCENARIOS = [
    ("базова система", {}, {}),
    ("4 місця на M2", {"m2_capacity": 4}, {}),
    ("4-й робот на маршруті 1 (A → M1)", {"n_robots": 4}, {"dedicated_routes": (1, 2, 3, 1)}),
    ("4-й робот на маршруті 2 (M1 → M2)", {"n_robots": 4}, {"dedicated_routes": (1, 2, 3, 2)}),
    ("4-й робот на маршруті 3 (M2 → W)", {"n_robots": 4}, {"dedicated_routes": (1, 2, 3, 3)}),
    # у варіанті 1 4-й робот у комбінації – на маршруті 2, бо окремо він дає найбільший приріст
    ("4 місця на M2 і 4-й робот на маршруті 2", {"m2_capacity": 4, "n_robots": 4},
     {"dedicated_routes": (1, 2, 3, 2)}),
]

RESPONSES = ("throughput_per_hour", "sojourn_mean", "buffer_max", "buffer_mean", "wip_end",
             "M1_occupied", "M2_occupied", "M2_blocked", "robots_util_mean")


def run(reps: int, t_end: float, t_warmup: float, jobs: int, base: Params | None = None) -> list[dict]:
    base = base or Params()
    rows = []
    seen: set[tuple] = set()
    for name, change, dedicated_change in SCENARIOS:
        for strategy in (DEDICATED, NEAREST):
            params = base.with_(strategy=strategy, **change,
                                **(dedicated_change if strategy == DEDICATED else {}))
            # у варіанті 2 маршрут 4-го робота не задається: три сценарії з ним однакові
            key = (strategy, params.m2_capacity, params.n_robots, params.dedicated_routes)
            if key in seen:
                continue
            seen.add(key)
            results, _ = replicate(params, reps, t_end, t_warmup, jobs=jobs)
            input_per_hour = 3600 / base.arrival.mean
            label = name if strategy == DEDICATED else name.split(" на маршруті")[0]
            row = {"scenario": label,
                   "strategy": strategy, "m2_capacity": params.m2_capacity, "n_robots": params.n_robots,
                   "arrival_mean": base.arrival.mean, "input_per_hour": input_per_hour}
            for k in RESPONSES:
                st = summarize([r[k] for r in results])
                row[k] = st["mean"]
                row[k + "_ci"] = st["ci"]
            row["output_input_ratio"] = row["throughput_per_hour"] / input_per_hour
            row["stable"] = row["output_input_ratio"] >= 0.98 and row["wip_end"] < 30
            rows.append(row)
    return rows
