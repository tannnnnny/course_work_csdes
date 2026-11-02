from __future__ import annotations

from ..params import Params
from ..rng import exp
from .common import run_batch, summarize, variants

DEFAULT_ARRIVALS = (34, 36, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 55, 60, 70, 80)
RESPONSES = ("throughput_per_hour", "sojourn_mean", "buffer_max", "buffer_mean", "buffer_q99",
             "wip_end", "M1_occupied", "M2_occupied", "M2_processing", "M2_blocked", "robots_util_mean",
             "robot1_util", "robot2_util", "robot3_util")


def run(arrivals, strategy, reps: int, t_end: float, t_warmup: float, jobs: int,
        base: Params | None = None) -> list[dict]:
    base = base or Params()
    points = [(s, a) for s in variants(strategy) for a in arrivals]
    tasks = [(base.with_(strategy=s, arrival=exp(float(a))), seed, t_end, t_warmup, False)
             for s, a in points for seed in range(1, reps + 1)]
    out = run_batch(tasks, jobs)
    rows = []
    for i, (s, a) in enumerate(points):
        results = [r for r, _ in out[i * reps:(i + 1) * reps]]
        row = {"strategy": s, "arrival_mean": a, "input_per_hour": 3600 / a}
        for k in RESPONSES:
            st = summarize([r[k] for r in results])
            row[k] = st["mean"]
            row[k + "_ci"] = st["ci"]
        row["output_input_ratio"] = row["throughput_per_hour"] / row["input_per_hour"]
        row["stable"] = row["output_input_ratio"] >= 0.98 and row["wip_end"] < 30
        rows.append(row)
    return rows


def max_stable(rows: list[dict]) -> list[dict]:
    out = {}
    for r in rows:
        key = r["strategy"]
        cur = out.setdefault(key, {"strategy": key,
                                   "max_stable_input_per_hour": 0.0, "min_stable_arrival_mean": None,
                                   "max_throughput_per_hour": 0.0})
        cur["max_throughput_per_hour"] = max(cur["max_throughput_per_hour"], r["throughput_per_hour"])
        if r["stable"] and r["input_per_hour"] > cur["max_stable_input_per_hour"]:
            cur["max_stable_input_per_hour"] = r["input_per_hour"]
            cur["min_stable_arrival_mean"] = r["arrival_mean"]
    return list(out.values())
