"""Визначення тривалості перехідного періоду (формули 3.1–3.2 методички)."""

from __future__ import annotations

import math

from ..model import Model
from ..params import Params

METRICS = ("throughput_per_hour", "sojourn_mean", "buffer_mean", "M2_occupied", "robots_util_mean")


def _snapshot(m: Model) -> dict:
    t = m.tcurr
    soj = [p.t_done - p.t_arrival for p in m.completed]
    busy = [sum(v for s, v in r.state_time.items() if s != "free") / t for r in m.robots]
    return {
        "t": t,
        "throughput_per_hour": m.completed_total / t * 3600,
        "sojourn_mean": sum(soj) / len(soj) if soj else math.nan,
        "buffer_mean": m.buffer.mean_length(t),
        "M2_occupied": m.m2.utilization(t)["occupied"],
        "robots_util_mean": sum(busy) / len(busy),
        # поточна довжина черги на площадці A і її максимум від початку прогону
        "buffer_now": m.buffer.length,
        "buffer_max": m.buffer.max_length,
    }


def observe(params: Params, seeds: list[int], t_end: float, step: float) -> dict[int, list[dict]]:
    series = {}
    for seed in seeds:
        snaps: list[dict] = []
        Model(params, seed=seed).run(t_end, observe_every=step, observer=lambda m: snaps.append(_snapshot(m)))
        series[seed] = snaps
    return series


def transient_time(series: dict[int, list[dict]], metric: str, eps: float) -> float:
    """Найменший момент T, після якого max|ȳ_i(t) − ȳ_j(t)| < ε для всіх t ≥ T (або nan)."""
    runs = list(series.values())
    n_points = min(len(r) for r in runs)
    T = math.nan
    for k in range(n_points - 1, -1, -1):
        vals = [r[k][metric] for r in runs]
        if any(math.isnan(v) for v in vals) or max(vals) - min(vals) >= eps:
            break
        T = runs[0][k]["t"]
    return T


def run(params: Params, n_runs: int, t_end: float, step: float, eps_rel: float):
    seeds = list(range(1, n_runs + 1))
    series = observe(params, seeds, t_end, step)
    rows = [{"strategy": params.strategy, "arrival_mean": params.arrival.mean, "seed": s, **snap}
            for s, snaps in series.items() for snap in snaps]
    summary = []
    for metric in METRICS:
        final = [snaps[-1][metric] for snaps in series.values()]
        level = sum(final) / len(final)
        eps = eps_rel * abs(level)
        summary.append({
            "strategy": params.strategy, "arrival_mean": params.arrival.mean, "metric": metric,
            "final_mean": level, "eps": eps, "T_transient": transient_time(series, metric, eps),
            # ознака відсутності стаціонарного режиму: середнє продовжує зростати
            "growth_last_half": _growth(series, metric),
        })
    return rows, summary


def _growth(series: dict[int, list[dict]], metric: str) -> float:
    """Відносна зміна середнього відгуку між серединою та кінцем прогону."""
    ratios = []
    for snaps in series.values():
        mid, last = snaps[len(snaps) // 2][metric], snaps[-1][metric]
        if mid and not math.isnan(mid):
            ratios.append(last / mid - 1.0)
    return sum(ratios) / len(ratios) if ratios else math.nan
