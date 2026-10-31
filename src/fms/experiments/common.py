"""Спільні засоби для експериментів: серії прогонів, збереження результатів, статистика."""

from __future__ import annotations

import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

from ..model import simulate
from ..params import DEDICATED, NEAREST, Params

# основні відгуки моделі
KEY_OUTPUTS = (
    "throughput_per_hour", "sojourn_mean", "buffer_max", "buffer_mean",
    "M1_occupied", "M1_processing", "M1_blocked", "M2_occupied", "M2_processing", "M2_blocked",
    "robot1_util", "robot2_util", "robot3_util", "robots_util_mean",
    "M1_robot_waiting", "M2_robot_waiting", "empty_route1_mean", "empty_route2_mean", "empty_route3_mean",
)

STRATEGY_LABEL = {DEDICATED: "1) закріплені", NEAREST: "2) найближчий"}


def variants(strategy: str | None) -> list[str]:
    """Обраний спосіб закріплення роботів або обидва."""
    return [strategy] if strategy else [DEDICATED, NEAREST]


def _one(args):
    params, seed, t_end, t_warmup, keep_sojourn = args
    res, soj = simulate(params, seed, t_end, t_warmup)
    return res, (soj if keep_sojourn else None)


def run_batch(tasks: list[tuple], jobs: int = 1) -> list[tuple[dict, list[float] | None]]:
    """Виконати набір прогонів (params, seed, t_end, t_warmup, keep_sojourn), за потреби паралельно."""
    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            return list(ex.map(_one, tasks, chunksize=max(1, len(tasks) // (4 * jobs))))
    return [_one(t) for t in tasks]


def replicate(params: Params, n: int, t_end: float, t_warmup: float,
              seed0: int = 1, jobs: int = 1, keep_sojourn: bool = False):
    """n повторних прогонів з зернами seed0..seed0+n-1.

    Однакові зерна для різних варіантів дають спільні випадкові числа.
    """
    tasks = [(params, seed0 + i, t_end, t_warmup, keep_sojourn) for i in range(n)]
    out = run_batch(tasks, jobs)
    results = [r for r, _ in out]
    sojourns = [s for _, s in out] if keep_sojourn else None
    return results, sojourns


def summarize(values, beta: float = 0.95) -> dict:
    """Середнє, стандартне відхилення та довірчий інтервал (Стьюдент)."""
    x = np.asarray([v for v in values if not (isinstance(v, float) and math.isnan(v))], dtype=float)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "ci": math.nan}
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else math.nan
    ci = float(stats.t.ppf((1 + beta) / 2, n - 1) * std / math.sqrt(n)) if n > 1 else math.nan
    return {"n": n, "mean": mean, "std": std, "ci": ci}


def out_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def fmt(v, digits: int = 3) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        return f"{v:.{digits}f}"
    return str(v)


def print_table(rows: list[dict], columns: list[str], headers: list[str] | None = None) -> None:
    headers = headers or columns
    cells = [[fmt(r.get(c, "")) for c in columns] for r in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) if cells else len(h) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for c in cells:
        print("  ".join(v.ljust(w) for v, w in zip(c, widths)))
