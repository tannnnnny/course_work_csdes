from __future__ import annotations

import numpy as np
from scipy import stats

from ..params import DEDICATED, NEAREST, Params
from .common import KEY_OUTPUTS, replicate, summarize

RESPONSES = ("throughput_per_hour", "sojourn_mean", "buffer_max", "buffer_mean",
             "M1_occupied", "M2_occupied", "robots_util_mean")


def anova(groups: list[np.ndarray], alpha: float = 0.05) -> dict:
    q = len(groups)
    p = len(groups[0])
    means = np.array([g.mean() for g in groups])
    grand = np.concatenate(groups).mean()
    s_f = p * ((means - grand) ** 2).sum()
    s_r = sum(((g - g.mean()) ** 2).sum() for g in groups)
    d_f_guide = s_f / (q * p - 1)
    d_r = s_r / (q * (p - 1))
    df1, df2 = q - 1, q * (p - 1)
    f_guide = d_f_guide / d_r if d_r > 0 else np.inf
    f_classic = (s_f / df1) / d_r if d_r > 0 else np.inf
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    return {
        "S_f": s_f, "S_r": s_r, "D_f(3.10)": d_f_guide, "D_r(3.11)": d_r,
        "F(3.10/3.11)": f_guide, "F_classic": f_classic, "F_crit": f_crit,
        "p_value": float(stats.f.sf(f_classic, df1, df2)),
        "significant": bool(f_classic > f_crit),
    }


def run(reps: int, t_end: float, t_warmup: float, jobs: int,
        base: Params | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    base = base or Params()
    runs: dict[str, list[dict]] = {}
    for strategy in (DEDICATED, NEAREST):
        runs[strategy], _ = replicate(base.with_(strategy=strategy), reps, t_end, t_warmup, jobs=jobs)

    summary = []
    for strategy, results in runs.items():
        row = {"strategy": strategy, "arrival_mean": base.arrival.mean}
        for k in KEY_OUTPUTS:
            s = summarize([r[k] for r in results])
            row[k] = s["mean"]
            row[k + "_ci"] = s["ci"]
        summary.append(row)

    tests = []
    for resp in RESPONSES:
        a = np.array([r[resp] for r in runs[DEDICATED]], dtype=float)
        b = np.array([r[resp] for r in runs[NEAREST]], dtype=float)
        res = {"response": resp, "arrival_mean": base.arrival.mean,
               "mean_dedicated": a.mean(), "mean_nearest": b.mean(), **anova([a, b])}
        diff = b - a
        t_stat, t_p = stats.ttest_rel(b, a)
        res.update({"paired_diff_mean": diff.mean(), "paired_diff_ci": summarize(diff)["ci"],
                    "paired_t": float(t_stat), "paired_p": float(t_p)})
        tests.append(res)

    raw = [r for results in runs.values() for r in results]
    return summary, tests, raw
