from __future__ import annotations

import math

import numpy as np
from scipy import stats

from ..params import Params
from .common import replicate

CANDIDATES = {
    "normal": (stats.norm, {}, 2),
    "exponential (shifted)": (stats.expon, {}, 2),
    "gamma (shifted)": (stats.gamma, {}, 3),
    "lognormal (shifted)": (stats.lognorm, {}, 3),
    "erlang/gamma (loc=0)": (stats.gamma, {"floc": 0.0}, 2),
}


def histogram(data: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(data, bins=bins)
    return counts, edges


def chi_square(data: np.ndarray, dist, params: tuple, n_params: int, bins: int) -> dict:
    counts, edges = histogram(data, bins)
    # крайні інтервали розширюємо до нескінченності, щоб Σ p_i = 1
    cdf = dist.cdf(edges, *params)
    cdf[0], cdf[-1] = 0.0, 1.0
    probs = np.diff(cdf)
    n = len(data)
    obs, exp_ = [], []
    acc_o = acc_e = 0.0
    for o, p in zip(counts, probs):
        acc_o += o
        acc_e += n * p
        if acc_e >= 5:
            obs.append(acc_o)
            exp_.append(acc_e)
            acc_o = acc_e = 0.0
    if acc_e > 0 and exp_:
        obs[-1] += acc_o
        exp_[-1] += acc_e
    obs, exp_ = np.array(obs), np.array(exp_)
    chi2 = float(((obs - exp_) ** 2 / exp_).sum())
    df = len(obs) - 1 - n_params
    crit = float(stats.chi2.ppf(0.95, df)) if df > 0 else math.nan
    return {"chi2": chi2, "df": df, "chi2_crit_0.05": crit, "intervals": len(obs),
            "accepted": bool(df > 0 and chi2 < crit)}


def gamma_shifted_moments(data: np.ndarray) -> tuple[float, float, float]:
    mean, std = float(data.mean()), float(data.std(ddof=1))
    skew = max(float(stats.skew(data)), 1e-3)
    k = (2.0 / skew) ** 2
    theta = std / np.sqrt(k)
    loc = min(mean - k * theta, float(data.min()) - 1e-6)
    return k, loc, theta


def fit_gamma_shifted(data: np.ndarray) -> tuple:
    k0, loc0, theta0 = gamma_shifted_moments(data)
    k, loc, theta = stats.gamma.fit(data, k0, loc=loc0, scale=theta0)
    if k < 1.0 or loc >= data.min():
        return k0, loc0, theta0
    return k, loc, theta


def fit_all(data: np.ndarray, bins: int) -> list[dict]:
    rows = []
    for name, (dist, fixed, k) in CANDIDATES.items():
        if name.startswith("exponential"):
            params = (float(data.min()), float(data.mean() - data.min()))
        elif name == "gamma (shifted)":
            params = fit_gamma_shifted(data)
        else:
            params = dist.fit(data, **fixed)
        res = chi_square(data, dist, params, k, bins)
        rows.append({"law": name, "params": ", ".join(f"{v:.4g}" for v in params),
                     "params_raw": [float(v) for v in params], **res})
    rows.sort(key=lambda r: r["chi2"])
    return rows


def run(params: Params, reps: int, t_end: float, t_warmup: float, jobs: int,
        bins: int, thin: int, max_n: int = 0) -> dict:
    _, sojourns = replicate(params, reps, t_end, t_warmup, jobs=jobs, keep_sojourn=True)
    # проріджування зменшує автокореляцію сусідніх деталей
    data = np.concatenate([np.asarray(s[::thin], dtype=float) for s in sojourns])
    if max_n and len(data) > max_n:
        # рівномірна підвибірка: при дуже великих n критерій χ² відкидає будь-яку гіпотезу
        data = data[np.linspace(0, len(data) - 1, max_n).astype(int)]
    counts, edges = histogram(data, bins)
    hist = [{"left": float(edges[i]), "right": float(edges[i + 1]), "count": int(counts[i]),
             "freq": float(counts[i] / len(data))} for i in range(len(counts))]
    describe = {
        "n": int(len(data)), "mean": float(data.mean()), "std": float(data.std(ddof=1)),
        "cv": float(data.std(ddof=1) / data.mean()), "min": float(data.min()),
        "p50": float(np.quantile(data, 0.5)), "p90": float(np.quantile(data, 0.9)),
        "p95": float(np.quantile(data, 0.95)), "max": float(data.max()),
        "skew": float(stats.skew(data)),
    }
    return {"describe": describe, "histogram": hist, "fits": fit_all(data, bins), "data": data}
