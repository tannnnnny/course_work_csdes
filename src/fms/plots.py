from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np  
from scipy import stats  

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BAR = "#9ec5f4"
CURVES = ("#eb6834", "#1baf7a", "#6250d6")
LINESTYLES = ("-", "--", ":")

SCIPY_DIST = {
    "normal": stats.norm,
    "exponential (shifted)": stats.expon,
    "gamma (shifted)": stats.gamma,
    "lognormal (shifted)": stats.lognorm,
    "erlang/gamma (loc=0)": stats.gamma,
}
LAW_UA = {
    "normal": "нормальний",
    "exponential (shifted)": "експоненціальний (зсунутий)",
    "gamma (shifted)": "гамма (зсунутий)",
    "lognormal (shifted)": "логнормальний (зсунутий)",
    "erlang/gamma (loc=0)": "гамма / Ерланга (без зсуву)",
}
STRATEGY_UA = {"dedicated": "варіант 1 (закріплені роботи)", "nearest": "варіант 2 (найближчий робот)"}


def available_tags(results: Path) -> list[str]:
    return sorted(p.name[len("dist_data_"):-len(".txt")] for p in results.glob("dist_data_*.txt"))


def _title(tag: str) -> str:
    parts = tag.split("_")
    if len(parts) == 2 and parts[1].startswith("a"):
        strategy, arrival = parts
        return f"{STRATEGY_UA.get(strategy, strategy)}, інтервал надходження exp({arrival[1:]}) с"
    return tag


def plot_distribution(results: Path, tag: str, top: int = 3, out: Path | None = None, dpi: int = 200) -> Path:
    data = np.loadtxt(results / f"dist_data_{tag}.txt")
    with (results / f"dist_hist_{tag}.csv").open(encoding="utf-8") as f:
        hist = list(csv.DictReader(f))
    fits = json.loads((results / f"dist_fits_{tag}.json").read_text(encoding="utf-8"))
    if fits and "params_raw" not in fits[0]:
        raise ValueError("У файлі підбору немає параметрів повної точності – перезапустіть `fms dist`.")

    edges = np.array([float(h["left"]) for h in hist] + [float(hist[-1]["right"])])
    freq = np.array([float(h["freq"]) for h in hist])
    width = edges[1] - edges[0]
    n = len(data)

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar(edges[:-1], freq, width=width, align="edge", color=BAR, edgecolor=SURFACE, linewidth=1.5,
           label=f"відносна частота (n = {n})", zorder=2)

    xs = np.linspace(max(0.0, edges[0] - width), edges[-1] + width, 600)
    for i, fit in enumerate(fits[:top]):
        dist = SCIPY_DIST[fit["law"]]
        ys = dist.pdf(xs, *fit["params_raw"]) * width
        verdict = "прийнято" if fit["accepted"] else "відхилено"
        label = (f"{LAW_UA.get(fit['law'], fit['law'])}: χ² = {fit['chi2']:.1f}, "
                 f"χ²кр = {fit['chi2_crit_0.05']:.1f} ({verdict})")
        ax.plot(xs, ys, color=CURVES[i % len(CURVES)], linestyle=LINESTYLES[i % len(LINESTYLES)],
                linewidth=2, label=label, zorder=3)

    mean, std = data.mean(), data.std(ddof=1)
    ax.set_title("Розподіл часу проходження деталей", loc="left", fontsize=13, color=INK, pad=22)
    ax.text(0, 1.02, f"{_title(tag)};  середнє {mean:.0f} с, σ = {std:.0f} с",
            transform=ax.transAxes, fontsize=9.5, color=INK_2)
    ax.set_xlabel("Час проходження, с", color=INK_2)
    ax.set_ylabel("Відносна частота", color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(0, max(freq.max(), *(l.get_ydata().max() for l in ax.lines)) * 1.12)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in leg.get_texts():
        text.set_color(INK)
    fig.tight_layout()

    out = out or results / f"dist_hist_{tag}.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out
