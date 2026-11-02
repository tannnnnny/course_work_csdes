from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
FREE_FILL = "#e9e8e3"

CM = 1 / 2.54
WIDTH = 16 * CM
STRATEGIES = ("dedicated", "nearest")
LABEL = {"dedicated": "Варіант 1 (закріплені роботи)", "nearest": "Варіант 2 (найближчий робот)"}
SHORT = {"dedicated": "Варіант 1", "nearest": "Варіант 2"}
STYLE = {
    "dedicated": {"color": SLOTS[0], "linestyle": "-", "marker": "o", "hatch": "////"},
    "nearest": {"color": SLOTS[1], "linestyle": "--", "marker": "s", "hatch": "\\\\\\\\"},
}


def _setup() -> None:
    plt.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "text.color": INK,
        "hatch.linewidth": 0.6,
        "savefig.dpi": 300,
        "savefig.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })


def num(v: float, digits: int = 1) -> str:
    """Число з десятковою комою (українська нотація)."""
    return f"{v:.{digits}f}".replace(".", ",").replace("-", "−")


def comma_axis(ax, axis: str = "both", digits: int | None = None) -> None:
    def fmt(v, _pos):
        if digits is None:
            s = f"{v:g}"
        else:
            s = f"{v:.{digits}f}"
        return s.replace(".", ",").replace("-", "−")
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float:
    v = row.get(key, "")
    return float(v) if v not in ("", None) else float("nan")


def _save(fig, out: Path) -> Path:
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def plot_capacity(results: Path, out: Path) -> Path:
    rows = _read(results / "capacity.csv")
    limits = {r["strategy"]: r for r in _read(results / "capacity_limits.csv")}
    fig, ax = plt.subplots(figsize=(WIDTH, 8.5 * CM))
    xs_all = sorted({_f(r, "input_per_hour") for r in rows})
    lo, hi = min(xs_all) - 3, max(xs_all) + 3
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1.0, zorder=1)
    ax.axvline(90, color=MUTED, linewidth=1.0, zorder=1)
    ax.text(90.8, 89.2, "умова завдання:\nexp(40), 90 дет./год", color=INK_2, fontsize=8, va="top")
    for s in STRATEGIES:
        pts = sorted((_f(r, "input_per_hour"), _f(r, "throughput_per_hour")) for r in rows if r["strategy"] == s)
        x, y = zip(*pts)
        st = STYLE[s]
        ax.plot(x, y, color=st["color"], linestyle=st["linestyle"], linewidth=1.8, marker=st["marker"],
                markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8, label=LABEL[s], zorder=3)
        sat = float(limits[s]["max_throughput_per_hour"])
        ax.annotate(f"{num(sat)} дет./год", xy=(x[-1], y[-1]), xytext=(4, 0 if s == "dedicated" else -3),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
        lim = float(limits[s]["max_stable_input_per_hour"])
        ylim = next(v for u, v in pts if abs(u - lim) < 1e-6)
        ax.annotate(f"межа стійкості ≈ {num(lim)} дет./год", xy=(lim, ylim),
                    xytext=(56, 80.5) if s == "dedicated" else (80, 66), textcoords="data",
                    fontsize=7.5, color=INK_2, ha="left", va="center", zorder=4,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.8},
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8, "shrinkA": 2, "shrinkB": 3})
    ax.set_xlim(lo, hi + 9)
    ax.set_ylim(40, 90)
    (x1, y1), (x2, y2) = ax.transData.transform([(80, 80), (90, 90)])
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    ax.text(84.2, 85.4, "вихід = вхід", color=INK_2, fontsize=8, rotation=angle, rotation_mode="anchor",
            ha="center", va="bottom")
    ax.set_xlabel("Інтенсивність надходження деталей, дет./год")
    ax.set_ylabel("Пропускна здатність, дет./год")
    comma_axis(ax)
    ax.legend(loc="lower right")
    return _save(fig, out)


def _series(results: Path, tag: str):
    data = defaultdict(lambda: defaultdict(list))
    for r in _read(results / f"warmup_series_{tag}.csv"):
        data[r["strategy"]][int(r["seed"])].append(r)
    return data


def plot_queue_growth(results: Path, out: Path, tag: str = "a40") -> Path:
    data = _series(results, tag)
    fig, ax = plt.subplots(figsize=(WIDTH, 7.5 * CM))
    for s in STRATEGIES:
        runs = list(data[s].values())
        n = min(len(r) for r in runs)
        t = np.array([_f(runs[0][k], "t") for k in range(n)]) / 1000
        q = np.array([[_f(r[k], "buffer_now") for k in range(n)] for r in runs])
        st = STYLE[s]
        ax.fill_between(t, q.min(axis=0), q.max(axis=0), color=st["color"], alpha=0.12, linewidth=0)
        ax.plot(t, q.mean(axis=0), color=st["color"], linestyle=st["linestyle"], linewidth=1.8,
                label=f"{LABEL[s]}: середнє 5 прогонів")
        ax.annotate(f"{q.mean(axis=0)[-1]:.0f} дет.", xy=(t[-1], q.mean(axis=0)[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.fill_between([], [], [], color=MUTED, alpha=0.25, label="діапазон між прогонами (мін.–макс.)")
    ax.set_xlim(0, t[-1] * 1.08)
    ax.set_ylim(0, None)
    ax.set_xlabel("Модельний час, тис. с")
    ax.set_ylabel("Деталей на площадці A, шт.")
    comma_axis(ax)
    ax.legend(loc="upper left")
    return _save(fig, out)


def plot_warmup(results: Path, out: Path, tag: str, metric: str, ylabel: str, xscale: float = 1000.0,
                xlabel: str = "Модельний час, тис. с", digits: int = 1) -> Path:
    data = _series(results, tag)
    summary = {(r["strategy"], r["metric"]): r for r in _read(results / f"warmup_summary_{tag}.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 7.2 * CM), sharey=True)
    for ax, s in zip(axes, STRATEGIES):
        runs = list(data[s].values())
        st = STYLE[s]
        sm = summary[(s, metric)]
        level, eps, T = _f(sm, "final_mean"), _f(sm, "eps"), _f(sm, "T_transient")
        tmax = 0.0
        for i, r in enumerate(runs):
            t = np.array([_f(x, "t") for x in r]) / xscale
            y = np.array([_f(x, metric) for x in r])
            tmax = max(tmax, t[-1])
            ax.plot(t, y, color=st["color"], linewidth=0.9, alpha=0.85)
        ax.axhspan(level - eps, level + eps, color=MUTED, alpha=0.18, linewidth=0,
                   label="рівень ± ε (ε = 5%)")
        if not np.isnan(T):
            ax.axvline(T / xscale, color=INK_2, linewidth=1.0, label="момент T за (3.1)")
            ax.annotate(f"T = {num(T / xscale, 0 if T / xscale >= 10 else 1)}", xy=(T / xscale, 1.0),
                        xycoords=("data", "axes fraction"), xytext=(3, -10), textcoords="offset points",
                        fontsize=8, color=INK)
        ax.set_title(LABEL[s], fontsize=9, color=INK, loc="left")
        ax.set_xlim(0, tmax)
        ax.set_xlabel(xlabel)
        comma_axis(ax, "x")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.{digits}f}".replace(".", ",")))
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    handles = [Line2D([], [], color=INK_2, linewidth=0.9)] + handles
    labels = ["прогони 1–5 (колір – варіант)"] + labels
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, out)


def _bar_labels(ax, bars, values, digits=2, inside=False):
    for b, v in zip(bars, values):
        ax.annotate(num(v, digits), xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5,
                    color=INK)


def plot_utilization(results: Path, out: Path, tag: str = "a40") -> Path:
    summary = {r["strategy"]: r for r in _read(results / f"compare_{tag}_summary.csv")}
    keys = [("robot1_util", "Робот R1"), ("robot2_util", "Робот R2"), ("robot3_util", "Робот R3"),
            ("M1_occupied", "Верстат M1"), ("M2_occupied", "Верстат M2")]
    fig, ax = plt.subplots(figsize=(WIDTH, 7.5 * CM))
    x = np.arange(len(keys))
    w = 0.34
    for i, s in enumerate(STRATEGIES):
        st = STYLE[s]
        vals = [_f(summary[s], k) for k, _ in keys]
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w * 0.92, color=st["color"], hatch=st["hatch"],
                      edgecolor=SURFACE, linewidth=0.0, label=LABEL[s], zorder=2)
        for b in bars:  # штрихування темнішим відтінком того самого кольору (для друку)
            b.set_edgecolor(_darker(st["color"]))
            b.set_linewidth(0)
        _bar_labels(ax, bars, vals)
    ax.set_xticks(x, [lbl for _, lbl in keys])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Коефіцієнт використання")
    comma_axis(ax, "y", 1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13))
    return _save(fig, out)


def _darker(hex_color: str, factor: float = 0.72) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(*(int(c * factor) for c in (r, g, b)))


PHASES = [("empty_move", "холостий хід, повернення"), ("grab", "захоплення"),
          ("loaded_move", "переміщення з деталлю"), ("wait_slot", "блокування: робот тримає деталь"),
          ("release", "вивільнення")]


def plot_robot_phases(results: Path, out: Path, tag: str = "a40") -> Path:
    raw = _read(results / f"compare_{tag}_raw.csv")
    rows = []
    for s in STRATEGIES:
        runs = [r for r in raw if r["strategy"] == s]
        for k in (1, 2, 3):
            shares = [np.mean([_f(r, f"robot{k}_{p}") for r in runs]) for p, _ in PHASES]
            rows.append((f"{SHORT[s]}, R{k}", shares))
    fig, ax = plt.subplots(figsize=(WIDTH, 8.0 * CM))
    y = np.arange(len(rows))[::-1].astype(float)
    y[3:] -= 0.0
    y = np.array([5.6, 4.6, 3.6, 2.0, 1.0, 0.0])
    for (label, shares), yy in zip(rows, y):
        left = 0.0
        for (p, name), v, color in zip(PHASES, shares, SLOTS):
            ax.barh(yy, v * 100, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=1.2,
                    label=name if yy == y[0] else None, zorder=2)
            if v >= 0.07:
                txt_color = INK if color == SLOTS[3] else "#ffffff"
                ax.text(left + v * 50, yy, f"{v * 100:.0f}%", ha="center", va="center", fontsize=7.5,
                        color=txt_color, zorder=3)
            left += v * 100
        free = 100 - left
        ax.barh(yy, free, left=left, height=0.62, color=FREE_FILL, edgecolor=SURFACE, linewidth=1.2,
                label="вільний" if yy == y[0] else None, zorder=2)
        if free >= 7:
            ax.text(left + free / 2, yy, f"{free:.0f}%", ha="center", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Частка часу моделювання, %")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.45, 1.24))
    return _save(fig, out)


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
    "erlang/gamma (loc=0)": "гамма без зсуву",
}
CURVE_STYLES = [(SLOTS[1], "-"), (SLOTS[2], "--"), ("#6250d6", ":")]


def plot_distribution(results: Path, out: Path, strategy: str, tag: str = "a60", top: int = 3) -> Path:
    name = f"{strategy}_{tag}"
    data = np.loadtxt(results / f"dist_data_{name}.txt")
    hist = _read(results / f"dist_hist_{name}.csv")
    fits = json.loads((results / f"dist_fits_{name}.json").read_text(encoding="utf-8"))
    edges = np.array([_f(h, "left") for h in hist] + [_f(hist[-1], "right")])
    freq = np.array([_f(h, "freq") for h in hist])
    width = edges[1] - edges[0]
    fig, ax = plt.subplots(figsize=(WIDTH, 8.0 * CM))
    ax.bar(edges[:-1], freq, width=width, align="edge", color="#b7d3f6", edgecolor=SURFACE, linewidth=1.2,
           label=f"відносна частота (n = {len(data)})", zorder=2)
    xs = np.linspace(max(0.0, edges[0] - width), edges[-1] + width, 600)
    for (color, ls), fit in zip(CURVE_STYLES, fits[:top]):
        dist = SCIPY_DIST[fit["law"]]
        ys = dist.pdf(xs, *fit["params_raw"]) * width
        verdict = "приймається" if fit["accepted"] else "відхиляється"
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.8, zorder=3,
                label=f"{LAW_UA[fit['law']]}: χ² = {num(fit['chi2'])}, χ²кр = {num(fit['chi2_crit_0.05'])} "
                      f"(ν = {fit['df']}) – {verdict}")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(0, max(freq.max(), *(ln.get_ydata().max() for ln in ax.lines)) * 1.5)
    ax.set_xlabel("Час проходження деталі, с")
    ax.set_ylabel("Відносна частота")
    comma_axis(ax, "y", 2)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, out)


def plot_improvement(results: Path, out: Path, tag: str = "a40") -> Path:
    rows = _read(results / f"improvement_{tag}.csv")
    groups = {s: [r for r in rows if r["strategy"] == s] for s in STRATEGIES}
    heights = [len(groups[s]) for s in STRATEGIES]
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 10.5 * CM), sharex=True,
                             gridspec_kw={"height_ratios": heights})
    inp = _f(rows[0], "input_per_hour")
    for ax, s in zip(axes, STRATEGIES):
        st = STYLE[s]
        rs = groups[s]
        y = np.arange(len(rs))[::-1]
        vals = [_f(r, "throughput_per_hour") for r in rs]
        ax.barh(y, vals, height=0.6, color=st["color"], hatch=st["hatch"], edgecolor=_darker(st["color"]),
                linewidth=0, zorder=2)
        for yy, v, r in zip(y, vals, rs):
            stable = r["stable"] in ("True", "true", "1")
            ax.text(v + 0.8, yy, f"{num(v)} ({'стійкий' if stable else 'черга росте'})", va="center",
                    fontsize=7.5, color=INK, zorder=4,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.8})
        ax.set_yticks(y, [r["scenario"] for r in rs], fontsize=8)
        ax.axvline(inp, color=INK_2, linewidth=1.0, zorder=3)
        ax.set_title(LABEL[s], fontsize=9, color=INK, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 112)
    axes[0].annotate(f"вхідний потік {num(inp, 0)} дет./год", xy=(inp, 1.0), xycoords=("data", "axes fraction"),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=7.5,
                     color=INK_2)
    axes[-1].set_xlabel("Пропускна здатність, дет./год")
    comma_axis(axes[-1], "x")
    fig.tight_layout()
    return _save(fig, out)


def make_all(results: str | Path = "results") -> list[Path]:
    _setup()
    results = Path(results)
    out = results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    made = [
        plot_capacity(results, out / "fig_capacity.png"),
        plot_queue_growth(results, out / "fig_queue_growth_a40.png"),
        plot_warmup(results, out / "fig_warmup_throughput_a40.png", "a40", "throughput_per_hour",
                    "Пропускна здатність, дет./год", digits=0),
        plot_warmup(results, out / "fig_warmup_sojourn_a60.png", "a60", "sojourn_mean",
                    "Середній час проходження, с", digits=0),
        plot_utilization(results, out / "fig_utilization_a40.png"),
        plot_robot_phases(results, out / "fig_robot_phases_a40.png"),
        *(plot_distribution(results, out / f"fig_dist_{s}_a60.png", s) for s in STRATEGIES),
    ]
    if (results / "improvement_a40.csv").exists():
        made.append(plot_improvement(results, out / "fig_improvement_a40.png"))
    return made
