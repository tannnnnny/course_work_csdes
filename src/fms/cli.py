"""Командний інтерфейс: прогони моделі та експерименти.

Приклади:
  fms run --strategy nearest
  fms trace --strategy dedicated --until 600
  fms verify --jobs 4
  fms warmup --arrival 60
  fms reps
  fms compare --reps 30
  fms capacity --reps 10 --jobs 4
  fms dist --arrival 60
  fms improve --reps 20
"""

from __future__ import annotations

import argparse
import os
import sys

from .model import Model, simulate
from .params import DEDICATED, NEAREST, Params
from .rng import exp
from .experiments import capacity, compare, distribution, improvement, replications, verification, warmup
from .experiments.common import out_dir, print_table, variants, write_csv, write_json


def _params(a, strategy=None) -> Params:
    return Params(strategy=strategy or a.strategy or DEDICATED, arrival=exp(a.arrival))


def _tag(a) -> str:
    return f"a{a.arrival:g}"


def cmd_run(a):
    for s in variants(a.strategy):
        res, _ = simulate(_params(a, s), a.seed, a.t_end, a.warmup)
        print(f"\n=== {s} (seed={a.seed}, Tmod={a.t_end:g}, перехідний={a.warmup:g}) ===")
        for k, v in res.items():
            print(f"  {k:<28} {v:.4f}" if isinstance(v, float) else f"  {k:<28} {v}")


def cmd_trace(a):
    d = out_dir(a.out)
    for s in variants(a.strategy):
        m = Model(_params(a, s), seed=a.seed, trace=True, trace_until=a.until).run(a.until)
        path = d / f"trace_{s}.txt"
        header = f"Протокол подій: {m.params.describe()}, seed={a.seed}\n"
        legend = ("Стан: A – кількість зайнятих місць площадки; для пристроїв СМО K = 0 вільний, "
                  "1 зайнятий, 2 блокований; R@точка – положення робота\n")
        path.write_text(header + legend + "\n".join(m.trace_lines) + "\n", encoding="utf-8")
        print(f"{s}: {len(m.trace_lines)} рядків -> {path}")
        if a.print:
            print("\n".join(m.trace_lines))


def cmd_verify(a):
    d = out_dir(a.out)
    rows = []
    for s in variants(a.strategy):
        part = verification.run(s, a.reps, a.t_end, a.warmup, a.jobs)
        print(f"\n=== Верифікація: {s} ({a.reps} прогонів по {a.t_end:g} с, перехідний {a.warmup:g} с) ===")
        print_table(part, ["scenario", *verification.COLUMNS],
                    ["сценарій", "дет/год", "Tпрох", "maxA", "серA", "M1зайн", "M2зайн", "M2обр", "роботи", "WIPкін"])
        rows += part
    write_csv(d / "verification.csv", rows)
    print(f"\n-> {d / 'verification.csv'}")


def cmd_warmup(a):
    d = out_dir(a.out)
    all_rows, all_summary = [], []
    for s in variants(a.strategy):
        rows, summary = warmup.run(_params(a, s), a.runs, a.t_end, a.step, a.eps)
        all_rows += rows
        all_summary += summary
        print(f"\n=== Перехідний період: {s}, exp({a.arrival:g}) ({a.runs} прогонів, ε = {a.eps:.0%} від рівня) ===")
        print_table(summary, ["metric", "final_mean", "eps", "T_transient", "growth_last_half"],
                    ["відгук", "рівень", "ε", "T перех.", "приріст 2-ї пол."])
    write_csv(d / f"warmup_series_{_tag(a)}.csv", all_rows)
    write_csv(d / f"warmup_summary_{_tag(a)}.csv", all_summary)
    print(f"\n-> {d}/warmup_*_{_tag(a)}.csv")
    print("Примітка: T перех. = — означає, що умова (3.1) не виконується до кінця прогону "
          "(стаціонарний режим відсутній або потрібен довший прогін).")


def cmd_reps(a):
    d = out_dir(a.out)
    rows = []
    for s in variants(a.strategy):
        part = replications.run(_params(a, s), a.pilot, a.t_end, a.warmup, a.eps, 0.95, a.jobs)
        print(f"\n=== Кількість прогонів: {s}, exp({a.arrival:g}) (пілотна серія {a.pilot}, ε = {a.eps:.0%}) ===")
        print_table(part, ["response", "mean", "std", "eps", "p_chebyshev", "p_clt"],
                    ["відгук", "середнє", "σ", "ε", "p Чебишов", "p ЦГТ"])
        rows += part
    write_csv(d / f"replications_{_tag(a)}.csv", rows)
    print(f"\n-> {d / f'replications_{_tag(a)}.csv'}")


def cmd_compare(a):
    d = out_dir(a.out)
    summary, tests, raw = compare.run(a.reps, a.t_end, a.warmup, a.jobs, base=Params(arrival=exp(a.arrival)))
    print(f"\n=== Порівняння способів закріплення роботів: exp({a.arrival:g}), {a.reps} прогонів ===")
    print_table(summary, ["strategy", "throughput_per_hour", "throughput_per_hour_ci", "sojourn_mean",
                          "buffer_max", "M1_occupied", "M2_occupied", "robot1_util", "robot2_util",
                          "robot3_util"],
                ["варіант", "дет/год", "±", "Tпрох", "maxA", "M1", "M2", "R1", "R2", "R3"])
    print("\nДисперсійний аналіз (фактор – спосіб закріплення):")
    print_table(tests, ["response", "mean_dedicated", "mean_nearest", "F_classic", "F_crit",
                        "p_value", "significant", "paired_diff_mean", "paired_p"],
                ["відгук", "варіант 1", "варіант 2", "F", "F кр", "p", "значущий", "Δ(2−1)", "p парн."])
    t = _tag(a)
    write_csv(d / f"compare_{t}_summary.csv", summary)
    write_csv(d / f"compare_{t}_anova.csv", tests)
    write_csv(d / f"compare_{t}_raw.csv", raw)
    print(f"\n-> {d}/compare_{t}_*.csv")


def cmd_capacity(a):
    d = out_dir(a.out)
    arrivals = [float(x) for x in a.arrivals.split(",")] if a.arrivals else capacity.DEFAULT_ARRIVALS
    rows = capacity.run(arrivals, a.strategy, a.reps, a.t_end, a.warmup, a.jobs)
    print(f"\n=== Пропускна здатність ({a.reps} прогонів на точку) ===")
    print_table(rows, ["strategy", "arrival_mean", "input_per_hour", "throughput_per_hour",
                       "output_input_ratio", "sojourn_mean", "buffer_max", "buffer_q99", "wip_end", "stable"],
                ["варіант", "інтервал", "вхід/год", "вихід/год", "вих/вх", "Tпрох", "maxA",
                 "A q99", "WIPкін", "стійкий"])
    best = capacity.max_stable(rows)
    print("\nМежа стійкості:")
    print_table(best, ["strategy", "max_stable_input_per_hour", "min_stable_arrival_mean",
                       "max_throughput_per_hour"],
                ["варіант", "макс. стійкий вхід/год", "мін. інтервал", "макс. вихід/год"])
    write_csv(d / "capacity.csv", rows)
    write_csv(d / "capacity_limits.csv", best)
    print(f"\n-> {d / 'capacity.csv'}, {d / 'capacity_limits.csv'}")


def cmd_dist(a):
    d = out_dir(a.out)
    for s in variants(a.strategy):
        params = _params(a, s)
        res = distribution.run(params, a.reps, a.t_end, a.warmup, a.jobs, a.bins, a.thin, a.max_n)
        tag = f"{s}_{_tag(a)}"
        print(f"\n=== Розподіл часу проходження: {s}, інтервал надходження exp({a.arrival:g}) ===")
        print("  " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in res["describe"].items()))
        print_table(res["fits"], ["law", "params", "chi2", "df", "chi2_crit_0.05", "accepted"],
                    ["закон", "параметри", "χ²", "df", "χ² кр", "прийнято"])
        write_csv(d / f"dist_hist_{tag}.csv", res["histogram"])
        write_csv(d / f"dist_fits_{tag}.csv", res["fits"])
        write_json(d / f"dist_fits_{tag}.json", res["fits"])
        write_json(d / f"dist_describe_{tag}.json", res["describe"])
        (d / f"dist_data_{tag}.txt").write_text("\n".join(f"{x:.4f}" for x in res["data"]), encoding="utf-8")
        print(f"-> {d}/dist_*_{tag}.*")


def cmd_improve(a):
    d = out_dir(a.out)
    rows = improvement.run(a.reps, a.t_end, a.warmup, a.jobs, base=Params(arrival=exp(a.arrival)))
    print(f"\n=== Пробні запуски пропозицій: exp({a.arrival:g}), {a.reps} прогонів ===")
    print_table(rows, ["scenario", "strategy", "throughput_per_hour", "throughput_per_hour_ci",
                       "output_input_ratio", "stable", "sojourn_mean", "buffer_max", "M2_occupied",
                       "robots_util_mean"],
                ["сценарій", "варіант", "дет/год", "±", "вих/вх", "стійкий", "Tпрох", "maxA", "M2",
                 "роботи"])
    write_csv(d / f"improvement_{_tag(a)}.csv", rows)
    print(f"\n-> {d / f'improvement_{_tag(a)}.csv'}")


def cmd_plots(a):
    from .report_plots import make_all

    for path in make_all(a.out):
        print(f"-> {path}")


def cmd_plot_dist(a):
    from pathlib import Path

    from .plots import available_tags, plot_distribution

    results = Path(a.out)
    tags = [a.tag] if a.tag else available_tags(results)
    if not tags:
        print(f"У {results} немає результатів `fms dist` (dist_data_*.txt). Спочатку запустіть `fms dist`.")
        return
    for tag in tags:
        path = plot_distribution(results, tag, top=a.top, dpi=a.dpi)
        print(f"{tag}: -> {path}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fms", description="Імітаційна модель роботизованої ГВС (B20)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp, t_end=200_000.0, warm=30_000.0, reps=10):
        sp.add_argument("--strategy", choices=[DEDICATED, NEAREST], help="за замовчуванням – обидві")
        sp.add_argument("--arrival", type=float, default=40.0, help="середній інтервал надходження, с")
        sp.add_argument("--t-end", type=float, default=t_end, help="час моделювання, с")
        sp.add_argument("--warmup", type=float, default=warm, help="перехідний період, с")
        sp.add_argument("--reps", type=int, default=reps, help="кількість прогонів")
        sp.add_argument("--seed", type=int, default=1)
        sp.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
        sp.add_argument("--out", default="results")

    sp = sub.add_parser("run", help="один прогін і всі вихідні змінні"); common(sp); sp.set_defaults(f=cmd_run)
    sp = sub.add_parser("trace", help="протокол подій"); common(sp)
    sp.add_argument("--until", type=float, default=600.0)
    sp.add_argument("--print", action="store_true")
    sp.set_defaults(f=cmd_trace)
    sp = sub.add_parser("verify", help="таблиця верифікації"); common(sp, 100_000, 10_000, 5); sp.set_defaults(f=cmd_verify)
    sp = sub.add_parser("warmup", help="перехідний період"); common(sp, 300_000, 0)
    sp.add_argument("--runs", type=int, default=5)
    sp.add_argument("--step", type=float, default=1_000.0)
    sp.add_argument("--eps", type=float, default=0.05, help="точність, частка від рівня відгуку")
    sp.set_defaults(f=cmd_warmup)
    sp = sub.add_parser("reps", help="кількість повторних прогонів"); common(sp)
    sp.add_argument("--pilot", type=int, default=20)
    sp.add_argument("--eps", type=float, default=0.02)
    sp.set_defaults(f=cmd_reps)
    sp = sub.add_parser("compare", help="порівняння способів закріплення + ANOVA"); common(sp, reps=30)
    sp.set_defaults(f=cmd_compare)
    sp = sub.add_parser("capacity", help="пропускна здатність vs інтенсивність надходження"); common(sp)
    sp.add_argument("--arrivals", help="список середніх інтервалів через кому")
    sp.set_defaults(f=cmd_capacity)
    sp = sub.add_parser("dist", help="розподіл часу проходження + χ²"); common(sp, reps=20)
    sp.add_argument("--bins", type=int, default=20)
    sp.add_argument("--thin", type=int, default=5, help="брати кожну k-ту деталь")
    sp.add_argument("--max-n", type=int, default=0, help="обмежити обсяг вибірки (0 – усі)")
    sp.set_defaults(f=cmd_dist)
    sp = sub.add_parser("improve", help="пробні запуски пропозицій щодо поліпшення"); common(sp, reps=20)
    sp.set_defaults(f=cmd_improve)
    sp = sub.add_parser("plots", help="рисунки для записки за результатами експериментів")
    sp.add_argument("--out", default="results", help="каталог з результатами (рисунки – у figures/)")
    sp.set_defaults(f=cmd_plots)
    sp = sub.add_parser("plot-dist", help="графік гістограми за результатами `fms dist`")
    sp.add_argument("--tag", help="напр. nearest_a60; за замовчуванням – усі знайдені")
    sp.add_argument("--top", type=int, default=3, help="скільки найкращих законів накласти")
    sp.add_argument("--dpi", type=int, default=200)
    sp.add_argument("--out", default="results", help="каталог з результатами (туди ж зберігається PNG)")
    sp.set_defaults(f=cmd_plot_dist)
    return ap


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    args.f(args)


if __name__ == "__main__":
    main()
