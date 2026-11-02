import math

import pytest

from fms.elements import BLOCKED, BUSY, FREE
from fms.model import Model, simulate
from fms.params import DEDICATED, NEAREST, Params
from fms.rng import Stream, const, exp, norm, unif

STRATEGIES = [DEDICATED, NEAREST]


def deterministic(**kw) -> Params:
    base = dict(arrival=const(1000.0), m1_time=const(60.0), m2_time=const(100.0), handling=const(8.0))
    base.update(kw)
    return Params(**base)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_single_part_time_is_exact(strategy):
    # 8+6+8 (A->M1) + 60 + 8+7+8 (M1->M2) + 100 + 8+5+8 (M2->W) = 226
    m = Model(deterministic(strategy=strategy), seed=1).run(1500)
    assert m.sojourn_times() == [pytest.approx(226.0)]


@pytest.mark.parametrize("strategy,expected", [
    (DEDICATED, [226.0] * 10),
    (NEAREST, [226.0, 232.0, 239.0] + [244.0] * 7),
])
def test_deterministic_sojourn_sequence(strategy, expected):
    m = Model(deterministic(strategy=strategy, arrival=const(300.0)), seed=1).run(3300)
    assert m.sojourn_times() == [pytest.approx(x) for x in expected]


def test_dedicated_robot_is_released_after_return():
    # робот R1: захоплення 8 + хід 6, після переходу деталі на M1 – вивільнення 8 + повернення 6
    m = Model(deterministic(), seed=1, trace=True).run(1100)
    lines = "\n".join(m.trace_lines)
    assert "R1      затримка звільнення до 1028.000" in lines


@pytest.mark.parametrize("dist,mean,std", [
    (exp(40.0), 40.0, 40.0),
    (unif(8.0, 1.0), 8.0, 2.0 / (2 * math.sqrt(3))),
    (norm(60.0, 10.0), 60.0, 10.0),
])
def test_generators_moments(dist, mean, std):
    s = Stream(123)
    xs = [dist.sample(s) for _ in range(200_000)]
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    assert m == pytest.approx(mean, rel=0.01)
    assert sd == pytest.approx(std, rel=0.02)


def test_uniform_handling_bounds():
    s = Stream(5)
    xs = [unif(8.0, 1.0).sample(s) for _ in range(10_000)]
    assert min(xs) >= 7.0 and max(xs) <= 9.0


def _parts_in_system(m: Model) -> int:
    return sum(qs.requests for qs in (m.A, m.M1, m.M2, m.W, *m.T.values()))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_conservation_capacity_and_states(strategy):
    m = Model(Params(strategy=strategy), seed=7)
    checks = []

    def observer(model: Model):
        assert model.create.created == model.completed_total + _parts_in_system(model)
        for qs in (model.M1, model.M2):
            assert sum(1 for ch in qs.channels if ch.K != FREE) <= 3
        for qs in model.stat_qs:
            assert all(ch.K in (FREE, BUSY, BLOCKED) for ch in qs.channels)
            assert qs.n == [sum(1 for ch in qs.channels if ch.K == k) for k in (FREE, BUSY, BLOCKED)]
        checks.append(1)

    m.run(50_000, observe_every=97.0, observer=observer)
    assert len(checks) > 400


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_no_deadlock_under_overload(strategy):
    # сильне перевантаження: роботи постійно чекають місць, але система не зупиняється
    m = Model(Params(strategy=strategy, arrival=exp(15.0)), seed=11)
    done = []
    m.run(100_000, observe_every=5_000.0, observer=lambda mm: done.append(mm.completed_total))
    increments = [b - a for a, b in zip(done, done[1:])]
    assert min(increments) > 0


def test_dedicated_robots_serve_only_own_route():
    res, _ = simulate(Params(strategy=DEDICATED), seed=3, t_end=50_000)
    for k in (1, 2, 3):
        for route in (1, 2, 3):
            jobs = res[f"robot{k}_jobs_route{route}"]
            assert (jobs > 0) if route == k else (jobs == 0)
    assert [res[f"empty_route{S}_mean"] for S in (1, 2, 3)] == [6.0, 7.0, 5.0]


def test_nearest_robots_serve_several_routes():
    res, _ = simulate(Params(strategy=NEAREST), seed=3, t_end=50_000)
    assert all(sum(res[f"robot{k}_jobs_route{r}"] > 0 for r in (1, 2, 3)) >= 2 for k in (1, 2, 3))


def test_nearest_rule_takes_closest_free_robot():
    m = Model(Params(strategy=NEAREST), seed=1, trace=True, trace_until=20.0).run(20.0)
    lines = "\n".join(m.trace_lines)
    assert "П:R2    холостий хід M1 -> A (6 с)" in lines


def test_at_most_one_robot_waits_per_machine():
    m = Model(Params(strategy=NEAREST, arrival=exp(20.0)), seed=2)
    waits = []

    def observer(model: Model):
        for S in (1, 2):
            assert model.T[S].n[BLOCKED] <= 1
            assert model.u(S) <= model.M[S].n[FREE] + 1
        waits.append(model.T[1].n[BLOCKED] + model.T[2].n[BLOCKED])

    m.run(50_000, observe_every=13.0, observer=observer)
    assert max(waits) >= 1  # під перевантаженням роботи справді чекають з деталями


def test_fourth_dedicated_robot_serves_given_route():
    p = Params(strategy=DEDICATED, n_robots=4, dedicated_routes=(1, 2, 3, 3))
    m = Model(p, seed=3)
    assert len(m.T[3].channels) == 2
    res, _ = simulate(p, seed=3, t_end=50_000)
    assert res["robot4_jobs_route3"] > 0
    assert res["robot4_jobs_route1"] == res["robot4_jobs_route2"] == 0


def test_dedicated_routes_must_cover_all_routes():
    with pytest.raises(ValueError):
        Model(Params(strategy=DEDICATED, n_robots=3, dedicated_routes=(1, 1, 3)), seed=1)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_light_load_is_stable(strategy):
    res, _ = simulate(Params(strategy=strategy, arrival=exp(120.0)), seed=4, t_end=200_000, t_warmup=10_000)
    assert res["wip_end"] < 15
    assert res["throughput_per_hour"] == pytest.approx(30.0, rel=0.07)


def test_same_seed_is_reproducible():
    a, _ = simulate(Params(strategy=NEAREST), seed=9, t_end=20_000)
    b, _ = simulate(Params(strategy=NEAREST), seed=9, t_end=20_000)
    assert a == b


def test_robot_utilization_is_sum_of_phases():
    res, _ = simulate(Params(strategy=NEAREST), seed=5, t_end=60_000, t_warmup=10_000)
    for k in (1, 2, 3):
        phases = sum(res[f"robot{k}_{p}"] for p in ("grab", "loaded_move", "wait_slot", "release", "empty_move"))
        assert res[f"robot{k}_util"] == pytest.approx(phases)
        assert 0.0 < res[f"robot{k}_util"] <= 1.0
