import math

import pytest

from fms.elements import FREE, R_FREE, R_WAIT
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
    # варіант 1: кожен робот повертається своїм маршрутом, холості ходи 6 + 7 + 5 = 18 с
    (DEDICATED, [226.0] + [244.0] * 9),
    # варіант 2: спершу роботи стоять ближче до точок завантаження, далі всі збираються на складі,
    # і деталь перевозить один робот: холостий хід W -> A 18 с, решта перевезень без холостого ходу
    (NEAREST, [226.0, 232.0, 239.0] + [244.0] * 7),
])
def test_deterministic_sojourn_sequence(strategy, expected):
    m = Model(deterministic(strategy=strategy, arrival=const(300.0)), seed=1).run(3300)
    assert m.sojourn_times() == [pytest.approx(x) for x in expected]


def test_second_part_includes_empty_return():
    # робот 1 після доставки стоїть у M1 і повертається до A порожнім (6 с)
    # деталь 2 надходить о 60, робот 1 звільнився о 30+22=52 у M1 -> холостий хід 6
    m = Model(deterministic(arrival=const(30.0)), seed=1, trace=True).run(400)
    lines = "\n".join(m.trace_lines)
    assert "холостий хід M1->A 6" in lines


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
    on_machines = sum(1 for mach in (m.m1, m.m2) for ch in mach.channels if ch.part is not None)
    carried = sum(1 for r in m.robots if r.task is not None and r.state in ("loaded_move", "wait_slot", "release"))
    # деталь, що захоплюється, ще лежить на місці-джерелі (у черзі A або ch.part)
    return m.buffer.length + on_machines + carried


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_conservation_and_capacity(strategy):
    p = Params(strategy=strategy)
    m = Model(p, seed=7)
    checks = []

    def observer(model: Model):
        assert model.create.created == model.completed_total + _parts_in_system(model)
        for mach in (model.m1, model.m2):
            assert sum(1 for ch in mach.channels if ch.state != FREE) <= mach.capacity
        checks.append(1)

    m.run(50_000, observe_every=97.0, observer=observer)
    assert len(checks) > 400


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_no_deadlock_under_overload(strategy):
    # сильне перевантаження: роботи постійно чекають місць, але система не зависає
    p = Params(strategy=strategy, arrival=exp(15.0))
    m = Model(p, seed=11)
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


def test_nearest_robots_serve_several_routes():
    res, _ = simulate(Params(strategy=NEAREST), seed=3, t_end=50_000)
    assert all(sum(res[f"robot{k}_jobs_route{r}"] > 0 for r in (1, 2, 3)) >= 2 for k in (1, 2, 3))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_at_most_one_extra_robot_per_machine(strategy):
    # правило u_i <= f_i при призначенні: біля верстата чекає не більше одного робота
    m = Model(Params(strategy=strategy, arrival=exp(20.0)), seed=2)
    waits = []

    def observer(model: Model):
        for mach in (model.m1, model.m2):
            assert mach.incoming <= mach.count(FREE) + 1
            assert len(mach.waiting_robots) <= 1
        waits.append(sum(1 for r in model.robots if r.state == R_WAIT))

    m.run(50_000, observe_every=13.0, observer=observer)
    assert max(waits) >= 1  # під перевантаженням роботи справді чекають з деталями


def test_fourth_dedicated_robot_serves_given_route():
    p = Params(strategy=DEDICATED, n_robots=4, dedicated_routes=(1, 2, 3, 3))
    m = Model(p, seed=3)
    assert [r.location for r in m.robots] == ["A", "M1", "M2", "M2"]
    res, _ = simulate(p, seed=3, t_end=50_000)
    assert res["robot4_jobs_route3"] > 0
    assert res["robot4_jobs_route1"] == res["robot4_jobs_route2"] == 0


def test_dedicated_routes_must_cover_all_routes():
    with pytest.raises(ValueError):
        Model(Params(strategy=DEDICATED, n_robots=3, dedicated_routes=(1, 1, 3)), seed=1)


def test_light_load_is_stable():
    res, _ = simulate(Params(arrival=exp(120.0)), seed=4, t_end=200_000, t_warmup=10_000)
    assert res["wip_end"] < 15
    assert res["throughput_per_hour"] == pytest.approx(30.0, rel=0.07)


def test_same_seed_is_reproducible():
    a, _ = simulate(Params(), seed=9, t_end=20_000)
    b, _ = simulate(Params(), seed=9, t_end=20_000)
    assert a == b


def test_robot_free_state_at_end_is_consistent():
    m = Model(Params(strategy=NEAREST), seed=1).run(10_000)
    for r in m.robots:
        assert (r.state == R_FREE) == (r.task is None)
