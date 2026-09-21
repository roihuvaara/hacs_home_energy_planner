"""Cell-balance constraint inside the LP/MILP engines.

The policy lives in test_balance.py; this file asserts that the solver
actually honours ``min_peak_buffer_kwh`` and that the premium it produces
means what balance.py assumes it means.
"""

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "custom_components"))

pytest.importorskip("highspy")

from home_energy_planner.balance import reference_charge_cost_cents  # noqa: E402
from home_energy_planner.battery_core import (  # noqa: E402
    current_to_period_kwh,
    CHARGE_EFF,
    BatteryParams,
    Period,
    compile_slots,
)
from home_energy_planner.milp_core import (  # noqa: E402
    TankParams,
    solve_best,
    solve_joint,
    solve_lp,
)

TZ = ZoneInfo("Europe/Helsinki")


def battery(soc=18.0, reserve=18.0):
    return BatteryParams(
        capacity_kwh=5.12,
        state_of_health_pct=97.0,
        soc_pct=soc,
        reserve_soc_pct=reserve,
        max_charge_current=25,
        max_discharge_current=25,
    )


def day(hourly_prices, hourly_load, hourly_solar=None, export_cents=0.0, days=1):
    hourly_solar = hourly_solar or [0.0] * len(hourly_prices)
    start = datetime(2026, 1, 15, 0, 0, tzinfo=TZ)
    periods = []
    index = 0
    for _ in range(days):
        for hour, (price, load, solar) in enumerate(
            zip(hourly_prices, hourly_load, hourly_solar)
        ):
            for quarter in range(4):
                periods.append(
                    Period(
                        start=start + timedelta(minutes=15 * index),
                        price_cents_per_kwh=price,
                        load_kwh=load / 4.0,
                        solar_kwh=solar / 4.0,
                        export_cents_per_kwh=export_cents,
                    )
                )
                index += 1
    return periods


def peak_soc(plan, params):
    return max(params.soc_from_buffer_kwh(p.buffer_end_kwh) for p in plan.periods)


# the August horizon that prompted this work: a 4.8 c/kWh spread, just
# under the round-trip + cycle-cost break-even, so the optimizer parks the
# pack at the reserve floor for the whole horizon
FLAT_AUGUST = ([6.5] * 6 + [8.7] * 12 + [11.2] * 4 + [6.8] * 2, [0.8] * 24)


def test_unconstrained_august_horizon_never_leaves_the_floor():
    params = battery()
    periods = day(*FLAT_AUGUST)
    plan = solve_lp(periods, params)
    assert peak_soc(plan, params) == pytest.approx(18.0, abs=0.5)


def test_balance_constraint_reaches_the_target():
    params = battery()
    periods = day(*FLAT_AUGUST)
    target = params.buffer_kwh_from_soc(100.0)
    plan = solve_lp(periods, params, target)
    assert peak_soc(plan, params) == pytest.approx(100.0, abs=0.5)


def test_balance_peak_is_not_an_endpoint():
    """The pack may be spent again inside the horizon after balancing.

    This is what makes the premium a fair net price: the constrained solve
    keeps the recovery discharge, so we are not charged for energy we
    still own.
    """
    params = battery()
    # cheap night, very expensive evening: balance overnight, sell it back
    # into the evening peak
    periods = day([4.0] * 6 + [10.0] * 11 + [60.0] * 5 + [8.0] * 2, [1.2] * 24)
    target = params.buffer_kwh_from_soc(100.0)
    plan = solve_lp(periods, params, target)
    assert peak_soc(plan, params) == pytest.approx(100.0, abs=0.5)
    # and it does not end the horizon still full
    assert plan.end_soc_pct < 60.0


# a full charge is 4.07 kWh into the pack at 0.15 kWh/quarter (the 12 A
# planned current), i.e. 6.8 h — a cheap window shorter than that cannot
# hold the whole charge, so give this fixture 8 cheap hours
CHEAP_NIGHT = [20.0] * 2 + [2.0] * 8 + [20.0] * 14


def test_solver_picks_the_cheapest_moment_to_balance():
    params = battery()
    periods = day(CHEAP_NIGHT, [0.8] * 24)
    target = params.buffer_kwh_from_soc(100.0)
    plan = solve_lp(periods, params, target)
    charged = [p for p in plan.periods if p.grid_charge_kwh > 1e-6]
    assert charged, "balance charge must import something"
    cheap = sum(p.grid_charge_kwh for p in charged if p.price_cents_per_kwh == 2.0)
    assert cheap > 0.95 * sum(p.grid_charge_kwh for p in charged)


def test_balance_charge_is_rate_limited_to_the_planned_current():
    """A trough too short to hold the charge spills into dearer hours.

    This is why the premium rises on horizons with narrow troughs and why
    the policy will wait days for a wider one. The pack is high voltage,
    so 12 A is ~0.61 kWh/quarter and a full charge takes well under two
    hours — the trough has to be genuinely short to bind.
    """
    params = battery()
    step = current_to_period_kwh(
        min(params.planned_charge_current, params.max_charge_current),
        params.nominal_voltage,
    )
    # one hour of cheap, against a charge that needs more than that
    periods = day([20.0] * 3 + [2.0] * 1 + [20.0] * 20, [0.8] * 24)
    target = params.buffer_kwh_from_soc(100.0)
    plan = solve_lp(periods, params, target)
    cheap = sum(
        p.grid_charge_kwh for p in plan.periods if p.price_cents_per_kwh == 2.0
    )
    assert cheap == pytest.approx(4 * step / CHARGE_EFF, abs=0.01)
    # the rest had to be bought dearer, which is the point
    assert sum(p.grid_charge_kwh for p in plan.periods) > cheap + 0.01
    assert peak_soc(plan, params) == pytest.approx(100.0, abs=0.5)


def test_premium_is_below_the_reference_when_a_wide_cheap_window_exists():
    params = battery()
    periods = day(CHEAP_NIGHT, [0.8] * 24)
    target = params.buffer_kwh_from_soc(100.0)
    base = solve_lp(periods, params)
    balanced = solve_lp(periods, params, target)
    premium = balanced.total_cost_cents - base.total_cost_cents
    reference = reference_charge_cost_cents(
        target, min(p.price_cents_per_kwh for p in periods), CHARGE_EFF
    )
    # cheaper than buying the same energy outright, because the pack sells
    # it back into the 20 c hours
    assert premium < reference


def test_short_trough_costs_more_than_a_wide_one():
    """The premium is what makes the policy wait for a better night."""
    params = battery()
    target = params.buffer_kwh_from_soc(100.0)

    def premium(prices):
        periods = day(prices, [0.8] * 24)
        return (
            solve_lp(periods, params, target).total_cost_cents
            - solve_lp(periods, params).total_cost_cents
        )

    # one cheap hour: shorter than the charge needs even at 2.5 kW
    narrow = premium([20.0] * 3 + [2.0] * 1 + [20.0] * 20)
    wide = premium(CHEAP_NIGHT)
    assert narrow > wide


def test_expensive_night_arms_when_the_next_day_is_worse():
    """The case the owner asked for, decided by the objective alone.

    Night one is dear (25 c) but day two peaks at 70 c. Charging through
    the expensive night is still net profitable, so the premium comes out
    at or below zero and the policy arms with a zero allowance.
    """
    params = battery()
    night = [25.0] * 6 + [30.0] * 12 + [35.0] * 4 + [28.0] * 2
    periods = day(night, [1.0] * 24)
    periods += day([70.0] * 24, [1.0] * 24)[: 24 * 4]
    # rebuild a contiguous horizon
    start = periods[0].start
    periods = [
        Period(
            start=start + timedelta(minutes=15 * i),
            price_cents_per_kwh=p.price_cents_per_kwh,
            load_kwh=p.load_kwh,
            solar_kwh=p.solar_kwh,
            export_cents_per_kwh=p.export_cents_per_kwh,
        )
        for i, p in enumerate(periods)
    ]
    target = params.buffer_kwh_from_soc(100.0)
    base = solve_lp(periods, params)
    balanced = solve_lp(periods, params, target)
    premium = balanced.total_cost_cents - base.total_cost_cents
    assert peak_soc(balanced, params) == pytest.approx(100.0, abs=0.5)
    # the balance charge pays for itself: nothing over the plain optimum
    assert premium <= 0.5


def test_premium_is_positive_on_a_uniformly_flat_horizon():
    """No spread to recover from — balancing genuinely costs money."""
    params = battery()
    periods = day([9.0] * 24, [0.8] * 24)
    target = params.buffer_kwh_from_soc(100.0)
    base = solve_lp(periods, params)
    balanced = solve_lp(periods, params, target)
    assert balanced.total_cost_cents - base.total_cost_cents > 0.0


def test_target_is_clamped_to_capacity_not_rejected():
    params = battery()
    periods = day(*FLAT_AUGUST)
    over = params.usable_above_reserve_kwh * 1.5
    plan = solve_lp(periods, params, over)
    assert peak_soc(plan, params) == pytest.approx(100.0, abs=0.5)


def test_no_target_leaves_the_plan_identical():
    params = battery()
    periods = day(*FLAT_AUGUST)
    assert solve_lp(periods, params, None).total_cost_cents == pytest.approx(
        solve_lp(periods, params).total_cost_cents
    )
    assert solve_lp(periods, params, 0.0).total_cost_cents == pytest.approx(
        solve_lp(periods, params).total_cost_cents
    )


def test_solve_best_threads_the_target_through():
    params = battery()
    periods = day(*FLAT_AUGUST)
    target = params.buffer_kwh_from_soc(100.0)
    plan, engine = solve_best(periods, params, "lp", target)
    assert engine == "lp"
    assert peak_soc(plan, params) == pytest.approx(100.0, abs=0.5)


def test_dp_fallback_ignores_the_target_without_failing():
    params = battery()
    periods = day(*FLAT_AUGUST)
    target = params.buffer_kwh_from_soc(100.0)
    plan, engine = solve_best(periods, params, "dp", target)
    assert engine == "dp"
    # the DP cannot express it; the coordinator skips balancing on this
    # engine rather than trusting an unconstrained plan
    assert peak_soc(plan, params) < 100.0


def test_joint_solve_honours_the_target_and_stays_under_the_fuse():
    params = battery()
    periods = day(*FLAT_AUGUST)
    tank = TankParams()
    target = params.buffer_kwh_from_soc(100.0)
    plan, tank_plan = solve_joint(
        periods, params, tank, 53.0, TZ, target
    )
    assert peak_soc(plan, params) == pytest.approx(100.0, abs=0.5)
    # a sliver of slack is numerical, not a comfort breach: the 58 C
    # ceiling leaves a tight band and the solver sits right on the floor
    assert tank_plan.floor_slack_c < 0.1
    for index, period in enumerate(plan.periods):
        net_load = max(0.0, periods[index].load_kwh - periods[index].solar_kwh)
        draw = net_load + period.grid_charge_kwh
        if tank_plan.on[index]:
            draw += tank.kwh_per_quarter
        assert draw <= tank.fuse_kw * 0.25 + 1e-3


def test_balance_charge_compiles_into_a_writable_slot():
    params = battery()
    periods = day(*FLAT_AUGUST)
    target = params.buffer_kwh_from_soc(100.0)
    plan = solve_lp(periods, params, target)
    charge, _discharge = compile_slots(
        plan.periods, params, now=periods[0].start
    )
    live = [s for s in charge if s.enabled]
    assert live, "the balance charge must survive slot compilation"
    assert max(s.soc for s in live) == 100
    assert all(s.current > 0 for s in live)
