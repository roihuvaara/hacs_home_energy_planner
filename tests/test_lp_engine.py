"""LP engine (HiGHS) vs DP cross-check (ADR 0009: DP validates the LP)."""

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "custom_components"))

pytest.importorskip("highspy")

from home_energy_planner.battery_core import (  # noqa: E402
    CYCLE_COST_CENTS_PER_KWH,
    DISCHARGE_EFF,
    BatteryParams,
    Period,
    compile_slots,
    solve,
)
from home_energy_planner.milp_core import solve_best, solve_lp  # noqa: E402
from home_energy_planner.solis_slots import find_cross_side_overlaps  # noqa: E402

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


def day(hourly_prices, hourly_load, hourly_solar=None, export_cents=0.0):
    hourly_solar = hourly_solar or [0.0] * len(hourly_prices)
    start = datetime(2026, 1, 15, 0, 0, tzinfo=TZ)
    periods = []
    for hour, (price, load, solar) in enumerate(
        zip(hourly_prices, hourly_load, hourly_solar)
    ):
        for quarter in range(4):
            periods.append(
                Period(
                    start=start + timedelta(hours=hour, minutes=15 * quarter),
                    price_cents_per_kwh=price,
                    load_kwh=load / 4.0,
                    solar_kwh=solar / 4.0,
                    export_cents_per_kwh=export_cents,
                )
            )
    return periods


def objective(plan, params, periods):
    """Terminal-adjusted objective: raw totals distort when end SOCs differ."""
    min_price = min(p.price_cents_per_kwh for p in periods)
    terminal = (
        max(0.0, DISCHARGE_EFF * (min_price - CYCLE_COST_CENTS_PER_KWH)) + 1e-3
    )
    return plan.total_cost_cents - terminal * params.buffer_kwh_from_soc(
        plan.end_soc_pct
    )


def bell(total):
    weights = [max(0.0, 1.0 - abs(13 - h) / 5.0) for h in range(24)]
    scale = total / sum(weights)
    return [w * scale for w in weights]


SCENARIOS = {
    "winter_spike": (
        [5.0] * 6 + [50.0] * 4 + [12.0] * 7 + [45.0] * 4 + [8.0] * 3,
        [0.8] * 24,
        None,
        18.0,
    ),
    "negative_midday": ([8.0] * 10 + [-1.0] * 4 + [18.0] * 10, [0.4] * 24, None, 18.0),
    "flat_summer": ([9.0] * 24, [0.3] * 24, bell(10.0), 18.0),
    "cheap_flat": ([5.0] * 24, [0.5] * 24, None, 50.0),
    "sunny_arbitrage": (
        [5.0] * 6 + [14.0] * 11 + [32.0] * 5 + [8.0] * 2,
        [0.5] * 24,
        bell(18.0),
        18.0,
    ),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_lp_matches_or_beats_dp_cost(name):
    prices, load, solar, soc = SCENARIOS[name]
    periods = day(prices, load, solar)
    params = battery(soc=soc)
    dp = solve(periods, params)
    lp = solve_lp(periods, params)
    assert lp.baseline_cost_cents == pytest.approx(dp.baseline_cost_cents, abs=0.02)
    # the LP relaxes the DP's 0.1 kWh quantization and forced solar
    # absorption, so it can only be equal or slightly cheaper — compared on
    # the terminal-adjusted objective so end-SOC differences don't distort
    lp_obj = objective(lp, params, periods)
    dp_obj = objective(dp, params, periods)
    assert lp_obj <= dp_obj + 0.02
    # ... and the DP must stay within its quantization band (0.05 kWh
    # states, integer charge steps), or one of the engines is wrong.
    # 4 % calibrated on sunny_arbitrage (3.8 % under the terminal-adjusted
    # metric — the raw-total comparison used to hide part of the gap)
    gap = dp_obj - lp_obj
    assert gap <= max(2.0, 0.04 * dp.baseline_cost_cents), (
        f"{name}: dp={dp_obj} lp={lp_obj}"
    )


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_lp_plans_respect_physics_and_compile(name):
    prices, load, solar, soc = SCENARIOS[name]
    periods = day(prices, load, solar)
    params = battery(soc=soc)
    plan = solve_lp(periods, params)
    capacity = params.usable_above_reserve_kwh
    for p in plan.periods:
        assert -1e-6 <= p.buffer_end_kwh <= capacity + 1e-6
        assert p.grid_charge_kwh >= 0 and p.discharge_to_load_kwh >= 0
        assert p.grid_import_kwh >= -1e-6
    charge, discharge = compile_slots(plan.periods, params)
    assert find_cross_side_overlaps(charge, discharge) == []


def test_solve_best_reports_engine_and_falls_back():
    periods = day([5.0] * 12 + [40.0] * 12, [0.4] * 24)
    plan, engine = solve_best(periods, battery(), "lp")
    assert engine == "lp"
    plan_dp, engine_dp = solve_best(periods, battery(), "dp")
    assert engine_dp == "dp"
    assert plan.total_cost_cents <= plan_dp.total_cost_cents + 0.02


def test_lp_negative_prices_charge_full_rate():
    prices = [8.0] * 10 + [-1.0] * 4 + [18.0] * 10
    plan = solve_lp(day(prices, [0.4] * 24), battery())
    negative = [p for p in plan.periods if p.price_cents_per_kwh < 0]
    assert all(p.grid_charge_kwh > 0 for p in negative)
    assert sum(p.discharge_to_load_kwh for p in negative) == 0


def joint(periods, soc=18.0, tank=None, temp0=55.0):
    from home_energy_planner.milp_core import TankParams, solve_joint

    return solve_joint(periods, battery(soc=soc), tank or TankParams(), temp0)


def test_joint_tank_heats_in_cheap_hours_and_respects_min_up():
    from home_energy_planner.milp_core import TankParams

    prices = [5.0] * 6 + [50.0] * 4 + [12.0] * 7 + [45.0] * 4 + [8.0] * 3
    periods = day(prices, [0.8] * 24)
    tank = TankParams()
    plan, tank_plan = joint(periods, tank=tank, temp0=52.0)

    assert tank_plan.windows, "tank must be scheduled"
    assert all(
        end - start >= tank.min_run_quarters for start, end in tank_plan.windows
    )
    run_prices = [
        periods[t].price_cents_per_kwh
        for start, end in tank_plan.windows
        for t in range(start, end)
    ]
    assert max(run_prices) <= 12.0, run_prices
    assert plan.total_cost_cents < plan.baseline_cost_cents


def test_joint_temp_trajectory_physics():
    from home_energy_planner.milp_core import TankParams

    tank = TankParams()
    prices = [5.0] * 8 + [20.0] * 16
    periods = day(prices, [0.4] * 24)
    _plan, tank_plan = joint(periods, tank=tank, temp0=55.0)

    assert all(tank.min_c - 0.01 <= T <= tank.max_c + 0.01 for T in tank_plan.temp_c)
    assert tank_plan.floor_slack_c == 0.0
    for t in range(1, len(tank_plan.temp_c)):
        prev, cur = tank_plan.temp_c[t - 1], tank_plan.temp_c[t]
        if not tank_plan.on[t]:
            # off: decays (loss + draws), never rises
            assert cur <= prev + 0.01
        else:
            # on: net rise bounded by the heat rate
            assert cur <= prev + tank.gain_c_per_quarter + 0.01


def test_joint_hot_tank_coasts_through_expensive_evening():
    from home_energy_planner.milp_core import TankParams

    tank = TankParams()
    prices = [30.0] * 20 + [5.0] * 4  # expensive until late night
    periods = day(prices, [0.4] * 24)
    _plan, tank_plan = joint(periods, tank=tank, temp0=64.0)
    # plenty of stored heat: no runs during the expensive stretch
    expensive_on = [
        on
        for on, p in zip(tank_plan.on, periods)
        if p.price_cents_per_kwh == 30.0 and on
    ]
    assert expensive_on == []


def test_joint_cold_tank_heats_first_via_slack():
    from home_energy_planner.milp_core import TankParams

    tank = TankParams()
    prices = [20.0] * 24
    periods = day(prices, [0.4] * 24)
    _plan, tank_plan = joint(periods, tank=tank, temp0=40.0)
    # below-floor start is solvable and the first window starts immediately
    assert tank_plan.windows and tank_plan.windows[0][0] == 0
    # slack covers the initial deficit only; temp recovers to the floor
    assert tank_plan.temp_c[-1] >= tank.min_c - 0.01


def test_joint_start_shortfall_consolidates_runs():
    from home_energy_planner.milp_core import TankParams

    # the start transient wastes heat, so runs get longer on average
    # (window COUNT can legitimately grow: wasted starts may force extra
    # runs to hold the floor — length per run is the consolidation signal)
    prices = ([5.0] * 4 + [20.0] * 8) * 2
    periods = day(prices, [0.4] * 24)
    no_shortfall = TankParams(start_shortfall_c=0.0, daily_draw_kwh=3.0)
    shortfall = TankParams(daily_draw_kwh=3.0)  # measured 4.4 C
    _p1, plan_free = joint(periods, tank=no_shortfall, temp0=55.0)
    _p2, plan_costly = joint(periods, tank=shortfall, temp0=55.0)

    def avg_len(tank_plan):
        lengths = [end - start for start, end in tank_plan.windows]
        return sum(lengths) / len(lengths) if lengths else 0.0

    assert avg_len(plan_costly) >= avg_len(plan_free)


def test_joint_fuse_cap_limits_simultaneous_load():
    from home_energy_planner.milp_core import TankParams

    # tiny fuse: tank (3.3 kW) + house (1.6 kW) leaves no room for battery
    # charging during tank runs
    prices = [5.0] * 8 + [40.0] * 16
    periods = day(prices, [1.6] * 24)
    tank = TankParams(fuse_kw=5.0)
    plan, tank_plan = joint(periods, tank=tank, temp0=51.0)
    on_quarters = {t for start, end in tank_plan.windows for t in range(start, end)}
    # fuse headroom during a tank run: 5 kW - 1.6 house - 3.3 tank = 0.1 kW
    headroom_kwh = (tank.fuse_kw - 1.6 - tank.power_kw) * 0.25
    for t in on_quarters:
        assert plan.periods[t].grid_charge_kwh <= headroom_kwh + 1e-6, (
            t,
            plan.periods[t].grid_charge_kwh,
        )


def test_joint_surplus_feeds_tank_as_dump_load():
    from home_energy_planner.milp_core import TankParams

    tank = TankParams()
    # flat price, plenty of solar, tiny export value: heating on surplus
    # is nearly free, so the tank is driven up as a dump load
    prices = [9.0] * 24
    periods = day(prices, [0.2] * 24, bell(14.0), export_cents=0.5)
    _plan, tank_plan = joint(periods, tank=tank, temp0=51.0)
    assert sum(tank_plan.surplus_kwh) > 1.0
    assert max(tank_plan.temp_c) > 60.0
    # high export value attenuates the dump: selling beats storing heat
    _plan2, rich_export = joint(
        day(prices, [0.2] * 24, bell(14.0), export_cents=12.0),
        tank=tank,
        temp0=51.0,
    )
    assert sum(rich_export.surplus_kwh) <= sum(tank_plan.surplus_kwh) + 1e-6
    assert max(rich_export.temp_c) <= max(tank_plan.temp_c) + 0.01


def test_joint_solve_time_stays_fast():
    import time

    from home_energy_planner.milp_core import TankParams

    prices = ([5.0, 8.0, 12.0, 7.0, 30.0, 25.0] * 8)[:48]
    periods = day(prices, [0.5] * 48, None)
    started = time.monotonic()
    joint(periods, tank=TankParams(), temp0=55.0)
    # runs in an executor once per 15-min tick; 5 s is ample headroom
    assert time.monotonic() - started < 5.0


# --- export economics (LP only; DP stays export-blind by design) -------------


def test_lp_exports_when_export_beats_storage():
    # flat 9 c all-in, export 8 c: storing a surplus kWh is worth
    # eff_c*eff_d*(9-4) ~= 4.3 c later vs 8 c exported now -> export wins
    prices = [9.0] * 24
    plan = solve_lp(day(prices, [0.2] * 24, bell(12.0), export_cents=8.0), battery())
    assert sum(p.export_kwh for p in plan.periods) > 5.0
    assert plan.export_revenue_cents > 40.0


def test_lp_still_absorbs_for_evening_spike_on_modest_export():
    # export 2 c vs avoided 32 c evening import: absorb wins
    prices = [9.0] * 17 + [32.0] * 5 + [9.0] * 2
    plan = solve_lp(day(prices, [0.4] * 24, bell(10.0), export_cents=2.0), battery())
    evening = [p for p in plan.periods if p.price_cents_per_kwh == 32.0]
    assert sum(p.discharge_to_load_kwh for p in evening) >= 1.5
    # only surplus beyond battery capacity/need leaks to export
    total_surplus = sum(
        max(0.0, p.solar_kwh - p.load_kwh)
        for p in day(prices, [0.4] * 24, bell(10.0))
    )
    assert sum(p.export_kwh for p in plan.periods) < total_surplus


def test_lp_negative_export_price_absorbs_everything_possible():
    # paying to export: every absorbable surplus kWh goes into the battery
    prices = [9.0] * 24
    plan = solve_lp(day(prices, [0.2] * 24, bell(4.0), export_cents=-1.0), battery())
    exported = sum(p.export_kwh for p in plan.periods)
    assert exported == pytest.approx(0.0, abs=0.05)


def test_lp_export_accounting_consistent():
    prices = [9.0] * 12 + [15.0] * 12
    periods = day(prices, [0.3] * 24, bell(8.0), export_cents=3.0)
    plan = solve_lp(periods, battery())
    assert all(p.export_kwh >= 0 for p in plan.periods)
    revenue = sum(p.export_kwh * 3.0 for p in plan.periods)
    assert plan.export_revenue_cents == pytest.approx(revenue, abs=0.1)
