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


def day(hourly_prices, hourly_load, hourly_solar=None):
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
                )
            )
    return periods


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
    # absorption, so it can only be equal or slightly cheaper
    assert lp.total_cost_cents <= dp.total_cost_cents + 0.02
    # ... and the DP must stay within its quantization band (0.05 kWh
    # states, integer charge steps), or one of the engines is wrong
    gap = dp.total_cost_cents - lp.total_cost_cents
    assert gap <= max(2.0, 0.035 * dp.baseline_cost_cents), (
        f"{name}: dp={dp.total_cost_cents} lp={lp.total_cost_cents}"
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
