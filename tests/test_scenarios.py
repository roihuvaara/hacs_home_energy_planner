"""Named synthetic-day scenarios asserting qualitative dispatch properties.

Regression net for optimizer changes (including a future LP swap): each
scenario encodes behaviour the planner must keep — where energy moves and
why — not exact numbers, which may legitimately shift between engines.
"""

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "custom_components"))

from home_energy_planner.battery_core import (  # noqa: E402
    BatteryParams,
    Period,
    compile_slots,
    solve,
)
from home_energy_planner.solis_slots import find_cross_side_overlaps  # noqa: E402

TZ = ZoneInfo("Europe/Helsinki")
DAY_START = datetime(2026, 1, 15, 0, 0, tzinfo=TZ)


def battery(soc=18.0, reserve=18.0):
    return BatteryParams(
        capacity_kwh=5.12,
        state_of_health_pct=97.0,
        soc_pct=soc,
        reserve_soc_pct=reserve,
        max_charge_current=25,
        max_discharge_current=25,
    )


def day_periods(hourly_prices, hourly_load, hourly_solar=None, start=DAY_START):
    """Expand hourly series (cents/kWh, kWh/h) into quarter-hour Periods."""
    hourly_solar = hourly_solar or [0.0] * len(hourly_prices)
    assert len(hourly_prices) == len(hourly_load) == len(hourly_solar)
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


def solar_bell(total_kwh, hours=range(24)):
    """Daily solar total distributed over a 09-18 bell, kWh per hour."""
    weights = [max(0.0, 1.0 - abs(13 - h) / 5.0) for h in hours]
    scale = total_kwh / sum(weights)
    return [w * scale for w in weights]


def window(plan, start_hour, end_hour):
    return [
        p
        for p in plan.periods
        if start_hour <= p.start.astimezone(TZ).hour < end_hour
    ]


def charged(periods):
    return sum(p.grid_charge_kwh for p in periods)


def discharged(periods):
    return sum(p.discharge_to_load_kwh for p in periods)


# --- winter spike: cheap night, extreme morning + evening peaks -------------

WINTER_PRICES = [5.0] * 6 + [50.0] * 4 + [12.0] * 7 + [45.0] * 4 + [8.0] * 3
WINTER_LOAD = [0.8] * 24


def test_winter_spike_charges_night_discharges_morning_peak():
    plan = solve(day_periods(WINTER_PRICES, WINTER_LOAD), battery())
    assert charged(window(plan, 0, 6)) >= 2.0
    assert discharged(window(plan, 6, 10)) >= 1.5
    # never grid-charge inside a peak
    assert charged(window(plan, 6, 10)) == 0
    assert charged(window(plan, 17, 21)) == 0
    assert plan.total_cost_cents < plan.baseline_cost_cents


def test_winter_spike_recharges_midday_for_evening_peak():
    plan = solve(day_periods(WINTER_PRICES, WINTER_LOAD), battery())
    assert charged(window(plan, 10, 17)) >= 1.5
    assert discharged(window(plan, 17, 21)) >= 2.0


def test_winter_spike_compiles_to_clean_slot_tables():
    plan = solve(day_periods(WINTER_PRICES, WINTER_LOAD), battery())
    charge, discharge = compile_slots(plan.periods, battery())
    assert find_cross_side_overlaps(charge, discharge) == []
    enabled_charge = [s for s in charge if s.enabled]
    assert enabled_charge, "expected at least one charge slot"
    assert all(1 <= s.current <= 25 for s in enabled_charge)
    assert all(s.current == 0 for s in discharge if s.enabled)


# --- negative midday prices: paid to charge ---------------------------------

NEGATIVE_PRICES = [8.0] * 10 + [-1.0] * 4 + [18.0] * 10
NEGATIVE_LOAD = [0.4] * 24


def test_negative_prices_charge_at_full_rate_never_discharge():
    plan = solve(day_periods(NEGATIVE_PRICES, NEGATIVE_LOAD), battery())
    negative = window(plan, 10, 14)
    assert charged(negative) >= 1.5
    assert discharged(negative) == 0
    # every negative-price quarter grid-charges at the planned rate
    assert all(p.grid_charge_kwh > 0 for p in negative)


def test_negative_price_energy_serves_the_expensive_evening():
    plan = solve(day_periods(NEGATIVE_PRICES, NEGATIVE_LOAD), battery())
    assert discharged(window(plan, 14, 24)) >= 2.0
    assert plan.total_cost_cents < plan.baseline_cost_cents


# --- flat summer day: nothing to arbitrage, just soak solar -----------------


def test_flat_summer_day_soaks_solar_without_grid_cycling():
    prices = [9.0] * 24
    load = [0.3] * 24
    plan = solve(day_periods(prices, load, solar_bell(10.0)), battery())
    assert charged(plan.periods) == 0
    assert discharged(plan.periods) == 0
    assert plan.end_soc_pct >= 90.0
    assert plan.total_cost_cents <= plan.baseline_cost_cents + 0.01


# --- cheap two-day stretch: spread below cycle cost, stay idle --------------


def test_cheap_two_day_stretch_does_not_cycle():
    hourly = ([4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 5.0, 5.0] * 3) * 2  # 48 h
    load = [0.5] * 48
    plan = solve(day_periods(hourly, load), battery(soc=50.0))
    assert charged(plan.periods) == 0
    assert discharged(plan.periods) == 0
    assert abs(plan.total_cost_cents - plan.baseline_cost_cents) < 0.5


# --- sunny vs cloudy: solar displaces grid charging -------------------------

SOLAR_DAY_PRICES = [5.0] * 6 + [14.0] * 11 + [32.0] * 5 + [8.0] * 2
SOLAR_DAY_LOAD = [0.5] * 24


def test_sunny_day_needs_less_grid_charge_than_cloudy():
    sunny = solve(
        day_periods(SOLAR_DAY_PRICES, SOLAR_DAY_LOAD, solar_bell(18.0)),
        battery(),
    )
    cloudy = solve(
        day_periods(SOLAR_DAY_PRICES, SOLAR_DAY_LOAD, solar_bell(2.0)),
        battery(),
    )
    assert charged(sunny.periods) <= charged(cloudy.periods)
    assert sunny.total_cost_cents < cloudy.total_cost_cents
    # both still ride out the evening peak on the battery
    assert discharged(window(sunny, 17, 22)) >= 1.0
    assert discharged(window(cloudy, 17, 22)) >= 1.0


def test_all_scenarios_compile_without_cross_side_overlap():
    scenarios = [
        (WINTER_PRICES, WINTER_LOAD, None),
        (NEGATIVE_PRICES, NEGATIVE_LOAD, None),
        (SOLAR_DAY_PRICES, SOLAR_DAY_LOAD, solar_bell(18.0)),
        (SOLAR_DAY_PRICES, SOLAR_DAY_LOAD, solar_bell(2.0)),
    ]
    for prices, load, solar in scenarios:
        params = battery()
        plan = solve(day_periods(prices, load, solar), params)
        charge, discharge = compile_slots(plan.periods, params)
        assert find_cross_side_overlaps(charge, discharge) == []
        assert len(charge) == 6 and len(discharge) == 6
