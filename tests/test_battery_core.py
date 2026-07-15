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


def battery(soc=50.0, reserve=18.0):
    return BatteryParams(
        capacity_kwh=5.12,
        state_of_health_pct=97.0,
        soc_pct=soc,
        reserve_soc_pct=reserve,
        max_charge_current=25,
        max_discharge_current=25,
    )


def make_periods(prices, load=0.3, solar=0.0, start_hour=0):
    start = datetime(2026, 7, 6, start_hour, 0, tzinfo=TZ)
    return [
        Period(
            start=start + timedelta(minutes=15 * i),
            price_cents_per_kwh=price,
            load_kwh=load,
            solar_kwh=solar,
        )
        for i, price in enumerate(prices)
    ]


def test_charges_cheap_discharges_expensive():
    # 2h cheap night, then 2h very expensive morning
    prices = [3.0] * 8 + [40.0] * 8
    plan = solve(make_periods(prices), battery(soc=18.0))
    cheap = plan.periods[:8]
    expensive = plan.periods[8:]
    assert sum(p.grid_charge_kwh for p in cheap) > 0.5
    assert sum(p.discharge_to_load_kwh for p in expensive) > 0.5
    assert plan.total_cost_cents < plan.baseline_cost_cents


def test_no_cycling_when_flat_prices():
    prices = [10.0] * 16
    plan = solve(make_periods(prices), battery(soc=50.0))
    assert sum(p.grid_charge_kwh for p in plan.periods) == 0
    # cycle cost makes discharging not worth it on flat prices
    assert sum(p.discharge_to_load_kwh for p in plan.periods) == 0


def test_small_spread_not_worth_cycle_cost():
    # spread below round-trip + cycle cost threshold: no grid charging
    prices = [10.0] * 8 + [12.0] * 8
    plan = solve(make_periods(prices), battery(soc=18.0))
    assert sum(p.grid_charge_kwh for p in plan.periods) == 0


def test_solar_surplus_charges_battery_free():
    prices = [10.0] * 4 + [30.0] * 4
    periods = make_periods(prices, load=0.1, solar=0.0, start_hour=10)
    periods = periods[:4] + [
        Period(p.start, p.price_cents_per_kwh, 0.1, 0.6) for p in periods[:4]
    ]
    # first 4 periods have surplus solar; battery should end higher than start
    plan = solve(
        [
            Period(p.start, p.price_cents_per_kwh, p.load_kwh, p.solar_kwh)
            for p in periods
        ],
        battery(soc=18.0),
    )
    assert plan.end_soc_pct >= 18.0


def test_respects_reserve_floor():
    prices = [50.0] * 8
    plan = solve(make_periods(prices), battery(soc=30.0, reserve=18.0))
    assert all(p.buffer_end_kwh >= -1e-9 for p in plan.periods)
    assert plan.end_soc_pct >= 18.0


def test_compiled_slots_never_overlap_cross_side():
    # alternating cheap/expensive pattern across 33h to force many windows
    prices = ([3.0] * 8 + [35.0] * 8) * 8 + [3.0] * 4
    plan = solve(make_periods(prices), battery(soc=18.0))
    charge, discharge = compile_slots(plan.periods, battery(soc=18.0))
    assert find_cross_side_overlaps(charge, discharge) == []
    assert len(charge) == 6 and len(discharge) == 6
    enabled_charge = [s for s in charge if s.enabled]
    assert all(1 <= s.current <= 25 for s in enabled_charge)
    enabled_hold = [s for s in discharge if s.enabled]
    assert all(s.current == 0 for s in enabled_hold)


def test_no_hold_slots_for_empty_battery():
    # empty buffer + spread too small to charge: the DP labels ties "hold",
    # which must not become 0 A slots (observed noise: 12:00-22:00 hold at
    # an empty battery)
    prices = [10.0] * 24 + [12.0] * 24
    plan = solve(make_periods(prices), battery(soc=18.0))
    charge, discharge = compile_slots(plan.periods, battery(soc=18.0))
    assert not any(s.enabled for s in discharge)
    assert not any(s.enabled for s in charge)


def test_hold_windows_protect_before_expensive():
    prices = [8.0] * 8 + [40.0] * 8
    plan = solve(make_periods(prices), battery(soc=80.0))
    # cheap periods should hold (no discharge) so the buffer survives
    cheap = plan.periods[:8]
    assert sum(p.discharge_to_load_kwh for p in cheap) == 0
    charge, discharge = compile_slots(plan.periods, battery(soc=80.0))
    assert any(s.enabled for s in discharge)


# --- compile_slots device-reality handling (live bugs 2026-07-15) ------------

from home_energy_planner.battery_core import PeriodPlan  # noqa: E402

UTC = ZoneInfo("UTC")


def hold_plans(start_utc, quarters, price=8.0, buffer_kwh=1.0):
    return [
        PeriodPlan(
            start=start_utc + timedelta(minutes=15 * i),
            action="hold",
            buffer_start_kwh=buffer_kwh,
            buffer_end_kwh=buffer_kwh,
            grid_charge_kwh=0.0,
            discharge_to_load_kwh=0.0,
            grid_import_kwh=0.3,
            price_cents_per_kwh=price,
        )
        for i in range(quarters)
    ]


def test_compile_slots_renders_local_wall_clock():
    # periods carry UTC datetimes; the inverter runs local wall clock.
    # 19:00 UTC = 22:00 Helsinki — the written window must say 22:00.
    plans = hold_plans(datetime(2026, 7, 15, 19, 0, tzinfo=UTC), 8)
    now = datetime(2026, 7, 15, 19, 40, tzinfo=TZ)
    _charge, discharge = compile_slots(plans, battery(soc=40.0), now=now)
    enabled = [s for s in discharge if s.enabled]
    assert [s.time for s in enabled] == ["22:00-00:00"]


def test_compile_slots_defers_window_whose_wall_clock_fires_early():
    # hold intended TOMORROW 18:15-01:00 local; Solis slots recur daily, so
    # written today it would also fire tonight, fighting tonight's plan.
    plans = hold_plans(datetime(2026, 7, 16, 15, 15, tzinfo=UTC), 27)
    tonight = datetime(2026, 7, 15, 19, 40, tzinfo=TZ)
    _charge, discharge = compile_slots(plans, battery(soc=40.0), now=tonight)
    assert not any(s.enabled for s in discharge)

    # once today's colliding occurrence has passed, the window is written
    after_midnight = datetime(2026, 7, 16, 2, 0, tzinfo=TZ)
    _charge, discharge = compile_slots(
        plans, battery(soc=40.0), now=after_midnight
    )
    enabled = [s for s in discharge if s.enabled]
    assert [s.time for s in enabled] == ["18:15-01:00"]


def test_compile_slots_without_now_keeps_legacy_behaviour():
    plans = hold_plans(datetime(2026, 7, 16, 15, 15, tzinfo=UTC), 27)
    _charge, discharge = compile_slots(plans, battery(soc=40.0))
    assert any(s.enabled for s in discharge)
