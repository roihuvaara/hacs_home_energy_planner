"""Battery-to-grid selling: the gate, the self-use comparison, the slots.

The economics only exist because the purchase contract stopped tracking
spot (fixed 7.39 c/kWh energy, 2026-09-15..2027-03-15) while the sell
side still does (spot - 0.3 c/kWh, VAT 0). Buy price flat, sell price
spiky -> the spikes are margin. Numbers below are the owner's real
2026-09-22 horizon.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "custom_components"))

pytest.importorskip("highspy")

from home_energy_planner.battery_core import (  # noqa: E402
    BatteryParams,
    Period,
    compile_slots,
)
from home_energy_planner.milp_core import solve_lp  # noqa: E402

NIGHT = 10.51  # all-in buy, fixed contract + night transfer
DAY = 12.50  # all-in buy, fixed contract + day transfer
BATTERY = BatteryParams(
    capacity_kwh=5.12,
    state_of_health_pct=100.0,
    soc_pct=100.0,
    reserve_soc_pct=18.0,
    max_charge_current=25,
    max_discharge_current=25,
    planned_charge_current=12,
)
START = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)


def horizon(prices, exports, *, load=0.2, solar=0.0):
    return [
        Period(
            start=START + timedelta(minutes=15 * i),
            price_cents_per_kwh=p,
            load_kwh=load,
            solar_kwh=solar,
            export_cents_per_kwh=e,
        )
        for i, (p, e) in enumerate(zip(prices, exports))
    ]


def sold(plan):
    return sum(p.export_from_battery_kwh for p in plan.periods)


def test_no_gate_means_no_selling():
    """Default stays put: an unset gate keeps the old load-capped model."""
    prices = [DAY] * 16
    exports = [34.7] * 16  # a spike the planner is not allowed to touch
    plan = solve_lp(horizon(prices, exports), BATTERY)
    assert sold(plan) == 0.0


def test_spike_below_the_gate_is_left_alone():
    prices = [NIGHT] * 16
    exports = [18.0] * 16  # clears break-even, short of 2x the buy price
    plan = solve_lp(horizon(prices, exports), BATTERY, None, NIGHT * 2.0)
    assert sold(plan) == 0.0


def test_sells_into_a_clear_spike():
    prices = [NIGHT] * 16
    exports = [3.0] * 8 + [34.7] * 8
    plan = solve_lp(horizon(prices, exports), BATTERY, None, NIGHT * 2.0)
    assert sold(plan) > 0.0
    # every sold kWh lands in the spike, none in the cheap half
    assert all(p.export_from_battery_kwh == 0 for p in plan.periods[:8])
    assert all(p.action == "export" for p in plan.periods[8:] if p.export_from_battery_kwh > 0)


def test_only_spare_inverter_capacity_reaches_the_grid():
    """One meter: the house is served first, the leftover is the sale.

    Discharge and load net out inside the interval, so what can be sold
    is the inverter's rate MINUS the load, not its rate. At 25 A / 50 V
    the pack delivers 0.296 kWh per quarter; a house pulling more than
    that exports nothing however high the price goes.
    """
    step = 25 * 50 * 0.25 / 1000 * (0.9**0.5)  # ~0.296 kWh delivered
    exports = [34.7] * 8

    # load above the discharge rate: nothing to sell, spike or no spike
    hungry = solve_lp(
        horizon([DAY] * 8, exports, load=step + 0.1), BATTERY, None, DAY * 2.0
    )
    assert sold(hungry) == 0.0

    # load well under it: the spare capacity goes to the grid
    quiet = solve_lp(
        horizon([DAY] * 8, exports, load=0.05), BATTERY, None, DAY * 2.0
    )
    assert sold(quiet) > 0.0
    for period in quiet.periods:
        if period.export_from_battery_kwh > 0:
            assert period.export_from_battery_kwh <= step - 0.05 + 1e-6


def test_never_buys_and_sells_in_the_same_quarter():
    """Importing while exporting is two numbers that cancel, not a trade."""
    prices = [NIGHT] * 8
    exports = [34.7] * 8
    plan = solve_lp(horizon(prices, exports, load=0.05), BATTERY, None, NIGHT * 2.0)
    for period in plan.periods:
        assert not (period.export_from_battery_kwh > 0 and period.grid_charge_kwh > 0)


def test_the_kwh_goes_wherever_it_pays_most():
    """Self-use vs sell is settled by price alone, quarter by quarter.

    Sell price is a flat 11.50 against a buy price that steps 12.50 ->
    10.51. The same stored kWh should therefore cover load in the dear
    quarters (saving 12.50 beats earning 11.50) and be sold in the cheap
    ones (earning 11.50 beats saving 10.51) - even though selling there
    means importing at 10.51 to cover the load, which is +0.99 c/kWh, not
    the losing swap it looks like. The cycle cost is identical on both
    sides and so cancels out of the comparison, which is the whole point
    of keeping the gate a permission rather than a tax.
    """
    prices = [DAY] * 4 + [NIGHT] * 4
    exports = [11.5] * 8
    # small load, so there is spare inverter capacity to sell at all -
    # with a big load the house absorbs everything and the question does
    # not arise (see test_only_spare_inverter_capacity_reaches_the_grid)
    plan = solve_lp(horizon(prices, exports, load=0.05), BATTERY, None, 1.0)
    dear, cheap = plan.periods[:4], plan.periods[4:]
    # The claim is about where the LOAD's energy comes from. Spare
    # capacity is sold in both halves - it would otherwise just sit - so
    # "exports nothing" is not the dear-half signature; "serves the house
    # itself" is.
    assert all(p.discharge_to_load_kwh > 0 for p in dear), "12.50 saved beats 11.50 earned"
    assert all(p.discharge_to_load_kwh == 0 for p in cheap), "11.50 earned beats 10.51 saved"
    assert all(p.export_from_battery_kwh > 0 for p in cheap)
    # never sell below what the same quarter is paying to import
    for period in plan.periods:
        if period.export_from_battery_kwh > 0 and period.grid_import_kwh > 0:
            assert period.export_cents_per_kwh >= period.price_cents_per_kwh


def test_discharge_rate_is_shared_with_self_use():
    """One inverter: selling and covering load cannot both run flat out."""
    prices = [DAY] * 8
    exports = [34.7] * 8
    plan = solve_lp(horizon(prices, exports, load=5.0), BATTERY, None, DAY * 2.0)
    step = 25 * 50 * 0.25 / 1000  # kWh/quarter at 25 A
    for period in plan.periods:
        drain = (
            period.discharge_to_load_kwh + period.export_from_battery_kwh
        ) / (0.9**0.5)
        assert drain <= step + 1e-6


def test_round_trip_through_the_grid_on_a_real_spike():
    """Buy at the flat night price, sell into 2026-09-22's 19:00 peak."""
    empty = BatteryParams(
        capacity_kwh=5.12,
        state_of_health_pct=100.0,
        soc_pct=18.0,
        reserve_soc_pct=18.0,
        max_charge_current=25,
        max_discharge_current=25,
        planned_charge_current=12,
    )
    prices = [NIGHT] * 48 + [DAY] * 8
    exports = [2.2] * 48 + [34.7] * 8
    plan = solve_lp(horizon(prices, exports, load=0.2), empty, None, NIGHT * 2.0)
    assert sum(p.grid_charge_kwh for p in plan.periods) > 0.0
    assert sold(plan) > 0.0
    # and it is worth doing: cheaper than the no-battery baseline
    assert plan.total_cost_cents < plan.baseline_cost_cents


def test_export_window_compiles_to_a_forced_discharge_slot():
    prices = [DAY] * 8
    exports = [34.7] * 8
    plan = solve_lp(horizon(prices, exports, load=0.05), BATTERY, None, DAY * 2.0)
    now = START.astimezone(timezone.utc)
    _charge, discharge = compile_slots(plan.periods, BATTERY, now=now)
    forced = [s for s in discharge if s.enabled and s.current > 0]
    assert forced, "an export window must produce a current-carrying slot"
    assert forced[0].soc < 100  # stops at the planned floor, not "empty"


@pytest.mark.parametrize("multiple", [1.6, 2.0, 3.0])
def test_higher_gate_never_sells_more(multiple):
    prices = [NIGHT] * 16
    exports = [12.0] * 4 + [18.0] * 4 + [22.0] * 4 + [34.7] * 4
    plan = solve_lp(horizon(prices, exports), BATTERY, None, NIGHT * multiple)
    gate = NIGHT * multiple
    for period in plan.periods:
        if period.export_from_battery_kwh > 0:
            assert period.export_cents_per_kwh >= gate


def test_a_small_spike_now_does_not_burn_the_cycle_a_big_one_needs():
    """One pack, one cycle, two spikes: it waits for the one that pays.

    Grid charging is disabled here so the stored kWh really is scarce —
    with charging available the planner sells into BOTH spikes and is
    right to, since it refills at the flat night price in between and
    each window is rate-capped anyway. What this pins is the scarce
    case: a 21.5 c/kWh window tonight against a 34.7 c/kWh window
    tomorrow evening, big enough to take the whole pack.

    Nothing in the model counts cycles, and it does not need to. A kWh
    can only be sold once, so the objective reaches for the richer
    window by itself; the cycle cost is paid once either way and cancels
    out. The real limit is the horizon edge, not the arithmetic — a
    spike beyond the last published day-ahead price cannot be waited for
    because it is not known yet.
    """
    no_charging = BatteryParams(
        capacity_kwh=5.12,
        state_of_health_pct=100.0,
        soc_pct=100.0,
        reserve_soc_pct=18.0,
        max_charge_current=0,
        max_discharge_current=25,
        planned_charge_current=0,
    )
    prices = [NIGHT] * 96
    exports = [2.0] * 96
    exports[8:16] = [21.5] * 8  # tonight: clears the gate, modestly
    # tomorrow evening, long enough to take the whole pack at the spare
    # rate that is left after the load
    exports[64:96] = [34.7] * 32
    plan = solve_lp(horizon(prices, exports, load=0.05), no_charging, None, NIGHT * 2.0)
    early = sum(p.export_from_battery_kwh for p in plan.periods[8:16])
    late = sum(p.export_from_battery_kwh for p in plan.periods[64:96])
    assert late > 0
    assert early == 0, "sold into the small spike and had nothing left for the big one"
