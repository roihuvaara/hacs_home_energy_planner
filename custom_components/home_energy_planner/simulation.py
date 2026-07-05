"""On-demand plan simulation through the real solve/compile path.

Backs the ``home_energy_planner.simulate_plan`` service: price series and
load/solar/battery overrides in, dispatch plan plus compiled slot tables
out. No coordinator refresh, no device writes — anything not overridden
falls back to the same live sources the battery coordinator uses (price
horizon, 7-day load baseline, solar forecast, battery state).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .battery_core import PERIOD_MINUTES, Period, compile_slots, solve
from .battery_coordinator import BatteryCoordinator
from .coordinator import PricingCoordinator


def _quarter_bucket(ts: datetime, tz) -> int:
    local = ts.astimezone(tz)
    return local.hour * 60 + (local.minute // PERIOD_MINUTES) * PERIOD_MINUTES


def _override_series(value: Any, count: int, name: str) -> list[float] | None:
    """A scalar becomes a constant series; a list is padded with its last value."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)] * count
    try:
        series = [float(item) for item in value]
    except (TypeError, ValueError) as err:
        raise HomeAssistantError(f"{name} must be a number or list of numbers") from err
    if not series:
        raise HomeAssistantError(f"{name} list must not be empty")
    if len(series) < count:
        series = series + [series[-1]] * (count - len(series))
    return series[:count]


async def async_simulate_plan(
    pricing: PricingCoordinator,
    battery: BatteryCoordinator,
    data: dict[str, Any],
) -> dict[str, Any]:
    hass = battery.hass
    now = dt_util.now()

    prices = data.get("prices")
    if prices is not None:
        if isinstance(prices, (int, float)) or not prices:
            raise HomeAssistantError("prices must be a non-empty list of numbers")
        price_series = _override_series(list(prices), len(prices), "prices")
        start_raw = data.get("start")
        start = dt_util.parse_datetime(str(start_raw)) if start_raw else None
        if start_raw and start is None:
            raise HomeAssistantError(f"Could not parse start '{start_raw}'")
        if start is None:
            floored = now.replace(second=0, microsecond=0)
            start = floored - timedelta(minutes=floored.minute % PERIOD_MINUTES)
        start = dt_util.as_local(start)
        starts = [
            start + timedelta(minutes=PERIOD_MINUTES * index)
            for index in range(len(price_series))
        ]
    else:
        horizon = pricing.data
        if horizon is None or not horizon.periods:
            raise HomeAssistantError(
                "No live price horizon available; pass an explicit price series"
            )
        starts = [p.start for p in horizon.periods]
        price_series = [p.all_in_cents_per_kwh for p in horizon.periods]

    count = len(starts)
    load_series = _override_series(data.get("load"), count, "load")
    if load_series is None:
        baseline = await battery.load_baseline_kwh_by_quarter(now)
        load_series = [
            baseline.get(_quarter_bucket(ts, now.tzinfo), 0.15) for ts in starts
        ]
    solar_series = _override_series(data.get("solar"), count, "solar")
    if solar_series is None:
        solar_series = await battery.async_solar_series_kwh(starts, now)

    params = battery.battery_params(
        {
            "capacity_kwh": data.get("capacity_kwh"),
            "state_of_health_pct": data.get("soh_pct"),
            "soc_pct": data.get("soc_pct"),
            "reserve_soc_pct": data.get("reserve_soc_pct"),
            "max_charge_current": data.get("max_charge_current"),
            "max_discharge_current": data.get("max_discharge_current"),
        }
    )

    periods = [
        Period(
            start=ts,
            price_cents_per_kwh=price,
            load_kwh=round(load, 4),
            solar_kwh=round(solar, 4),
        )
        for ts, price, load, solar in zip(starts, price_series, load_series, solar_series)
    ]

    plan = await hass.async_add_executor_job(solve, periods, params)
    charge_slots, discharge_slots = compile_slots(plan.periods, params)

    return {
        "summary": {
            "periods": count,
            "start": starts[0].isoformat(),
            "end": (starts[-1] + timedelta(minutes=PERIOD_MINUTES)).isoformat(),
            "total_cost_cents": plan.total_cost_cents,
            "baseline_cost_cents": plan.baseline_cost_cents,
            "savings_cents": round(plan.baseline_cost_cents - plan.total_cost_cents, 2),
            "end_soc_pct": plan.end_soc_pct,
            "battery": {
                "soc_pct": params.soc_pct,
                "reserve_soc_pct": params.reserve_soc_pct,
                "capacity_kwh": params.capacity_kwh,
                "state_of_health_pct": params.state_of_health_pct,
                "max_charge_current": params.max_charge_current,
                "max_discharge_current": params.max_discharge_current,
            },
        },
        "charge_slots": [slot.as_dict() for slot in charge_slots],
        "discharge_slots": [slot.as_dict() for slot in discharge_slots],
        "periods": [
            {
                "start": p.start.isoformat(),
                "action": p.action,
                "price": p.price_cents_per_kwh,
                "load_kwh": periods[index].load_kwh,
                "solar_kwh": periods[index].solar_kwh,
                "grid_charge_kwh": p.grid_charge_kwh,
                "discharge_kwh": p.discharge_to_load_kwh,
                "grid_import_kwh": p.grid_import_kwh,
                "buffer_end_kwh": p.buffer_end_kwh,
            }
            for index, p in enumerate(plan.periods)
        ],
    }
