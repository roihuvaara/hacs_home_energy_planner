"""Backtest harness: replay recorded days through the real planner.

Backs the ``home_energy_planner.backtest`` service (ADR 0009 phase 1b
pre-control gate). For each of the last N full local days it rebuilds the
quarter-hour all-in price series from historical Nord Pool data, takes
recorded load and solar from long-term recorder statistics, and solves
the dispatch problem with perfect foresight of that day. Battery state
chains from day to day. The report compares, per day:

- baseline: net load bought from the grid with no battery dispatch
- planned: the planner's cost (imports + cycle cost) for the same day
- actual: recorded grid import x price, when a grid import entity is set

Perfect foresight makes ``planned`` an upper bound on achievable savings,
not a promise; it is the gate evidence for flipping battery_mode to
control, read together with the observe-mode plan log.
"""

from __future__ import annotations

from datetime import date as date_type, datetime, timedelta
from typing import Any

from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .battery_core import PERIOD_MINUTES, Period, solve
from .battery_coordinator import BatteryCoordinator
from .coordinator import PricingCoordinator
from .pricing import build_price_horizon

DEFAULT_DAYS = 30
MAX_DAYS = 60
DEFAULT_SOLAR_ENTITY = "sensor.solis_ac_output_total_power"
DEFAULT_GRID_IMPORT_ENTITY = "sensor.solis_daily_grid_energy_purchased"
MAX_MISSING_LOAD_FRACTION = 0.25


def _hourly_kwh(rows: list[dict[str, Any]], unit: str | None, tz) -> dict[datetime, float]:
    """Map hour start -> kWh from recorder statistics rows.

    Energy sensors (total/total_increasing) provide ``change`` in their own
    unit; power sensors provide ``mean`` which is integrated over the hour.
    """
    factor = 1.0 if unit in ("kW", "kWh") else 1.0 / 1000.0
    result: dict[datetime, float] = {}
    for row in rows:
        start = row.get("start")
        if isinstance(start, (int, float)):
            start = dt_util.utc_from_timestamp(start)
        if start is None:
            continue
        change = row.get("change")
        mean = row.get("mean")
        if change is not None:
            result[start.astimezone(tz)] = float(change) * (
                1.0 if unit == "kWh" else factor
            )
        elif mean is not None:
            result[start.astimezone(tz)] = float(mean) * factor
    return result


async def async_backtest(
    pricing: PricingCoordinator,
    battery: BatteryCoordinator,
    data: dict[str, Any],
) -> dict[str, Any]:
    from homeassistant.components.recorder.statistics import (
        statistics_during_period,
    )

    hass = battery.hass
    tz = dt_util.get_default_time_zone()
    days = max(1, min(MAX_DAYS, int(data.get("days", DEFAULT_DAYS))))

    load_entity = str(data.get("load_entity") or battery._option("load_power_entity"))
    solar_entity = str(data.get("solar_entity") or DEFAULT_SOLAR_ENTITY)
    grid_entity = str(data.get("grid_import_entity") or DEFAULT_GRID_IMPORT_ENTITY)

    today_start = dt_util.start_of_local_day()
    range_start = today_start - timedelta(days=days)

    entities = {load_entity, solar_entity} | ({grid_entity} if grid_entity else set())
    stats = await hass.async_add_executor_job(
        statistics_during_period,
        hass,
        range_start,
        today_start,
        entities,
        "hour",
        None,
        {"mean", "change"},
    )

    def unit_of(entity_id: str) -> str | None:
        state = hass.states.get(entity_id)
        return state.attributes.get("unit_of_measurement") if state else None

    load_kwh = _hourly_kwh(stats.get(load_entity, []), unit_of(load_entity), tz)
    solar_kwh = _hourly_kwh(stats.get(solar_entity, []), unit_of(solar_entity), tz)
    grid_kwh = (
        _hourly_kwh(stats.get(grid_entity, []), unit_of(grid_entity), tz)
        if grid_entity
        else {}
    )
    if not load_kwh:
        raise HomeAssistantError(
            f"No hourly statistics found for load entity {load_entity}"
        )

    config = pricing.pricing_config()
    base_params = battery.battery_params()
    soc = float(data.get("initial_soc_pct", base_params.reserve_soc_pct))

    # Nord Pool serves dates in the CET market timezone, so one fetched
    # "day" does not cover the local (EET) day's first hour; merge each
    # day with the previous day's fetch.
    price_cache: dict[date_type, list] = {}

    async def fetch_day_cached(target: date_type) -> list:
        if target not in price_cache:
            price_cache[target] = await pricing.async_fetch_day(target)
        return price_cache[target]

    day_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    totals = {"baseline_cents": 0.0, "planned_cents": 0.0, "actual_cents": 0.0}
    actual_days = 0

    for offset in range(days, 0, -1):
        day_start = today_start - timedelta(days=offset)
        day_end = today_start - timedelta(days=offset - 1)
        day: date_type = day_start.astimezone(tz).date()

        try:
            raw_slots = await fetch_day_cached(
                day - timedelta(days=1)
            ) + await fetch_day_cached(day)
        except Exception as err:  # noqa: BLE001 - report and continue
            skipped.append({"date": day.isoformat(), "reason": f"price fetch: {err}"})
            continue
        price_periods = build_price_horizon(
            [slot for slot in raw_slots if slot.start is not None],
            now=day_start,
            config=config,
            local_tz=tz,
        )
        price_by_start = {p.start: p.all_in_cents_per_kwh for p in price_periods}

        quarters: list[datetime] = []
        cursor = day_start
        while cursor < day_end:
            quarters.append(cursor)
            cursor += timedelta(minutes=PERIOD_MINUTES)

        if any(q not in price_by_start for q in quarters):
            skipped.append({"date": day.isoformat(), "reason": "incomplete prices"})
            continue

        hours = [q for q in quarters if q.minute == 0]
        missing_load = [h for h in hours if h not in load_kwh]
        if len(missing_load) > MAX_MISSING_LOAD_FRACTION * len(hours):
            skipped.append(
                {
                    "date": day.isoformat(),
                    "reason": f"load statistics missing for {len(missing_load)}/{len(hours)} hours",
                }
            )
            continue
        present = [load_kwh[h] for h in hours if h in load_kwh]
        load_fill = sum(present) / len(present)

        def hour_of(q: datetime) -> datetime:
            return q.replace(minute=0)

        periods = [
            Period(
                start=q,
                price_cents_per_kwh=price_by_start[q],
                load_kwh=round(load_kwh.get(hour_of(q), load_fill) / 4.0, 4),
                solar_kwh=round(solar_kwh.get(hour_of(q), 0.0) / 4.0, 4),
            )
            for q in quarters
        ]

        params = battery.battery_params({"soc_pct": soc})
        plan = await hass.async_add_executor_job(solve, periods, params)

        actual_cents: float | None = None
        if grid_entity:
            covered = [h for h in hours if h in grid_kwh]
            if len(covered) >= (1 - MAX_MISSING_LOAD_FRACTION) * len(hours):
                actual_cents = round(
                    sum(
                        grid_kwh.get(hour_of(q), 0.0) / 4.0 * price_by_start[q]
                        for q in quarters
                    ),
                    1,
                )

        row = {
            "date": day.isoformat(),
            "baseline_cents": round(plan.baseline_cost_cents, 1),
            "planned_cents": round(plan.total_cost_cents, 1),
            "planned_savings_cents": round(
                plan.baseline_cost_cents - plan.total_cost_cents, 1
            ),
            "actual_cents": actual_cents,
            "load_kwh": round(sum(p.load_kwh for p in periods), 2),
            "solar_kwh": round(sum(p.solar_kwh for p in periods), 2),
            "charged_kwh": round(
                sum(p.grid_charge_kwh for p in plan.periods), 2
            ),
            "discharged_kwh": round(
                sum(p.discharge_to_load_kwh for p in plan.periods), 2
            ),
            "start_soc_pct": round(soc, 1),
            "end_soc_pct": plan.end_soc_pct,
        }
        day_rows.append(row)
        totals["baseline_cents"] += plan.baseline_cost_cents
        totals["planned_cents"] += plan.total_cost_cents
        if actual_cents is not None:
            totals["actual_cents"] += actual_cents
            actual_days += 1
        soc = plan.end_soc_pct

    savings = totals["baseline_cents"] - totals["planned_cents"]
    return {
        "config": {
            "days_requested": days,
            "load_entity": load_entity,
            "solar_entity": solar_entity,
            "grid_import_entity": grid_entity or None,
            "initial_soc_pct": float(
                data.get("initial_soc_pct", base_params.reserve_soc_pct)
            ),
        },
        "totals": {
            "days_evaluated": len(day_rows),
            "days_skipped": len(skipped),
            "baseline_cents": round(totals["baseline_cents"], 1),
            "planned_cents": round(totals["planned_cents"], 1),
            "planned_savings_cents": round(savings, 1),
            "planned_savings_pct": round(
                savings / totals["baseline_cents"] * 100.0, 1
            )
            if totals["baseline_cents"]
            else 0.0,
            "actual_cents": round(totals["actual_cents"], 1) if actual_days else None,
            "actual_days": actual_days,
        },
        "days": day_rows,
        "skipped": skipped,
        "notes": [
            "planned assumes perfect foresight of that day's load/solar: an upper bound on achievable savings",
            "hourly statistics are integrated flat across each hour's quarters",
        ],
    }
