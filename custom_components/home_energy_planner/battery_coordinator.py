"""Battery planner coordinator: gathers inputs, solves, observes or controls.

Modes (config option `battery_mode`):
- off: no planning at all
- observe: plan every quarter and publish the result, write nothing
- control: additionally apply the compiled slot tables via the writer
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .battery_core import (
    PERIOD_MINUTES,
    BatteryParams,
    DispatchPlan,
    Period,
    compile_slots,
)
from .const import DOMAIN
from .coordinator import PricingCoordinator
from .solis_slots import SlotSpec
from .solis_writer import apply_slots

_LOGGER = logging.getLogger(__name__)

CONF_BATTERY_MODE = "battery_mode"
MODE_OFF = "off"
MODE_OBSERVE = "observe"
MODE_CONTROL = "control"

DEFAULTS = {
    "battery_soc_entity": "sensor.solis_remaining_battery_capacity",
    "battery_reserve_entity": "number.inverter_control_110ca2228060121_battery_reserve_soc",
    "battery_max_charge_entity": "number.inverter_control_110ca2228060121_battery_max_charge_current",
    "battery_max_discharge_entity": "number.inverter_control_110ca2228060121_battery_max_discharge_current",
    "load_power_entity": "sensor.solis_total_consumption_power",
    "solar_today_entity": "sensor.energy_production_today_remaining",
    "solar_tomorrow_entity": "sensor.energy_production_tomorrow",
    "battery_capacity_kwh": 5.12,
    "battery_soh_pct": 97.0,
    "battery_engine": "lp",
    # daily-energy temperature model fitted 2026-07-06 on 182 days of
    # statistics vs Open-Meteo (R2 0.62 on heating days): expected
    # kWh/day = max(warm_floor, base + slope * T_out)
    "outdoor_temp_entity": "sensor.ilp_ulkolampotila",
    "weather_entity": "weather.forecast_koti",
    "load_temp_base_kwh": 81.9,
    "load_temp_slope_kwh_per_c": -2.80,
    "load_temp_warm_floor_kwh": 58.5,
}
LOAD_SCALE_MIN = 0.7
LOAD_SCALE_MAX = 1.4


class BatteryPlanData:
    def __init__(
        self,
        plan: DispatchPlan,
        charge_slots: list[SlotSpec],
        discharge_slots: list[SlotSpec],
        mode: str,
        engine: str,
        load_info: dict[str, Any],
        tank_windows: list[dict[str, Any]] | None,
        input_periods: list[Period],
        applied: dict[str, Any] | None,
    ) -> None:
        self.plan = plan
        self.charge_slots = charge_slots
        self.discharge_slots = discharge_slots
        self.mode = mode
        self.engine = engine
        self.load_info = load_info
        self.tank_windows = tank_windows
        self.input_periods = input_periods
        self.applied = applied


class BatteryCoordinator(DataUpdateCoordinator[BatteryPlanData]):
    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        pricing: PricingCoordinator,
    ) -> None:
        super().__init__(hass, _LOGGER, name=f"{DOMAIN} battery", update_interval=None)
        self._entry = entry
        self._pricing = pricing

    def async_schedule_ticks(self) -> None:
        @callback
        def _tick(_now: datetime) -> None:
            self.hass.async_create_task(self.async_request_refresh())

        self._entry.async_on_unload(
            async_track_time_change(
                self.hass, _tick, minute=[0, 15, 30, 45], second=20
            )
        )

    def _option(self, key: str) -> Any:
        return self._entry.options.get(
            key, self._entry.data.get(key, DEFAULTS.get(key))
        )

    @property
    def mode(self) -> str:
        return str(
            self._entry.options.get(
                CONF_BATTERY_MODE, self._entry.data.get(CONF_BATTERY_MODE, MODE_OBSERVE)
            )
        )

    def _float_state(self, key: str, fallback: float = 0.0) -> float:
        entity_id = str(self._option(key))
        state = self.hass.states.get(entity_id)
        try:
            return float(state.state)  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            return fallback

    async def load_baseline_kwh_by_quarter(self, now: datetime) -> dict[int, float]:
        """Mean load kWh per quarter-hour-of-day from 7 days of recorder stats."""
        from homeassistant.components.recorder.statistics import (
            statistics_during_period,
        )

        entity_id = str(self._option("load_power_entity"))
        rows = (
            await self.hass.async_add_executor_job(
                statistics_during_period,
                self.hass,
                now - timedelta(days=7),
                now,
                {entity_id},
                "5minute",
                None,
                {"mean"},
            )
        ).get(entity_id, [])
        sums: dict[int, float] = {}
        counts: dict[int, int] = {}
        for row in rows:
            mean = row.get("mean")
            if mean is None:
                continue
            start = row["start"]
            if isinstance(start, (int, float)):
                start = datetime.fromtimestamp(start, tz=now.tzinfo)
            local = start.astimezone(now.tzinfo)
            bucket = (local.hour * 60 + (local.minute // PERIOD_MINUTES) * PERIOD_MINUTES)
            sums[bucket] = sums.get(bucket, 0.0) + float(mean)
            counts[bucket] = counts.get(bucket, 0) + 1
        return {
            bucket: (sums[bucket] / counts[bucket]) / 1000.0 * (PERIOD_MINUTES / 60.0)
            for bucket in sums
        }

    def _expected_daily_kwh(self, temp_c: float) -> float:
        return max(
            float(self._option("load_temp_warm_floor_kwh")),
            float(self._option("load_temp_base_kwh"))
            + float(self._option("load_temp_slope_kwh_per_c")) * temp_c,
        )

    async def _mean_outdoor_temp_7d(self, now: datetime) -> float | None:
        from homeassistant.components.recorder.statistics import (
            statistics_during_period,
        )

        entity_id = str(self._option("outdoor_temp_entity"))
        rows = (
            await self.hass.async_add_executor_job(
                statistics_during_period,
                self.hass,
                now - timedelta(days=7),
                now,
                {entity_id},
                "day",
                None,
                {"mean"},
            )
        ).get(entity_id, [])
        means = [row["mean"] for row in rows if row.get("mean") is not None]
        return sum(means) / len(means) if means else None

    async def _forecast_mean_temp_24h(self) -> float | None:
        weather_entity = str(self._option("weather_entity"))
        try:
            response = await self.hass.services.async_call(
                "weather",
                "get_forecasts",
                {"entity_id": weather_entity, "type": "hourly"},
                blocking=True,
                return_response=True,
            )
        except Exception as err:  # noqa: BLE001 - scaling degrades to 1.0
            _LOGGER.debug("Weather forecast unavailable for load scaling: %s", err)
            return None
        rows = []
        if isinstance(response, dict):
            rows = (response.get(weather_entity) or {}).get("forecast") or []
        temps = []
        for row in rows[:24]:
            try:
                temps.append(float(row["temperature"]))
            except (KeyError, TypeError, ValueError):
                continue
        return sum(temps) / len(temps) if temps else None

    async def async_load_forecast_kwh_by_quarter(
        self, now: datetime
    ) -> tuple[dict[int, float], dict[str, Any]]:
        """7-day bucket baseline scaled by the temperature model."""
        buckets = await self.load_baseline_kwh_by_quarter(now)
        t_recent = await self._mean_outdoor_temp_7d(now)
        t_forecast = await self._forecast_mean_temp_24h()
        scale = 1.0
        if t_recent is not None and t_forecast is not None:
            reference = self._expected_daily_kwh(t_recent)
            if reference > 0:
                scale = max(
                    LOAD_SCALE_MIN,
                    min(LOAD_SCALE_MAX, self._expected_daily_kwh(t_forecast) / reference),
                )
        info = {
            "scale": round(scale, 3),
            "outdoor_mean_7d": round(t_recent, 1) if t_recent is not None else None,
            "forecast_mean_24h": round(t_forecast, 1) if t_forecast is not None else None,
        }
        return {bucket: kwh * scale for bucket, kwh in buckets.items()}, info

    async def async_solar_series_kwh(
        self, starts: list[datetime], now: datetime
    ) -> list[float]:
        """Per-period solar kWh from Forecast.Solar's hourly Wh series.

        Falls back to the daylight bell curve when the forecast_solar
        integration or its energy-platform data is unavailable.
        """
        wh_by_hour = await self._forecast_solar_wh_by_hour(now)
        if not wh_by_hour:
            return self.solar_series_kwh(starts, now)
        series = []
        for ts in starts:
            hour = ts.astimezone(now.tzinfo).replace(
                minute=0, second=0, microsecond=0
            )
            series.append(round(wh_by_hour.get(hour, 0.0) / 4000.0, 4))
        return series

    async def _forecast_solar_wh_by_hour(
        self, now: datetime
    ) -> dict[datetime, float]:
        from .solar_forecast import async_wh_by_hour

        return await async_wh_by_hour(self.hass, now.tzinfo)

    def solar_series_kwh(self, starts: list[datetime], now: datetime) -> list[float]:
        """Distribute daily-total solar forecasts over a daylight bell curve.

        Fallback path behind ``async_solar_series_kwh``.
        """
        today_remaining = self._float_state("solar_today_entity")
        tomorrow_total = self._float_state("solar_tomorrow_entity")

        def weight(ts: datetime) -> float:
            hour = ts.astimezone(now.tzinfo).hour
            if not 6 <= hour < 20:
                return 0.0
            return max(0.2, 1.0 - abs(13 - hour) / 7.0)

        weights = [weight(ts) for ts in starts]
        day_of = [ts.astimezone(now.tzinfo).date() for ts in starts]
        totals = {now.date(): today_remaining, now.date() + timedelta(days=1): tomorrow_total}
        result = []
        for date in set(day_of):
            day_weight = sum(w for w, d in zip(weights, day_of) if d == date)
            budget = totals.get(date, 0.0)
            for index, (w, d) in enumerate(zip(weights, day_of)):
                if d == date:
                    while len(result) <= index:
                        result.append(0.0)
                    result[index] = budget * w / day_weight if day_weight > 0 else 0.0
        while len(result) < len(starts):
            result.append(0.0)
        return result

    def battery_params(self, overrides: dict[str, Any] | None = None) -> BatteryParams:
        """Battery parameters from live state/options, with explicit overrides."""
        values = {
            "capacity_kwh": float(self._option("battery_capacity_kwh")),
            "state_of_health_pct": float(self._option("battery_soh_pct")),
            "soc_pct": self._float_state("battery_soc_entity", 0.0),
            "reserve_soc_pct": self._float_state("battery_reserve_entity", 18.0),
            "max_charge_current": int(self._float_state("battery_max_charge_entity", 25.0)),
            "max_discharge_current": int(
                self._float_state("battery_max_discharge_entity", 25.0)
            ),
        }
        for key, value in (overrides or {}).items():
            if value is None or key not in values:
                continue
            values[key] = int(value) if key.startswith("max_") else float(value)
        return BatteryParams(**values)

    async def _async_update_data(self) -> BatteryPlanData:
        mode = self.mode
        pricing = self._pricing.data
        if mode == MODE_OFF:
            raise UpdateFailed("battery module is off")
        if pricing is None or not pricing.periods:
            raise UpdateFailed("no price horizon available")

        now = dt_util.now()
        starts = [p.start for p in pricing.periods]
        load_by_quarter, load_info = await self.async_load_forecast_kwh_by_quarter(now)
        solar = await self.async_solar_series_kwh(starts, now)

        periods = []
        for index, price_period in enumerate(pricing.periods):
            local = price_period.start.astimezone(now.tzinfo)
            bucket = local.hour * 60 + (local.minute // PERIOD_MINUTES) * PERIOD_MINUTES
            periods.append(
                Period(
                    start=price_period.start,
                    price_cents_per_kwh=price_period.all_in_cents_per_kwh,
                    load_kwh=load_by_quarter.get(bucket, 0.15),
                    solar_kwh=solar[index],
                    export_cents_per_kwh=price_period.export_cents_per_kwh,
                )
            )

        battery = self.battery_params()

        from .milp_core import solve_best

        plan, engine = await self.hass.async_add_executor_job(
            solve_best, periods, battery, str(self._option("battery_engine"))
        )
        charge_slots, discharge_slots = compile_slots(
            plan.periods, battery, now=dt_util.now()
        )

        # joint battery+tank MILP runs as an observe artifact alongside the
        # rule-based water heater module; its windows publish for the gate
        tank_windows: list[dict[str, Any]] | None = None
        if engine == "lp":
            try:
                from .milp_core import TankParams, solve_joint

                _joint_plan, windows = await self.hass.async_add_executor_job(
                    solve_joint, periods, battery, TankParams()
                )
                tank_windows = [
                    {
                        "start": periods[start].start.isoformat(),
                        "end": (
                            periods[end - 1].start + timedelta(minutes=PERIOD_MINUTES)
                        ).isoformat(),
                    }
                    for start, end in windows
                ]
            except Exception as err:  # noqa: BLE001 - observe artifact only
                _LOGGER.debug("Joint tank solve unavailable: %s", err)

        applied: dict[str, Any] | None = None
        if mode == MODE_CONTROL:
            try:
                applied = await apply_slots(
                    self.hass,
                    charge_slots=[s.as_dict() for s in charge_slots],
                    discharge_slots=[s.as_dict() for s in discharge_slots],
                )
            except Exception as err:  # noqa: BLE001 - plan survives apply failure
                applied = {"success": False, "error": str(err)}
            if not applied.get("success"):
                _LOGGER.warning("Battery plan apply failed: %s", applied)

        return BatteryPlanData(
            plan,
            charge_slots,
            discharge_slots,
            mode,
            engine,
            load_info,
            tank_windows,
            periods,
            applied,
        )
