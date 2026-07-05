"""Climate coordinator: gathers inputs, computes the target, observes or controls.

Modes (config option `climate_mode`):
- off: no computation
- observe: compute and publish the target every quarter, write nothing
- control: additionally write the target to the Gree Versati climate entity

The lead-boost hold (legacy timer.climate_lead_boost_hold) is managed
internally: a falling room-temperature update below the comfort floor
starts/refreshes a 90-minute hold.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_change,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .climate_core import (
    ClimateConfig,
    ClimateInputs,
    ClimateResult,
    ForecastHour,
    compute_climate_target,
)
from .const import DOMAIN
from .coordinator import PricingCoordinator

_LOGGER = logging.getLogger(__name__)

CONF_CLIMATE_MODE = "climate_mode"
MODE_OFF = "off"
MODE_OBSERVE = "observe"
MODE_CONTROL = "control"

DEFAULTS = {
    "weather_entity": "weather.forecast_koti",
    "room_temp_entity": "sensor.olohuone_climate_lampotila",
    "room_humidity_entity": "sensor.olohuone_climate_kosteus",
    "climate_entity": "climate.gree_versati_space_heating_2",
    "solar_energy_current_hour_entity": "sensor.energy_current_hour",
    "solar_energy_next_hour_entity": "sensor.energy_next_hour",
    "legacy_climate_target_entity": "input_number.climate_target_temp",
    # hydronic cooling rollout: off | notify (announce good trial days) |
    # control (planner drives cool mode with the dew-point guard)
    "versati_cooling": "notify",
}


class ClimateData:
    def __init__(
        self,
        result: ClimateResult,
        mode: str,
        room_temp: float | None,
        lead_hold_until: datetime | None,
        legacy_target: float | None,
        cooling: dict[str, Any],
        applied: dict[str, Any] | None,
    ) -> None:
        self.result = result
        self.mode = mode
        self.room_temp = room_temp
        self.lead_hold_until = lead_hold_until
        self.legacy_target = legacy_target
        self.cooling = cooling
        self.applied = applied


class ClimateCoordinator(DataUpdateCoordinator[ClimateData]):
    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        pricing: PricingCoordinator,
    ) -> None:
        super().__init__(hass, _LOGGER, name=f"{DOMAIN} climate", update_interval=None)
        self._entry = entry
        self._pricing = pricing
        self._config = ClimateConfig()
        self._lead_hold_until: datetime | None = None
        self._last_write: tuple[float, datetime] | None = None
        self._trial_until: datetime | None = None
        self._last_test_notify = None
        from .manual_override import ManualOverrideTracker

        self._override = ManualOverrideTracker()

    def _override_hold(self) -> timedelta:
        return timedelta(
            hours=float(
                self._entry.options.get(
                    "manual_override_hours",
                    self._entry.data.get("manual_override_hours", 4.0),
                )
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
                CONF_CLIMATE_MODE, self._entry.data.get(CONF_CLIMATE_MODE, MODE_OBSERVE)
            )
        )

    def async_schedule_ticks(self) -> None:
        @callback
        def _tick(_now: datetime) -> None:
            self.hass.async_create_task(self.async_request_refresh())

        self._entry.async_on_unload(
            async_track_time_change(
                self.hass, _tick, minute=[0, 15, 30, 45], second=40
            )
        )

        @callback
        def _room_changed(event: Event) -> None:
            new_state = event.data.get("new_state")
            old_state = event.data.get("old_state")
            try:
                new_temp = float(new_state.state)  # type: ignore[union-attr]
                old_temp = float(old_state.state)  # type: ignore[union-attr]
            except (AttributeError, TypeError, ValueError):
                return
            if new_temp < old_temp and new_temp < self._config.lead_room_below:
                self._lead_hold_until = dt_util.now() + timedelta(
                    minutes=self._config.lead_hold_minutes
                )
            self.hass.async_create_task(self.async_request_refresh())

        self._entry.async_on_unload(
            async_track_state_change_event(
                self.hass, [str(self._option("room_temp_entity"))], _room_changed
            )
        )

    def _float_state(self, key: str, fallback: float | None = None) -> float | None:
        state = self.hass.states.get(str(self._option(key)))
        try:
            return float(state.state)  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            return fallback

    async def _hourly_forecast(self) -> tuple[list[ForecastHour], float]:
        weather_entity = str(self._option("weather_entity"))
        fallback = 0.0
        state = self.hass.states.get(weather_entity)
        if state is not None:
            try:
                fallback = float(state.attributes.get("temperature"))
            except (TypeError, ValueError):
                fallback = 0.0
        try:
            response = await self.hass.services.async_call(
                "weather",
                "get_forecasts",
                {"entity_id": weather_entity, "type": "hourly"},
                blocking=True,
                return_response=True,
            )
        except Exception as err:  # noqa: BLE001 - degrade to fallback temp
            _LOGGER.warning("Weather forecast unavailable: %s", err)
            return [], fallback
        rows = []
        if isinstance(response, dict):
            rows = (response.get(weather_entity) or {}).get("forecast") or []
        hours = []
        for row in rows[:24]:
            try:
                temperature = float(row["temperature"])
            except (KeyError, TypeError, ValueError):
                continue
            cloud = row.get("cloud_coverage")
            hours.append(
                ForecastHour(
                    temperature=temperature,
                    wind_gust_speed=float(row.get("wind_gust_speed") or 0.0),
                    condition=str(row.get("condition") or ""),
                    cloud_coverage=float(cloud) if cloud is not None else None,
                )
            )
        return hours, fallback

    async def _async_notify(self, title: str, message: str) -> None:
        await self.hass.services.async_call(
            "persistent_notification",
            "create",
            {"title": title, "message": message, "notification_id": f"{DOMAIN}_cooling"},
            blocking=False,
        )
        try:
            await self.hass.services.async_call(
                "notify", "notify", {"title": title, "message": message}, blocking=False
            )
        except Exception as err:  # noqa: BLE001 - persistent notification stands
            _LOGGER.debug("Mobile notify unavailable: %s", err)

    async def async_start_cool_trial(
        self, minutes: int, water_temp: float | None
    ) -> dict[str, Any]:
        """Bounded hydronic cooling trial: cool mode now, auto-revert later."""
        from homeassistant.helpers.event import async_call_later

        from .climate_core import cool_water_target

        minutes = max(10, min(240, minutes))
        room = self._float_state("room_temp_entity")
        humidity = self._float_state("room_humidity_entity")
        target, dew = cool_water_target(room, humidity, None, self._config)
        if water_temp is not None:
            target = max(float(water_temp), (dew + self._config.dew_point_margin) if dew is not None else float(water_temp))
        climate_entity = str(self._option("climate_entity"))
        await self.hass.services.async_call(
            "climate",
            "set_hvac_mode",
            {"entity_id": climate_entity, "hvac_mode": "cool"},
            blocking=True,
        )
        await self.hass.services.async_call(
            "climate",
            "set_temperature",
            {"entity_id": climate_entity, "temperature": target},
            blocking=True,
        )
        self._trial_until = dt_util.now() + timedelta(minutes=minutes)

        async def _revert(_now: Any) -> None:
            self._trial_until = None
            await self.hass.services.async_call(
                "climate",
                "set_hvac_mode",
                {"entity_id": climate_entity, "hvac_mode": "heat"},
                blocking=True,
            )
            await self._async_notify(
                "Hydronic cooling trial finished",
                "Reverted to heat mode. Check the pannuhuone manifold and exposed pipes for condensation before trusting cool mode.",
            )
            await self.async_request_refresh()

        async_call_later(self.hass, timedelta(minutes=minutes), _revert)
        await self._async_notify(
            "Hydronic cooling trial started",
            f"Cool mode at {target} C water for {minutes} min (room dew point {dew} C). Watch the manifold for sweating.",
        )
        return {
            "success": True,
            "water_target": target,
            "dew_point": dew,
            "revert_at": self._trial_until.isoformat(),
        }

    async def _async_update_data(self) -> ClimateData:
        mode = self.mode
        if mode == MODE_OFF:
            raise UpdateFailed("climate module is off")

        now = dt_util.now()
        if self._lead_hold_until is not None and self._lead_hold_until <= now:
            self._lead_hold_until = None

        hours, fallback = await self._hourly_forecast()
        pricing = self._pricing.data
        future_all_in = (
            [p.all_in_cents_per_kwh for p in pricing.periods] if pricing else []
        )

        inputs = ClimateInputs(
            hourly_forecast=hours,
            fallback_temp=fallback,
            room_temp=self._float_state("room_temp_entity"),
            future_all_in=future_all_in,
            solar_current_hour_kwh=self._float_state(
                "solar_energy_current_hour_entity", 0.0
            )
            or 0.0,
            solar_next_hour_kwh=self._float_state(
                "solar_energy_next_hour_entity", 0.0
            )
            or 0.0,
            lead_hold_active=self._lead_hold_until is not None,
        )
        result = compute_climate_target(inputs, self._config)

        # hydronic cooling: dew-point-guarded chilled-water target and
        # test-day detection (notify mode announces good trial days)
        from .climate_core import cool_water_target

        humidity = self._float_state("room_humidity_entity")
        forecast_max = max((h.temperature for h in hours), default=None)
        water_target, dew = cool_water_target(
            inputs.room_temp, humidity, forecast_max, self._config
        )
        suitable_test_day = (
            forecast_max is not None
            and forecast_max >= self._config.test_day_outdoor_max
        )
        cooling_mode = str(self._option("versati_cooling"))
        cooling = {
            "rollout": cooling_mode,
            "forecast_max_24h": forecast_max,
            "water_target": water_target,
            "dew_point": dew,
            "suitable_test_day": suitable_test_day,
            "trial_until": self._trial_until.isoformat() if self._trial_until else None,
        }
        if (
            cooling_mode == "notify"
            and suitable_test_day
            and 7 <= now.hour < 12
            and self._last_test_notify != now.date()
        ):
            self._last_test_notify = now.date()
            await self._async_notify(
                "Good day to trial hydronic cooling",
                f"Forecast max {forecast_max:.0f} C. Run home_energy_planner.versati_cool_trial "
                f"(water target {water_target} C, room dew point {dew} C) and check the "
                "pannuhuone manifold for condensation during the run.",
            )

        applied: dict[str, Any] | None = None
        if mode == MODE_CONTROL and self._trial_until is None:
            if cooling_mode == "control":
                applied = await self._async_apply_with_cooling(
                    result.target, water_target, inputs.room_temp
                )
            else:
                applied = await self._async_apply(result.target)

        return ClimateData(
            result=result,
            mode=mode,
            room_temp=inputs.room_temp,
            lead_hold_until=self._lead_hold_until,
            legacy_target=self._float_state("legacy_climate_target_entity"),
            cooling=cooling,
            applied=applied,
        )

    async def _async_apply_with_cooling(
        self, heat_target: float, water_target: float, room: float | None
    ) -> dict[str, Any]:
        """Drive heat/cool mode by the room band, dew-point guard on water."""
        climate_entity = str(self._option("climate_entity"))
        state = self.hass.states.get(climate_entity)
        hvac = state.state if state else None
        if room is not None and room >= self._config.cool_room_above and hvac != "cool":
            await self.hass.services.async_call(
                "climate",
                "set_hvac_mode",
                {"entity_id": climate_entity, "hvac_mode": "cool"},
                blocking=True,
            )
            hvac = "cool"
        elif hvac == "cool" and (room is None or room <= self._config.cool_room_stop):
            await self.hass.services.async_call(
                "climate",
                "set_hvac_mode",
                {"entity_id": climate_entity, "hvac_mode": "heat"},
                blocking=True,
            )
            hvac = "heat"
        if hvac == "cool":
            await self.hass.services.async_call(
                "climate",
                "set_temperature",
                {"entity_id": climate_entity, "temperature": water_target},
                blocking=True,
            )
            return {"success": True, "hvac": "cool", "water_target": water_target}
        return await self._async_apply(heat_target)

    async def _async_apply(self, target: float) -> dict[str, Any]:
        climate_entity = str(self._option("climate_entity"))
        state = self.hass.states.get(climate_entity)
        current = state.attributes.get("temperature") if state else None
        try:
            current_value = float(current) if current is not None else None
        except (TypeError, ValueError):
            current_value = None
        unchanged = current_value is not None and current_value == target
        if unchanged:
            return {"success": True, "written": False, "target": target}
        now_check = dt_util.now()
        if self._override.suppressed(
            current_value, target, now_check, self._override_hold()
        ):
            return {
                "success": True,
                "written": False,
                "target": target,
                "manual_override_until": self._override.until.isoformat()
                if self._override.until
                else None,
            }
        # setpoint dwell: a 1-degree wiggle within half an hour of the last
        # write is deferred so price-quartile flapping cannot short-cycle
        # the heat pump; bigger moves (comfort, cold dip, spikes) go through
        now = dt_util.now()
        if self._last_write is not None:
            last_target, last_time = self._last_write
            if (
                abs(target - last_target) < 2
                and now - last_time < timedelta(minutes=30)
            ):
                return {
                    "success": True,
                    "written": False,
                    "target": target,
                    "deferred": True,
                }
        try:
            await self.hass.services.async_call(
                "climate",
                "set_temperature",
                {"entity_id": climate_entity, "temperature": target},
                blocking=True,
            )
        except Exception as err:  # noqa: BLE001 - report, retry next tick
            _LOGGER.warning("Climate target write failed: %s", err)
            return {"success": False, "written": False, "error": str(err)}
        self._last_write = (target, now)
        self._override.record_write(target)
        return {"success": True, "written": True, "target": target}
