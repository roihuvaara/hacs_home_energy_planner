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
    "climate_entity": "climate.gree_versati_space_heating_2",
    "solar_energy_current_hour_entity": "sensor.energy_current_hour",
    "solar_energy_next_hour_entity": "sensor.energy_next_hour",
    "legacy_climate_target_entity": "input_number.climate_target_temp",
}


class ClimateData:
    def __init__(
        self,
        result: ClimateResult,
        mode: str,
        room_temp: float | None,
        lead_hold_until: datetime | None,
        legacy_target: float | None,
        applied: dict[str, Any] | None,
    ) -> None:
        self.result = result
        self.mode = mode
        self.room_temp = room_temp
        self.lead_hold_until = lead_hold_until
        self.legacy_target = legacy_target
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
        current_vat = (
            pricing.periods[0].vat_cents_per_kwh if pricing and pricing.periods else 0.0
        )
        current_all_in = future_all_in[0] if future_all_in else 0.0

        inputs = ClimateInputs(
            hourly_forecast=hours,
            fallback_temp=fallback,
            room_temp=self._float_state("room_temp_entity"),
            current_vat=current_vat,
            current_all_in=current_all_in,
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

        applied: dict[str, Any] | None = None
        if mode == MODE_CONTROL:
            applied = await self._async_apply(result.target)

        return ClimateData(
            result=result,
            mode=mode,
            room_temp=inputs.room_temp,
            lead_hold_until=self._lead_hold_until,
            legacy_target=self._float_state("legacy_climate_target_entity"),
            applied=applied,
        )

    async def _async_apply(self, target: float) -> dict[str, Any]:
        climate_entity = str(self._option("climate_entity"))
        state = self.hass.states.get(climate_entity)
        current = state.attributes.get("temperature") if state else None
        try:
            unchanged = current is not None and float(current) == target
        except (TypeError, ValueError):
            unchanged = False
        if unchanged:
            return {"success": True, "written": False, "target": target}
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
        return {"success": True, "written": True, "target": target}
