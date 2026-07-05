"""ILP (air-to-air) coordinator: cooling recommendations, observe or control.

Modes (config option `ilp_mode`):
- off: no computation
- observe: publish the recommended action every quarter, write nothing
- control: additionally drive climate.living_room (hvac_mode + target)

A 30-minute dwell protects the compressor from flapping between cool
and off; the pure core adds run-completion hysteresis on top.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_change,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .coordinator import PricingCoordinator
from .ilp_core import ACTION_COOL, IlpConfig, IlpInputs, IlpResult, compute_ilp_action

_LOGGER = logging.getLogger(__name__)

CONF_ILP_MODE = "ilp_mode"
MODE_OFF = "off"
MODE_OBSERVE = "observe"
MODE_CONTROL = "control"

DEFAULTS = {
    "ilp_climate_entity": "climate.living_room",
    "room_temp_entity": "sensor.olohuone_climate_lampotila",
    "grid_power_entity": "sensor.solis_power_grid_total_power",
    "weather_entity": "weather.forecast_koti",
}
MIN_ACTION_DWELL = timedelta(minutes=30)


class IlpData:
    def __init__(
        self,
        result: IlpResult,
        effective_action: str,
        mode: str,
        room_temp: float | None,
        applied: dict[str, Any] | None,
    ) -> None:
        self.result = result
        self.effective_action = effective_action
        self.mode = mode
        self.room_temp = room_temp
        self.applied = applied


class IlpCoordinator(DataUpdateCoordinator[IlpData]):
    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        pricing: PricingCoordinator,
    ) -> None:
        super().__init__(hass, _LOGGER, name=f"{DOMAIN} ilp", update_interval=None)
        self._entry = entry
        self._pricing = pricing
        self._config = IlpConfig()
        self._effective_action: str | None = None
        self._action_since: datetime | None = None
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
                CONF_ILP_MODE, self._entry.data.get(CONF_ILP_MODE, MODE_OBSERVE)
            )
        )

    def async_schedule_ticks(self) -> None:
        @callback
        def _tick(_now: datetime) -> None:
            self.hass.async_create_task(self.async_request_refresh())

        self._entry.async_on_unload(
            async_track_time_change(
                self.hass, _tick, minute=[0, 15, 30, 45], second=55
            )
        )

        @callback
        def _room_changed(_event) -> None:
            self.hass.async_create_task(self.async_request_refresh())

        self._entry.async_on_unload(
            async_track_state_change_event(
                self.hass, [str(self._option("room_temp_entity"))], _room_changed
            )
        )

    def _float_state(self, key: str) -> float | None:
        state = self.hass.states.get(str(self._option(key)))
        try:
            return float(state.state)  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            return None

    async def _forecast_max_temp_24h(self) -> float | None:
        weather_entity = str(self._option("weather_entity"))
        try:
            response = await self.hass.services.async_call(
                "weather",
                "get_forecasts",
                {"entity_id": weather_entity, "type": "hourly"},
                blocking=True,
                return_response=True,
            )
        except Exception as err:  # noqa: BLE001 - hot-day gate degrades off
            _LOGGER.debug("Weather forecast unavailable for ILP: %s", err)
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
        return max(temps) if temps else None

    def _apply_dwell(self, computed: str, now: datetime) -> str:
        previous, since = self._effective_action, self._action_since
        action = computed
        if (
            previous is not None
            and action != previous
            and since is not None
            and now - since < MIN_ACTION_DWELL
        ):
            action = previous
        if action != previous:
            self._effective_action = action
            self._action_since = now
        return action

    async def _async_update_data(self) -> IlpData:
        mode = self.mode
        if mode == MODE_OFF:
            raise UpdateFailed("ilp module is off")

        pricing = self._pricing.data
        future_all_in = (
            [p.all_in_cents_per_kwh for p in pricing.periods] if pricing else []
        )
        grid_power = self._float_state("grid_power_entity") or 0.0
        room_temp = self._float_state("room_temp_entity")
        inputs = IlpInputs(
            room_temp=room_temp,
            grid_export_w=max(0.0, grid_power),
            future_all_in=future_all_in,
            outdoor_forecast_max_24h=await self._forecast_max_temp_24h(),
            currently_cooling=self._effective_action == ACTION_COOL,
        )
        result = compute_ilp_action(inputs, self._config)
        effective = self._apply_dwell(result.action, dt_util.now())

        applied: dict[str, Any] | None = None
        if mode == MODE_CONTROL:
            applied = await self._async_apply(effective)

        return IlpData(
            result=result,
            effective_action=effective,
            mode=mode,
            room_temp=room_temp,
            applied=applied,
        )

    async def _async_apply(self, action: str) -> dict[str, Any]:
        entity_id = str(self._option("ilp_climate_entity"))
        state = self.hass.states.get(entity_id)
        current_mode = state.state if state else None
        desired_mode = "cool" if action == ACTION_COOL else "off"
        if current_mode == desired_mode and action != ACTION_COOL:
            return {"success": True, "action": action, "written": False}
        # a mode we didn't set (dry, heat, manual cool) is a human choice:
        # respected for the hold window, then the plan takes over again;
        # every detection is counted as preference-learning input
        if self._override.suppressed(
            current_mode, desired_mode, dt_util.now(), self._override_hold()
        ):
            return {
                "success": True,
                "action": action,
                "written": False,
                "manual_override_until": self._override.until.isoformat()
                if self._override.until
                else None,
            }
        try:
            if action == ACTION_COOL:
                if current_mode != "cool":
                    await self.hass.services.async_call(
                        "climate",
                        "set_hvac_mode",
                        {"entity_id": entity_id, "hvac_mode": "cool"},
                        blocking=True,
                    )
                await self.hass.services.async_call(
                    "climate",
                    "set_temperature",
                    {
                        "entity_id": entity_id,
                        "temperature": self._config.cool_target_temp,
                    },
                    blocking=True,
                )
            elif current_mode != "off":
                await self.hass.services.async_call(
                    "climate",
                    "set_hvac_mode",
                    {"entity_id": entity_id, "hvac_mode": "off"},
                    blocking=True,
                )
        except Exception as err:  # noqa: BLE001 - report, retry next tick
            _LOGGER.warning("ILP write failed: %s", err)
            return {"success": False, "error": str(err)}
        self._override.record_write(desired_mode)
        return {"success": True, "action": action, "written": True}
