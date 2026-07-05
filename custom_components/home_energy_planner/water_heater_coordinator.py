"""Hot water coordinator: computes the control mode, observes or controls.

Modes (config option `water_heater_mode`):
- off: no computation
- observe: compute and publish the mode every quarter, write nothing
- control: additionally write the tank target to the Versati water heater
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_change,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import DOMAIN
from .coordinator import PricingCoordinator
from .water_heater_core import (
    WaterHeaterConfig,
    WaterHeaterInputs,
    WaterHeaterResult,
    compute_water_heater_mode,
)

_LOGGER = logging.getLogger(__name__)

CONF_WATER_HEATER_MODE = "water_heater_mode"
MODE_OFF = "off"
MODE_OBSERVE = "observe"
MODE_CONTROL = "control"

DEFAULTS = {
    "solar_power_now_entity": "sensor.power_production_now",
    "solar_power_next_hour_entity": "sensor.power_production_next_hour",
    "solar_remaining_today_entity": "sensor.energy_production_today_remaining",
    "battery_soc_entity": "sensor.solis_remaining_battery_capacity",
    "water_heater_entity": "water_heater.gree_versati_water_heater_2",
    "legacy_water_heater_mode_entity": "input_select.water_heater_control_mode",
}


class WaterHeaterData:
    def __init__(
        self,
        result: WaterHeaterResult,
        mode: str,
        legacy_mode: str | None,
        applied: dict[str, Any] | None,
    ) -> None:
        self.result = result
        self.mode = mode
        self.legacy_mode = legacy_mode
        self.applied = applied


class WaterHeaterCoordinator(DataUpdateCoordinator[WaterHeaterData]):
    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        pricing: PricingCoordinator,
    ) -> None:
        super().__init__(
            hass, _LOGGER, name=f"{DOMAIN} water_heater", update_interval=None
        )
        self._entry = entry
        self._pricing = pricing
        self._config = WaterHeaterConfig()

    def _option(self, key: str) -> Any:
        return self._entry.options.get(
            key, self._entry.data.get(key, DEFAULTS.get(key))
        )

    @property
    def mode(self) -> str:
        return str(
            self._entry.options.get(
                CONF_WATER_HEATER_MODE,
                self._entry.data.get(CONF_WATER_HEATER_MODE, MODE_OBSERVE),
            )
        )

    def async_schedule_ticks(self) -> None:
        @callback
        def _tick(_now: datetime) -> None:
            self.hass.async_create_task(self.async_request_refresh())

        self._entry.async_on_unload(
            async_track_time_change(
                self.hass, _tick, minute=[0, 15, 30, 45], second=50
            )
        )

        @callback
        def _input_changed(_event) -> None:
            self.hass.async_create_task(self.async_request_refresh())

        self._entry.async_on_unload(
            async_track_state_change_event(
                self.hass,
                [
                    str(self._option("solar_power_now_entity")),
                    str(self._option("battery_soc_entity")),
                ],
                _input_changed,
            )
        )

    def _float_state(self, key: str, fallback: float = 0.0) -> float:
        state = self.hass.states.get(str(self._option(key)))
        try:
            return float(state.state)  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            return fallback

    async def _async_update_data(self) -> WaterHeaterData:
        mode = self.mode
        if mode == MODE_OFF:
            raise UpdateFailed("water heater module is off")

        pricing = self._pricing.data
        future_all_in = (
            [p.all_in_cents_per_kwh for p in pricing.periods] if pricing else []
        )
        inputs = WaterHeaterInputs(
            current_vat=(
                pricing.periods[0].vat_cents_per_kwh
                if pricing and pricing.periods
                else 0.0
            ),
            current_all_in=future_all_in[0] if future_all_in else 0.0,
            future_all_in=future_all_in,
            solar_now_w=self._float_state("solar_power_now_entity"),
            solar_next_hour_w=self._float_state("solar_power_next_hour_entity"),
            solar_remaining_today_kwh=self._float_state(
                "solar_remaining_today_entity"
            ),
            battery_soc_pct=self._float_state("battery_soc_entity"),
        )
        result = compute_water_heater_mode(inputs, self._config)

        applied: dict[str, Any] | None = None
        if mode == MODE_CONTROL:
            applied = await self._async_apply(result.target_temp)

        legacy_state = self.hass.states.get(
            str(self._option("legacy_water_heater_mode_entity"))
        )
        return WaterHeaterData(
            result=result,
            mode=mode,
            legacy_mode=legacy_state.state if legacy_state else None,
            applied=applied,
        )

    async def _async_apply(self, target: int) -> dict[str, Any]:
        entity_id = str(self._option("water_heater_entity"))
        state = self.hass.states.get(entity_id)
        current = state.attributes.get("temperature") if state else None
        try:
            unchanged = current is not None and int(float(current)) == target
        except (TypeError, ValueError):
            unchanged = False
        if unchanged:
            return {"success": True, "written": False, "target": target}
        try:
            await self.hass.services.async_call(
                "water_heater",
                "set_temperature",
                {"entity_id": entity_id, "temperature": target},
                blocking=True,
            )
        except Exception as err:  # noqa: BLE001 - report, retry next tick
            _LOGGER.warning("Water heater target write failed: %s", err)
            return {"success": False, "written": False, "error": str(err)}
        return {"success": True, "written": True, "target": target}
