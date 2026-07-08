"""Hot water coordinator: computes the control mode, observes or controls.

Modes (config option `water_heater_mode`):
- off: no computation
- observe: compute and publish the mode every quarter, write nothing
- control: additionally write the tank target to the Versati water heater

Surplus detection uses the measured grid power (positive = exporting on
`sensor.solis_power_grid_total_power` — verify the sign holds during the
observe gate); upcoming solar comes from the Forecast.Solar hourly series.
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
    "grid_power_entity": "sensor.solis_power_grid_total_power",
    "water_heater_entity": "water_heater.gree_versati_water_heater_2",
    "legacy_water_heater_mode_entity": "input_select.water_heater_control_mode",
}
PRESERVE_LOOKAHEAD_HOURS = 6
# compressor protection: minimum time in a boosting mode, and export
# hysteresis so surplus flutter around the entry threshold cannot flap
MIN_CHEAP_BOOST_DWELL = timedelta(minutes=45)
MIN_SOLAR_BOOST_DWELL = timedelta(minutes=30)
SOLAR_EXIT_EXPORT_W = 150.0


class WaterHeaterData:
    def __init__(
        self,
        result: WaterHeaterResult,
        effective_mode: str,
        effective_target: int,
        mode: str,
        legacy_mode: str | None,
        applied: dict[str, Any] | None,
    ) -> None:
        self.result = result
        self.effective_mode = effective_mode
        self.effective_target = effective_target
        self.mode = mode
        self.legacy_mode = legacy_mode
        self.applied = applied


class WaterHeaterCoordinator(DataUpdateCoordinator[WaterHeaterData]):
    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        pricing: PricingCoordinator,
        preferences=None,
    ) -> None:
        super().__init__(
            hass, _LOGGER, name=f"{DOMAIN} water_heater", update_interval=None
        )
        self._entry = entry
        self._pricing = pricing
        self._config = WaterHeaterConfig()
        self._preferences = preferences
        self._effective_mode: str | None = None
        self._mode_since: datetime | None = None
        from .manual_override import ManualOverrideTracker

        self._override = ManualOverrideTracker()
        # the on/off state is overridable in its own right: a manual off is
        # respected for the hold window, then reconciled back on (the tank
        # must never sit off indefinitely — it only cools and drains)
        self._power_override = ManualOverrideTracker()

    @property
    def preferences(self):
        return self._preferences

    def _override_hold(self) -> timedelta:
        return timedelta(
            hours=float(
                self._entry.options.get(
                    "manual_override_hours",
                    self._entry.data.get("manual_override_hours", 4.0),
                )
            )
        )

    def _apply_dwell(
        self, computed: str, export_w: float, now: datetime
    ) -> str:
        """Compressor-friendly mode transitions on top of the pure result.

        A boosting run, once started, keeps going for its minimum dwell
        (the pure planner window can end short when the horizon rolls);
        solar_boost persists while measurable export continues.
        """
        previous, since = self._effective_mode, self._mode_since
        mode = computed
        if previous == "solar_boost" and computed != "solar_boost":
            in_dwell = since is not None and now - since < MIN_SOLAR_BOOST_DWELL
            if export_w >= SOLAR_EXIT_EXPORT_W or (in_dwell and computed == "normal"):
                mode = "solar_boost"
        elif previous == "cheap_boost" and computed == "normal":
            if since is not None and now - since < MIN_CHEAP_BOOST_DWELL:
                mode = "cheap_boost"
        if mode != previous:
            self._effective_mode = mode
            self._mode_since = now
        return mode

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
                [str(self._option("grid_power_entity"))],
                _input_changed,
            )
        )

    def _float_state(self, key: str, fallback: float = 0.0) -> float:
        state = self.hass.states.get(str(self._option(key)))
        try:
            return float(state.state)  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            return fallback

    async def _upcoming_solar_kwh(self) -> float:
        from homeassistant.util import dt as dt_util

        from .solar_forecast import async_wh_by_hour

        now = dt_util.now()
        wh_by_hour = await async_wh_by_hour(self.hass, now.tzinfo)
        this_hour = now.replace(minute=0, second=0, microsecond=0)
        total_wh = 0.0
        for offset in range(PRESERVE_LOOKAHEAD_HOURS):
            total_wh += wh_by_hour.get(this_hour + timedelta(hours=offset), 0.0)
        return round(total_wh / 1000.0, 3)

    async def _async_update_data(self) -> WaterHeaterData:
        mode = self.mode
        if mode == MODE_OFF:
            raise UpdateFailed("water heater module is off")

        pricing = self._pricing.data
        future_all_in = (
            [p.all_in_cents_per_kwh for p in pricing.periods] if pricing else []
        )
        export_w = max(0.0, self._float_state("grid_power_entity"))
        inputs = WaterHeaterInputs(
            future_all_in=future_all_in,
            grid_export_w=export_w,
            upcoming_solar_kwh=await self._upcoming_solar_kwh(),
        )
        result = compute_water_heater_mode(inputs, self._config)

        from homeassistant.util import dt as dt_util

        effective_mode = self._apply_dwell(result.mode, export_w, dt_util.now())
        effective_target = self._config.targets[effective_mode]
        if self._preferences is not None:
            # per-weekday offset learned from tank-target overrides
            # (capped +-6 in preference.CAPS, so no further clamp needed)
            weekday_offset = self._preferences.adjustment(
                f"water_weekday_{dt_util.now().weekday()}"
            )
            if weekday_offset:
                effective_target = int(round(effective_target + weekday_offset))

        applied: dict[str, Any] | None = None
        if mode == MODE_CONTROL:
            applied = await self._async_apply(effective_target)

        legacy_state = self.hass.states.get(
            str(self._option("legacy_water_heater_mode_entity"))
        )
        return WaterHeaterData(
            result=result,
            effective_mode=effective_mode,
            effective_target=effective_target,
            mode=mode,
            legacy_mode=legacy_state.state if legacy_state else None,
            applied=applied,
        )

    async def _async_apply(self, target: int) -> dict[str, Any]:
        from homeassistant.util import dt as dt_util

        from .water_heater_core import HEATER_MODE_ON, normalized_power

        now = dt_util.now()
        hold = self._override_hold()
        entity_id = str(self._option("water_heater_entity"))
        state = self.hass.states.get(entity_id)

        # power invariant: a tank found off is respected for the hold window
        # (manual off is a real override) and then reconciled back on. Boost
        # (performance) normalizes to "on" and is never fought; a heater whose
        # mode we cannot read (None) is left alone.
        op_mode = state.attributes.get("operation_mode") if state else None
        power = normalized_power(op_mode)
        power_result: dict[str, Any] = {"operation_mode": op_mode}
        if power == "off" and not self._power_override.suppressed(
            "off", "on", now, hold
        ):
            try:
                await self.hass.services.async_call(
                    "water_heater",
                    "set_operation_mode",
                    {"entity_id": entity_id, "operation_mode": HEATER_MODE_ON},
                    blocking=True,
                )
            except Exception as err:  # noqa: BLE001 - report, retry next tick
                _LOGGER.warning("Water heater turn-on failed: %s", err)
                power_result["turn_on_error"] = str(err)
            else:
                self._power_override.record_write("on")
                power_result["turned_on"] = True
        elif power == "on":
            # observed running (normal or boost): our value holds, so a later
            # off reads as a fresh override rather than unknown provenance
            self._power_override.record_write("on")
        if self._power_override.until is not None:
            power_result["power_override_until"] = self._power_override.until.isoformat()

        current = state.attributes.get("temperature") if state else None
        try:
            current_value = int(float(current)) if current is not None else None
        except (TypeError, ValueError):
            current_value = None
        if current_value is not None and current_value == target:
            return {**power_result, "success": True, "written": False, "target": target}
        provenance_known = self._override.last_written is not None
        count_before = self._override.count
        if self._override.suppressed(current_value, target, now, hold):
            # tank-target overrides learn a per-weekday offset (weekly
            # rhythms: gym/laundry/sauna-ish days) via preference.py
            if (
                self._preferences is not None
                and provenance_known
                and self._override.count > count_before
            ):
                self._preferences.record(
                    "water_heater",
                    target,
                    current_value,
                    {"mode": self._effective_mode},
                )
            return {
                **power_result,
                "success": True,
                "written": False,
                "target": target,
                "manual_override_until": self._override.until.isoformat()
                if self._override.until
                else None,
            }
        try:
            await self.hass.services.async_call(
                "water_heater",
                "set_temperature",
                {"entity_id": entity_id, "temperature": target},
                blocking=True,
            )
        except Exception as err:  # noqa: BLE001 - report, retry next tick
            _LOGGER.warning("Water heater target write failed: %s", err)
            return {**power_result, "success": False, "written": False, "error": str(err)}
        self._override.record_write(target)
        return {**power_result, "success": True, "written": True, "target": target}
