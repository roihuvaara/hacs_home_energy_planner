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
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .climate_core import (
    ClimateConfig,
    ClimateInputs,
    ClimateResult,
    ForecastHour,
    compute_climate_target,
    project_targets,
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
    "grid_power_entity": "sensor.solis_power_grid_total_power",
    "water_inlet_entity": "sensor.pannuhuone_gree_versati_water_inlet_temperature",
    "water_outlet_entity": "sensor.pannuhuone_gree_versati_water_outlet_temperature",
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
        projection: dict[str, Any] | None = None,
    ) -> None:
        self.result = result
        self.mode = mode
        self.room_temp = room_temp
        self.lead_hold_until = lead_hold_until
        self.legacy_target = legacy_target
        self.cooling = cooling
        self.applied = applied
        self.projection = projection


class ClimateCoordinator(DataUpdateCoordinator[ClimateData]):
    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        pricing: PricingCoordinator,
        preferences=None,
    ) -> None:
        super().__init__(hass, _LOGGER, name=f"{DOMAIN} climate", update_interval=None)
        self._entry = entry
        self._pricing = pricing
        self._config = ClimateConfig()
        self._preferences = preferences
        self._tick_context: dict[str, Any] = {}
        self._lead_hold_until: datetime | None = None
        self._last_write: tuple[float, datetime] | None = None
        self._trial_until: datetime | None = None
        self._trial_revert_cancel: Any | None = None
        self._last_test_notify = None
        self._regime: str | None = None
        self._regime_since: datetime | None = None
        self._regime_reason: str = ""
        self._last_regime_eval: tuple | None = None
        # regime survives restarts/reloads: a fresh initial pick inside a
        # hysteresis band could silently flip HEAT->NEUTRAL (pump off)
        self._regime_store: Store = Store(
            hass, 1, f"{DOMAIN}.climate_regime_{entry.entry_id}"
        )
        from .manual_override import ManualOverrideTracker

        self._override = ManualOverrideTracker()
        self._hvac_override = ManualOverrideTracker()
        from .override_store import OverridePersistence

        self._override_persist = OverridePersistence(
            hass,
            entry.entry_id,
            "climate",
            {"target": self._override, "hvac": self._hvac_override},
        )

    async def async_restore_overrides(self) -> None:
        await self._override_persist.async_restore()

    @property
    def regime(self) -> str | None:
        return self._regime

    @property
    def preferences(self):
        return self._preferences

    def _regime_store_data(self) -> dict[str, Any]:
        return {
            "regime": self._regime,
            "regime_since": self._regime_since.isoformat()
            if self._regime_since
            else None,
            "reason": self._regime_reason,
        }

    async def async_restore_regime(self) -> None:
        from .climate_core import REGIME_COOL, REGIME_HEAT, REGIME_NEUTRAL

        try:
            data = await self._regime_store.async_load()
        except Exception as err:  # noqa: BLE001 - corrupt store: start fresh
            _LOGGER.warning("Regime restore failed: %s", err)
            return
        if not data:
            return
        regime = data.get("regime")
        if regime not in (REGIME_HEAT, REGIME_NEUTRAL, REGIME_COOL):
            return
        self._regime = regime
        since = data.get("regime_since")
        self._regime_since = dt_util.parse_datetime(since) if since else None
        self._regime_reason = f"restored: {data.get('reason') or regime}"

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
        # 72 h: the regime machine needs the multi-day mean; downstream
        # consumers slice their own shorter windows
        for row in rows[:72]:
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

    async def _room_mean_24h(self) -> float | None:
        from homeassistant.components.recorder.statistics import (
            statistics_during_period,
        )

        entity_id = str(self._option("room_temp_entity"))
        now = dt_util.now()
        rows = (
            await self.hass.async_add_executor_job(
                statistics_during_period,
                self.hass,
                now - timedelta(hours=24),
                now,
                {entity_id},
                "hour",
                None,
                {"mean"},
            )
        ).get(entity_id, [])
        means = [row["mean"] for row in rows if row.get("mean") is not None]
        return sum(means) / len(means) if means else None

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
        # extending/restarting a trial must kill the previous revert timer,
        # or the old one still fires and flips the pump back mid-trial
        if self._trial_revert_cancel is not None:
            self._trial_revert_cancel()
            self._trial_revert_cancel = None
        self._trial_until = dt_util.now() + timedelta(minutes=minutes)

        async def _revert(_now: Any) -> None:
            self._trial_until = None
            self._trial_revert_cancel = None
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

        self._trial_revert_cancel = async_call_later(
            self.hass, timedelta(minutes=minutes), _revert
        )
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
        pref_offset = (
            self._preferences.adjustment("climate_target_offset")
            if self._preferences is not None
            else 0.0
        )
        if pref_offset:
            from dataclasses import replace

            result = replace(
                result,
                target=round(
                    min(
                        self._config.max_target,
                        max(self._config.min_target, result.target + pref_offset),
                    ),
                    1,
                ),
            )

        # projected target over the price horizon for dashboard plotting;
        # includes the learned offset so quarter 0 matches the live target
        projection: dict[str, Any] | None = None
        if pricing and pricing.periods:
            projection = {
                "start": pricing.periods[0].start.isoformat(),
                "period_minutes": 15,
                "targets": project_targets(inputs, self._config, pref_offset),
            }

        # thermal regime: slow inputs only (forecasts + 24 h room mean)
        from .climate_core import (
            REGIME_COOL,
            REGIME_HEAT,
            cool_active_now,
            cool_water_target,
            decide_regime,
        )

        humidity = self._float_state("room_humidity_entity")
        forecast_max = max((h.temperature for h in hours[:24]), default=None)
        forecast_mean_72h = (
            sum(h.temperature for h in hours) / len(hours) if hours else None
        )
        forecast_avg_12h = (
            sum(h.temperature for h in hours[:12]) / len(hours[:12])
            if hours[:12]
            else None
        )
        eval_key = (now.date(), now.hour)
        if self._regime is None or self._last_regime_eval != eval_key:
            self._last_regime_eval = eval_key
            room_mean = await self._room_mean_24h()
            new_regime, reason = decide_regime(
                self._regime,
                forecast_avg_12h,
                forecast_mean_72h,
                forecast_max,
                room_mean,
                self._config,
            )
            changed = new_regime != self._regime
            if changed:
                self._regime_since = now
            self._regime = new_regime
            self._regime_reason = reason
            if changed:
                self._regime_store.async_delay_save(self._regime_store_data, 10)

        water_target, dew = cool_water_target(
            inputs.room_temp, humidity, forecast_max, self._config
        )
        cooling_mode = str(self._option("versati_cooling"))
        export_w = max(0.0, self._float_state("grid_power_entity") or 0.0)
        cool_active = self._regime == REGIME_COOL and cool_active_now(
            now.hour, export_w, self._config
        )
        cooling = {
            "rollout": cooling_mode,
            "regime": self._regime,
            "regime_since": self._regime_since.isoformat()
            if self._regime_since
            else None,
            "regime_reason": self._regime_reason,
            "cool_active": cool_active,
            "forecast_max_24h": forecast_max,
            "forecast_mean_72h": round(forecast_mean_72h, 1)
            if forecast_mean_72h is not None
            else None,
            "water_target": water_target,
            "dew_point": dew,
            "water_inlet": self._float_state("water_inlet_entity"),
            "water_outlet": self._float_state("water_outlet_entity"),
            "trial_until": self._trial_until.isoformat() if self._trial_until else None,
        }
        if (
            cooling_mode == "notify"
            and self._regime == REGIME_COOL
            and 7 <= now.hour < 12
            and self._last_test_notify != now.date()
        ):
            self._last_test_notify = now.date()
            await self._async_notify(
                "Hydronic cooling regime active — trial day",
                f"Regime COOL ({self._regime_reason}); forecast max {forecast_max:.0f} C. "
                f"Run home_energy_planner.versati_cool_trial (water target {water_target} C, "
                f"dew point {dew} C) and check the pannuhuone manifold for condensation, "
                "then set versati_cooling: control to let the planner run it.",
            )

        self._tick_context = {
            "room_temp": inputs.room_temp,
            "regime": self._regime,
            "price_position": result.price.position,
        }

        applied: dict[str, Any] | None = None
        if mode == MODE_CONTROL and self._trial_until is None:
            if cooling_mode == "off" or self._regime in (None, REGIME_HEAT):
                applied = await self._async_apply_regime_heat(result.target)
            elif self._regime == REGIME_COOL and cooling_mode == "control":
                applied = await self._async_apply_regime_cool(
                    water_target, cool_active
                )
            else:
                # NEUTRAL, or COOL while cooling writes are gated (notify)
                applied = await self._async_ensure_hvac("off") or {
                    "success": True,
                    "hvac": "off",
                }
            self._override_persist.save()

        return ClimateData(
            result=result,
            mode=mode,
            room_temp=inputs.room_temp,
            lead_hold_until=self._lead_hold_until,
            legacy_target=self._float_state("legacy_climate_target_entity"),
            cooling=cooling,
            applied=applied,
            projection=projection,
        )

    async def _async_ensure_hvac(self, desired: str) -> dict[str, Any] | None:
        """Set hvac mode when different; manual mode changes get the
        override grace window like every other manual control."""
        climate_entity = str(self._option("climate_entity"))
        state = self.hass.states.get(climate_entity)
        hvac = state.state if state else None
        if hvac == desired or hvac is None:
            return None
        provenance_known = self._hvac_override.last_written is not None
        count_before = self._hvac_override.count
        if self._hvac_override.suppressed(
            hvac, desired, dt_util.now(), self._override_hold()
        ):
            if (
                self._preferences is not None
                and provenance_known
                and self._hvac_override.count > count_before
            ):
                self._preferences.record(
                    "climate_hvac", desired, hvac, dict(self._tick_context)
                )
            return {
                "success": True,
                "hvac": hvac,
                "manual_override_until": self._hvac_override.until.isoformat()
                if self._hvac_override.until
                else None,
            }
        try:
            await self.hass.services.async_call(
                "climate",
                "set_hvac_mode",
                {"entity_id": climate_entity, "hvac_mode": desired},
                blocking=True,
            )
        except Exception as err:  # noqa: BLE001 - retry next tick
            _LOGGER.warning("hvac mode write failed: %s", err)
            return {"success": False, "error": str(err)}
        self._hvac_override.record_write(desired)
        return {"success": True, "hvac": desired, "written": True}

    async def _async_apply_regime_heat(self, heat_target: float) -> dict[str, Any]:
        ensured = await self._async_ensure_hvac("heat")
        if ensured is not None and not ensured.get("written"):
            return ensured  # blocked by manual override or write failure
        return await self._async_apply(heat_target)

    async def _async_apply_regime_cool(
        self, water_target: float, cool_active: bool
    ) -> dict[str, Any]:
        """COOL regime: cool mode in the night/surplus windows, off between."""
        if not cool_active:
            ensured = await self._async_ensure_hvac("off")
            return ensured or {"success": True, "hvac": "off"}
        ensured = await self._async_ensure_hvac("cool")
        if ensured is not None and not ensured.get("written"):
            return ensured
        climate_entity = str(self._option("climate_entity"))
        state = self.hass.states.get(climate_entity)
        current = state.attributes.get("temperature") if state else None
        try:
            if current is not None and float(current) == water_target:
                return {"success": True, "hvac": "cool", "water_target": water_target}
        except (TypeError, ValueError):
            pass
        await self.hass.services.async_call(
            "climate",
            "set_temperature",
            {"entity_id": climate_entity, "temperature": water_target},
            blocking=True,
        )
        return {
            "success": True,
            "hvac": "cool",
            "water_target": water_target,
            "written": True,
        }

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
        provenance_known = self._override.last_written is not None
        count_before = self._override.count
        if self._override.suppressed(
            current_value, target, now_check, self._override_hold()
        ):
            if (
                self._preferences is not None
                and provenance_known
                and self._override.count > count_before
            ):
                self._preferences.record(
                    "climate", target, current_value, dict(self._tick_context)
                )
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
