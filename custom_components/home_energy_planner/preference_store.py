"""Store-backed preference log shared by the module coordinators.

Thin HA glue over the pure `preference` module: loads/saves the event
log, keeps an hourly-refreshed weather-context snapshot, stamps it onto
every recorded event (history cannot be retrofitted — capture wide, the
kernel decides later what steers), and recomputes the learned
adjustments both on new events and as the live context drifts through
the seasons.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .preference import (
    PreferenceEvent,
    PreferenceLog,
    daylight_hours,
    derive_adjustments,
)

_LOGGER = logging.getLogger(__name__)

OUTDOOR_TEMP_ENTITY = "sensor.ilp_ulkolampotila"
WEATHER_ENTITY = "weather.forecast_koti"


class PreferenceStore:
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._store: Store = Store(
            hass, 1, f"{DOMAIN}.preferences_{entry.entry_id}"
        )
        self.log = PreferenceLog()
        self.weather_context: dict[str, Any] = {}
        self._adjustments: dict[str, float] = derive_adjustments([], dt_util.now())

    async def async_load(self) -> None:
        try:
            data = await self._store.async_load()
        except Exception as err:  # noqa: BLE001 - corrupt store: start fresh
            _LOGGER.warning("Preference restore failed: %s", err)
            return
        if data:
            self.log = PreferenceLog.from_dict(data)
        await self.async_refresh_context()

    def async_schedule_context_refresh(self) -> None:
        """Hourly context refresh: the learned offsets must track the
        season even when no new events arrive."""

        @callback
        def _tick(_now) -> None:
            self.hass.async_create_task(self.async_refresh_context())

        self._entry.async_on_unload(
            async_track_time_change(self.hass, _tick, minute=7, second=0)
        )

    def _option(self, key: str, default: str) -> str:
        return str(
            self._entry.options.get(key, self._entry.data.get(key, default))
        )

    def _float_state(self, entity_id: str, attribute: str | None = None):
        state = self.hass.states.get(entity_id)
        if state is None:
            return None
        raw = state.attributes.get(attribute) if attribute else state.state
        try:
            return float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    async def _outdoor_means(self, now) -> tuple[float | None, float | None]:
        """7- and 30-day trailing outdoor means from recorder daily stats."""
        from homeassistant.components.recorder.statistics import (
            statistics_during_period,
        )

        entity_id = self._option("outdoor_temp_entity", OUTDOOR_TEMP_ENTITY)
        rows = (
            await self.hass.async_add_executor_job(
                statistics_during_period,
                self.hass,
                now - timedelta(days=30),
                now,
                {entity_id},
                "day",
                None,
                {"mean"},
            )
        ).get(entity_id, [])
        means: list[tuple[Any, float]] = []
        for row in rows:
            mean = row.get("mean")
            if mean is not None:
                means.append((row.get("start"), float(mean)))
        if not means:
            return None, None
        mean_30d = sum(value for _s, value in means) / len(means)
        last7 = means[-7:]
        mean_7d = sum(value for _s, value in last7) / len(last7)
        return round(mean_7d, 2), round(mean_30d, 2)

    async def async_refresh_context(self) -> None:
        """Rebuild the weather-context snapshot; failures degrade to a
        partial context, never break recording or folding."""
        now = dt_util.now()
        context: dict[str, Any] = {}
        try:
            outdoor = self._float_state(
                self._option("outdoor_temp_entity", OUTDOOR_TEMP_ENTITY)
            )
            if outdoor is None:  # Gree reads unknown while the unit is off
                outdoor = self._float_state(
                    self._option("weather_entity", WEATHER_ENTITY), "temperature"
                )
            if outdoor is not None:
                context["outdoor_temp"] = round(outdoor, 1)

            mean_7d, mean_30d = await self._outdoor_means(now)
            if mean_7d is not None:
                context["outdoor_mean_7d"] = mean_7d
            if mean_30d is not None:
                context["outdoor_mean_30d"] = mean_30d
            if mean_7d is not None and mean_30d is not None:
                context["temp_trend"] = round(mean_7d - mean_30d, 2)

            latitude = float(self.hass.config.latitude)
            today = now.date()
            light_today = daylight_hours(latitude, today)
            context["daylight_hours"] = round(light_today, 2)
            context["daylight_trend"] = round(
                light_today - daylight_hours(latitude, today - timedelta(days=7)),
                2,
            )

            weather = self.hass.states.get(
                self._option("weather_entity", WEATHER_ENTITY)
            )
            if weather is not None and weather.state not in (
                "unknown",
                "unavailable",
            ):
                context["weather_condition"] = weather.state

            context["solar_remaining_today_kwh"] = await self._solar_remaining(now)
        except Exception as err:  # noqa: BLE001 - context is best-effort
            _LOGGER.debug("Weather context refresh partial: %s", err)
        self.weather_context = context
        self._recompute()

    async def _solar_remaining(self, now) -> float | None:
        try:
            from .solar_forecast import async_wh_by_hour

            wh_by_hour = await async_wh_by_hour(self.hass, now.tzinfo)
            total = sum(
                wh
                for hour, wh in wh_by_hour.items()
                if hour.date() == now.date() and hour >= now.replace(
                    minute=0, second=0, microsecond=0
                )
            )
            return round(total / 1000.0, 2)
        except Exception:  # noqa: BLE001
            return None

    def record(
        self,
        module: str,
        planner_value: object,
        manual_value: object,
        context: dict[str, Any],
    ) -> bool:
        event = PreferenceEvent(
            when=dt_util.now(),
            module=module,
            planner_value=planner_value,
            manual_value=manual_value,
            # module context wins on key collisions; weather features fill in
            context={**self.weather_context, **context},
        )
        stored = self.log.append(event)
        if stored:
            _LOGGER.info(
                "Preference event: %s planner=%s manual=%s (%s)",
                module,
                planner_value,
                manual_value,
                event.context,
            )
            self._recompute()
            self._store.async_delay_save(self.log.as_dict, 10)
        return stored

    def _recompute(self) -> None:
        self._adjustments = derive_adjustments(
            self.log.events, dt_util.now(), self.weather_context
        )

    @property
    def adjustments(self) -> dict[str, float]:
        return self._adjustments

    def adjustment(self, key: str) -> float:
        return float(self._adjustments.get(key, 0.0))

    def summary(self) -> dict[str, Any]:
        """Compact shape for sensor attributes / the monthly report."""
        return {
            "event_counts": self.log.counts_by_module(),
            "adjustments": dict(self._adjustments),
            "context": dict(self.weather_context),
        }
