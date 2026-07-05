"""Shared access to Forecast.Solar's hourly Wh series via its energy platform."""

from __future__ import annotations

import logging
from datetime import datetime, tzinfo

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)


async def async_wh_by_hour(
    hass: HomeAssistant, local_tz: tzinfo
) -> dict[datetime, float]:
    """Hour start (local tz) -> forecast Wh, empty dict when unavailable."""
    try:
        from homeassistant.components.forecast_solar.energy import (
            async_get_solar_forecast,
        )
    except ImportError:
        return {}

    result: dict[datetime, float] = {}
    for entry in hass.config_entries.async_entries("forecast_solar"):
        try:
            forecast = await async_get_solar_forecast(hass, entry.entry_id)
        except Exception as err:  # noqa: BLE001 - callers have fallbacks
            _LOGGER.debug("forecast_solar data unavailable: %s", err)
            continue
        for key, wh in ((forecast or {}).get("wh_hours") or {}).items():
            ts = key if isinstance(key, datetime) else dt_util.parse_datetime(str(key))
            if ts is None:
                continue
            hour = ts.astimezone(local_tz).replace(minute=0, second=0, microsecond=0)
            result[hour] = result.get(hour, 0.0) + float(wh)
    return result
