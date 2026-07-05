"""Pricing coordinator: fetches Nord Pool slots and publishes the horizon."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import (
    CONF_AREA,
    CONF_CURRENCY,
    CONF_DAY_END_HOUR,
    CONF_DAY_START_HOUR,
    CONF_DAY_TRANSFER_CENTS,
    CONF_MARGIN_CENTS,
    CONF_NIGHT_TRANSFER_CENTS,
    CONF_NORDPOOL_CONFIG_ENTRY_ID,
    CONF_TOMORROW_FETCH_HOUR,
    CONF_VAT_RATE_PCT,
    DEFAULT_AREA,
    DEFAULT_CURRENCY,
    DEFAULT_DAY_END_HOUR,
    DEFAULT_DAY_START_HOUR,
    DEFAULT_DAY_TRANSFER_CENTS,
    DEFAULT_MARGIN_CENTS,
    DEFAULT_NIGHT_TRANSFER_CENTS,
    DEFAULT_TOMORROW_FETCH_HOUR,
    DEFAULT_VAT_RATE_PCT,
    DOMAIN,
    NORDPOOL_DOMAIN,
    NORDPOOL_RESOLUTION_MINUTES,
    NORDPOOL_SERVICE_GET_PRICES,
)
from .pricing import (
    PERIOD_MINUTES,
    PricePeriod,
    PricingConfig,
    RawSlot,
    build_price_horizon,
    horizon_is_contiguous,
)

_LOGGER = logging.getLogger(__name__)

RETRY_INTERVAL = timedelta(minutes=5)


@dataclass(frozen=True)
class PricingData:
    periods: list[PricePeriod]
    horizon_start: datetime
    tomorrow_included: bool
    contiguous: bool


class PricingCoordinator(DataUpdateCoordinator[PricingData]):
    """Quarter-hour aligned price horizon built from the nordpool integration."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN} pricing",
            update_interval=RETRY_INTERVAL,
        )
        self._entry = entry
        self._raw_slots_by_date: dict[date, list[RawSlot]] = {}

    def async_schedule_quarter_ticks(self) -> None:
        """Refresh right after every quarter-hour boundary.

        The 5-minute update_interval acts as the retry/backstop path; this
        listener makes the horizon roll over exactly on the slot boundary.
        """

        @callback
        def _on_quarter(_now: datetime) -> None:
            self.hass.async_create_task(self.async_request_refresh())

        self._entry.async_on_unload(
            async_track_time_change(
                self.hass,
                _on_quarter,
                minute=[0, 15, 30, 45],
                second=2,
            )
        )

    def _option(self, key: str, default: Any) -> Any:
        return self._entry.options.get(key, self._entry.data.get(key, default))

    def pricing_config(self) -> PricingConfig:
        return PricingConfig(
            vat_rate=float(self._option(CONF_VAT_RATE_PCT, DEFAULT_VAT_RATE_PCT)) / 100.0,
            margin_cents_per_kwh=float(self._option(CONF_MARGIN_CENTS, DEFAULT_MARGIN_CENTS)),
            day_transfer_cents_per_kwh=float(
                self._option(CONF_DAY_TRANSFER_CENTS, DEFAULT_DAY_TRANSFER_CENTS)
            ),
            night_transfer_cents_per_kwh=float(
                self._option(CONF_NIGHT_TRANSFER_CENTS, DEFAULT_NIGHT_TRANSFER_CENTS)
            ),
            day_start_hour=int(self._option(CONF_DAY_START_HOUR, DEFAULT_DAY_START_HOUR)),
            day_end_hour=int(self._option(CONF_DAY_END_HOUR, DEFAULT_DAY_END_HOUR)),
        )

    def _nordpool_entry_id(self) -> str:
        configured = str(self._option(CONF_NORDPOOL_CONFIG_ENTRY_ID, "") or "").strip()
        if configured:
            return configured
        entries = self.hass.config_entries.async_entries(NORDPOOL_DOMAIN)
        loaded = [entry for entry in entries if entry.state is ConfigEntryState.LOADED]
        candidates = loaded or entries
        if not candidates:
            raise UpdateFailed("No nordpool config entry found")
        return candidates[0].entry_id

    async def async_fetch_day(self, target_date: date) -> list[RawSlot]:
        area = str(self._option(CONF_AREA, DEFAULT_AREA))
        response = await self.hass.services.async_call(
            NORDPOOL_DOMAIN,
            NORDPOOL_SERVICE_GET_PRICES,
            {
                "config_entry": self._nordpool_entry_id(),
                "date": target_date.isoformat(),
                "areas": area,
                "currency": str(self._option(CONF_CURRENCY, DEFAULT_CURRENCY)),
                "resolution": NORDPOOL_RESOLUTION_MINUTES,
            },
            blocking=True,
            return_response=True,
        )
        rows = response.get(area, []) if isinstance(response, dict) else []
        return [
            RawSlot(
                start=dt_util.parse_datetime(str(row["start"])),
                price_eur_per_mwh=float(row["price"]),
            )
            for row in rows
            if row.get("start") is not None and row.get("price") is not None
        ]

    async def _async_update_data(self) -> PricingData:
        now = dt_util.now()
        today = now.date()
        tomorrow = today + timedelta(days=1)
        self._raw_slots_by_date = {
            slot_date: slots
            for slot_date, slots in self._raw_slots_by_date.items()
            if slot_date >= today
        }

        try:
            self._raw_slots_by_date[today] = await self.async_fetch_day(today)
        except UpdateFailed:
            raise
        except Exception as err:
            if today not in self._raw_slots_by_date:
                raise UpdateFailed(f"Fetching today's prices failed: {err}") from err
            _LOGGER.warning("Reusing cached prices for %s: %s", today, err)

        tomorrow_fetch_hour = int(
            self._option(CONF_TOMORROW_FETCH_HOUR, DEFAULT_TOMORROW_FETCH_HOUR)
        )
        if now.hour >= tomorrow_fetch_hour and tomorrow not in self._raw_slots_by_date:
            try:
                slots = await self.async_fetch_day(tomorrow)
                if slots:
                    self._raw_slots_by_date[tomorrow] = slots
            except Exception as err:
                _LOGGER.debug("Tomorrow's prices not available yet: %s", err)

        raw_slots = [
            slot
            for slots in self._raw_slots_by_date.values()
            for slot in slots
            if slot.start is not None
        ]
        periods = build_price_horizon(
            raw_slots,
            now=now,
            config=self.pricing_config(),
            local_tz=dt_util.get_default_time_zone(),
        )
        if not periods:
            raise UpdateFailed("Price horizon is empty")
        if (periods[0].start + timedelta(minutes=PERIOD_MINUTES)) <= now:
            raise UpdateFailed("Price horizon does not cover the current period")
        return PricingData(
            periods=periods,
            horizon_start=periods[0].start,
            tomorrow_included=tomorrow in self._raw_slots_by_date,
            contiguous=horizon_is_contiguous(periods),
        )
