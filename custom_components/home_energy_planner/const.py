"""Constants for the Home Energy Planner integration."""

from __future__ import annotations

DOMAIN = "home_energy_planner"
PLATFORMS = ["sensor"]

CONF_NORDPOOL_CONFIG_ENTRY_ID = "nordpool_config_entry_id"
CONF_AREA = "area"
CONF_CURRENCY = "currency"
CONF_VAT_RATE_PCT = "vat_rate_pct"
CONF_MARGIN_CENTS = "margin_cents_per_kwh"
CONF_DAY_TRANSFER_CENTS = "day_transfer_cents_per_kwh"
CONF_NIGHT_TRANSFER_CENTS = "night_transfer_cents_per_kwh"
CONF_DAY_START_HOUR = "day_start_hour"
CONF_DAY_END_HOUR = "day_end_hour"
CONF_TOMORROW_FETCH_HOUR = "tomorrow_fetch_hour"

DEFAULT_AREA = "FI"
DEFAULT_CURRENCY = "EUR"
DEFAULT_VAT_RATE_PCT = 25.5
DEFAULT_MARGIN_CENTS = 2.82752
DEFAULT_DAY_TRANSFER_CENTS = 5.11
DEFAULT_NIGHT_TRANSFER_CENTS = 3.12
DEFAULT_DAY_START_HOUR = 7
DEFAULT_DAY_END_HOUR = 22
DEFAULT_TOMORROW_FETCH_HOUR = 14

NORDPOOL_DOMAIN = "nordpool"
NORDPOOL_SERVICE_GET_PRICES = "get_price_indices_for_date"
NORDPOOL_RESOLUTION_MINUTES = 15
