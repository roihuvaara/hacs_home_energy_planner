"""Config and options flow for Home Energy Planner."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback

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
)


def _options_schema(current: dict[str, Any]) -> vol.Schema:
    def _default(key: str, fallback: Any) -> Any:
        return current.get(key, fallback)

    return vol.Schema(
        {
            vol.Optional(CONF_AREA, default=_default(CONF_AREA, DEFAULT_AREA)): str,
            vol.Optional(
                CONF_CURRENCY, default=_default(CONF_CURRENCY, DEFAULT_CURRENCY)
            ): str,
            vol.Optional(
                CONF_VAT_RATE_PCT,
                default=_default(CONF_VAT_RATE_PCT, DEFAULT_VAT_RATE_PCT),
            ): vol.Coerce(float),
            vol.Optional(
                CONF_MARGIN_CENTS,
                default=_default(CONF_MARGIN_CENTS, DEFAULT_MARGIN_CENTS),
            ): vol.Coerce(float),
            vol.Optional(
                CONF_DAY_TRANSFER_CENTS,
                default=_default(CONF_DAY_TRANSFER_CENTS, DEFAULT_DAY_TRANSFER_CENTS),
            ): vol.Coerce(float),
            vol.Optional(
                CONF_NIGHT_TRANSFER_CENTS,
                default=_default(
                    CONF_NIGHT_TRANSFER_CENTS, DEFAULT_NIGHT_TRANSFER_CENTS
                ),
            ): vol.Coerce(float),
            vol.Optional(
                CONF_DAY_START_HOUR,
                default=_default(CONF_DAY_START_HOUR, DEFAULT_DAY_START_HOUR),
            ): vol.All(vol.Coerce(int), vol.Range(min=0, max=23)),
            vol.Optional(
                CONF_DAY_END_HOUR,
                default=_default(CONF_DAY_END_HOUR, DEFAULT_DAY_END_HOUR),
            ): vol.All(vol.Coerce(int), vol.Range(min=0, max=24)),
            vol.Optional(
                CONF_TOMORROW_FETCH_HOUR,
                default=_default(CONF_TOMORROW_FETCH_HOUR, DEFAULT_TOMORROW_FETCH_HOUR),
            ): vol.All(vol.Coerce(int), vol.Range(min=0, max=23)),
            vol.Optional(
                CONF_NORDPOOL_CONFIG_ENTRY_ID,
                default=_default(CONF_NORDPOOL_CONFIG_ENTRY_ID, ""),
            ): str,
            vol.Optional(
                "battery_mode", default=_default("battery_mode", "observe")
            ): vol.In(["off", "observe", "control"]),
            vol.Optional(
                "climate_mode", default=_default("climate_mode", "observe")
            ): vol.In(["off", "observe", "control"]),
            vol.Optional(
                "water_heater_mode", default=_default("water_heater_mode", "observe")
            ): vol.In(["off", "observe", "control"]),
            vol.Optional(
                "ilp_mode", default=_default("ilp_mode", "observe")
            ): vol.In(["off", "observe", "control"]),
            vol.Optional(
                "battery_export", default=_default("battery_export", True)
            ): bool,
            vol.Optional(
                "export_min_multiple",
                default=_default("export_min_multiple", 2.0),
            ): vol.All(vol.Coerce(float), vol.Range(min=1.0, max=10.0)),
            vol.Optional(
                "contracts_json", default=_default("contracts_json", "")
            ): str,
            vol.Optional(
                "export_contracts_json",
                default=_default("export_contracts_json", ""),
            ): str,
            vol.Optional(
                "tank_min_c", default=_default("tank_min_c", 50.0)
            ): vol.Coerce(float),
            vol.Optional(
                "tank_max_c", default=_default("tank_max_c", 66.0)
            ): vol.Coerce(float),
            vol.Optional(
                "tank_daily_draw_kwh",
                default=_default("tank_daily_draw_kwh", 1.0),
            ): vol.Coerce(float),
            vol.Optional(
                "water_heater_source",
                default=_default("water_heater_source", "milp"),
            ): vol.In(["rules", "milp"]),
            vol.Optional(
                "versati_cooling", default=_default("versati_cooling", "notify")
            ): vol.In(["off", "notify", "control"]),
            vol.Optional(
                "manual_override_hours",
                default=_default("manual_override_hours", 4.0),
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=48)),
            # periodic cell-balance charge (balance.py). soft = when we
            # start looking for a cheap moment, hard = when we stop caring
            # what it costs; premium_frac caps what "not caring" means.
            vol.Optional(
                "balance_enabled", default=_default("balance_enabled", True)
            ): bool,
            vol.Optional(
                "balance_soft_days", default=_default("balance_soft_days", 14.0)
            ): vol.All(vol.Coerce(float), vol.Range(min=1, max=180)),
            vol.Optional(
                "balance_hard_days", default=_default("balance_hard_days", 21.0)
            ): vol.All(vol.Coerce(float), vol.Range(min=2, max=365)),
            vol.Optional(
                "balance_max_premium_frac",
                default=_default("balance_max_premium_frac", 0.5),
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=5)),
        }
    )


class HomeEnergyPlannerConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()
        if user_input is not None:
            return self.async_create_entry(title="Home Energy Planner", data=user_input)
        return self.async_show_form(step_id="user", data_schema=_options_schema({}))

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> HomeEnergyPlannerOptionsFlow:
        return HomeEnergyPlannerOptionsFlow()


class HomeEnergyPlannerOptionsFlow(config_entries.OptionsFlow):
    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)
        current = {**self.config_entry.data, **self.config_entry.options}
        return self.async_show_form(step_id="init", data_schema=_options_schema(current))
