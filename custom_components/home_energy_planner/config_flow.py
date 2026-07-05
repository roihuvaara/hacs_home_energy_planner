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
