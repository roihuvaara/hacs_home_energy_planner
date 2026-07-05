"""Home Energy Planner integration setup."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse

from .const import DOMAIN, PLATFORMS
from .coordinator import PricingCoordinator
from .solis_writer import DEFAULT_REVERIFY_DELAY_S, apply_slots, read_slots


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    coordinator = PricingCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()
    coordinator.async_schedule_quarter_ticks()

    from .battery_coordinator import MODE_OFF, BatteryCoordinator

    battery = BatteryCoordinator(hass, entry, coordinator)
    if battery.mode != MODE_OFF:
        await battery.async_refresh()
        battery.async_schedule_ticks()

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
        "pricing": coordinator,
        "battery": battery,
    }
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_options_updated))
    _async_register_services(hass)
    return True


def _async_register_services(hass: HomeAssistant) -> None:
    if hass.services.has_service(DOMAIN, "solis_apply_slots"):
        return

    async def handle_apply_slots(call: ServiceCall) -> ServiceResponse:
        return await apply_slots(
            hass,
            charge_slots=call.data.get("charge_slots"),
            discharge_slots=call.data.get("discharge_slots"),
            dry_run=bool(call.data.get("dry_run", False)),
            allow_cross_side_overlap=bool(
                call.data.get("allow_cross_side_overlap", False)
            ),
            reverify_delay_s=float(
                call.data.get("reverify_delay_seconds", DEFAULT_REVERIFY_DELAY_S)
            ),
        )

    async def handle_read_slots(call: ServiceCall) -> ServiceResponse:
        return await read_slots(hass)

    hass.services.async_register(
        DOMAIN, "solis_apply_slots", handle_apply_slots, supports_response="optional"
    )
    hass.services.async_register(
        DOMAIN, "solis_read_slots", handle_read_slots, supports_response="only"
    )


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    return unload_ok


async def _async_options_updated(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)
