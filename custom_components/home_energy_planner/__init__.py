"""Home Energy Planner integration setup."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse
from homeassistant.exceptions import HomeAssistantError

from .const import DOMAIN, PLATFORMS
from .coordinator import PricingCoordinator
from .solis_writer import DEFAULT_REVERIFY_DELAY_S, apply_slots, read_slots


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    coordinator = PricingCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()
    coordinator.async_schedule_quarter_ticks()

    from .battery_coordinator import MODE_OFF, BatteryCoordinator
    from .climate_coordinator import ClimateCoordinator
    from .ilp_coordinator import IlpCoordinator
    from .water_heater_coordinator import WaterHeaterCoordinator

    battery = BatteryCoordinator(hass, entry, coordinator)
    if battery.mode != MODE_OFF:
        await battery.async_refresh()
        battery.async_schedule_ticks()

    climate = ClimateCoordinator(hass, entry, coordinator)
    if climate.mode != MODE_OFF:
        await climate.async_refresh()
        climate.async_schedule_ticks()

    water_heater = WaterHeaterCoordinator(hass, entry, coordinator)
    if water_heater.mode != MODE_OFF:
        await water_heater.async_refresh()
        water_heater.async_schedule_ticks()

    ilp = IlpCoordinator(hass, entry, coordinator)
    if ilp.mode != MODE_OFF:
        await ilp.async_refresh()
        ilp.async_schedule_ticks()

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
        "pricing": coordinator,
        "battery": battery,
        "climate": climate,
        "water_heater": water_heater,
        "ilp": ilp,
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

    def _coordinators() -> tuple[PricingCoordinator, "BatteryCoordinator"]:
        for bundle in hass.data.get(DOMAIN, {}).values():
            if isinstance(bundle, dict) and "pricing" in bundle:
                return bundle["pricing"], bundle["battery"]
        raise HomeAssistantError("home_energy_planner is not set up")

    async def handle_simulate_plan(call: ServiceCall) -> ServiceResponse:
        from .simulation import async_simulate_plan

        pricing, battery = _coordinators()
        return await async_simulate_plan(pricing, battery, dict(call.data))

    async def handle_backtest(call: ServiceCall) -> ServiceResponse:
        from .backtest import async_backtest

        pricing, battery = _coordinators()
        return await async_backtest(pricing, battery, dict(call.data))

    async def handle_set_mode(call: ServiceCall) -> None:
        module = str(call.data["module"])
        mode = str(call.data["mode"])
        if module not in ("battery", "climate", "water_heater", "ilp"):
            raise HomeAssistantError(f"Unknown module '{module}'")
        if mode not in ("off", "observe", "control"):
            raise HomeAssistantError(f"Unknown mode '{mode}'")
        for entry_id in list(hass.data.get(DOMAIN, {})):
            entry = hass.config_entries.async_get_entry(entry_id)
            if entry is None:
                continue
            hass.config_entries.async_update_entry(
                entry, options={**entry.options, f"{module}_mode": mode}
            )

    hass.services.async_register(
        DOMAIN, "solis_apply_slots", handle_apply_slots, supports_response="optional"
    )
    hass.services.async_register(
        DOMAIN, "solis_read_slots", handle_read_slots, supports_response="only"
    )
    hass.services.async_register(
        DOMAIN, "simulate_plan", handle_simulate_plan, supports_response="only"
    )
    hass.services.async_register(
        DOMAIN, "backtest", handle_backtest, supports_response="only"
    )
    async def handle_cool_trial(call: ServiceCall) -> ServiceResponse:
        for bundle in hass.data.get(DOMAIN, {}).values():
            if isinstance(bundle, dict) and "climate" in bundle:
                return await bundle["climate"].async_start_cool_trial(
                    int(call.data.get("minutes", 60)),
                    call.data.get("water_temp"),
                )
        raise HomeAssistantError("home_energy_planner is not set up")

    hass.services.async_register(DOMAIN, "set_mode", handle_set_mode)
    hass.services.async_register(
        DOMAIN, "versati_cool_trial", handle_cool_trial, supports_response="optional"
    )


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    return unload_ok


async def _async_options_updated(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)
