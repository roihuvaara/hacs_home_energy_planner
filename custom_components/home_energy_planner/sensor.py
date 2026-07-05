"""Price sensors exposing the current value and the full horizon."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .battery_coordinator import BatteryCoordinator
from .climate_coordinator import ClimateCoordinator
from .const import DOMAIN
from .coordinator import PricingCoordinator
from .ilp_coordinator import IlpCoordinator
from .water_heater_coordinator import WaterHeaterCoordinator

PRICE_KIND_RAW = "raw"
PRICE_KIND_VAT = "vat"
PRICE_KIND_ALL_IN = "all_in"

_KIND_NAMES = {
    PRICE_KIND_RAW: "Spot price",
    PRICE_KIND_VAT: "Price with VAT",
    PRICE_KIND_ALL_IN: "All-in price",
}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    data = hass.data[DOMAIN][entry.entry_id]
    coordinator: PricingCoordinator = data["pricing"]
    async_add_entities(
        [
            EnergyPriceSensor(coordinator, entry, PRICE_KIND_RAW),
            EnergyPriceSensor(coordinator, entry, PRICE_KIND_VAT),
            EnergyPriceSensor(coordinator, entry, PRICE_KIND_ALL_IN),
            BatteryPlanSensor(data["battery"], entry),
            ClimateTargetSensor(data["climate"], entry),
            WaterHeaterModeSensor(data["water_heater"], entry),
            IlpRecommendationSensor(data["ilp"], entry),
            ExportCurtailmentSensor(coordinator, entry),
        ]
    )


class EnergyPriceSensor(CoordinatorEntity[PricingCoordinator], SensorEntity):
    _attr_has_entity_name = True
    _attr_native_unit_of_measurement = "c/kWh"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:cash-clock"
    # The horizon series are planner inputs, not history; keep them out of
    # the recorder to avoid state-attribute bloat.
    _unrecorded_attributes = frozenset(
        {"horizon", "horizon_start", "period_minutes", "period_count"}
    )

    def __init__(
        self,
        coordinator: PricingCoordinator,
        entry: ConfigEntry,
        kind: str,
    ) -> None:
        super().__init__(coordinator)
        self._kind = kind
        self._attr_name = _KIND_NAMES[kind]
        self._attr_unique_id = f"{entry.entry_id}_price_{kind}"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": "Home Energy Planner",
        }

    def _series(self) -> list[float]:
        data = self.coordinator.data
        if data is None:
            return []
        if self._kind == PRICE_KIND_RAW:
            return [period.raw_cents_per_kwh for period in data.periods]
        if self._kind == PRICE_KIND_VAT:
            return [period.vat_cents_per_kwh for period in data.periods]
        return [period.all_in_cents_per_kwh for period in data.periods]

    @property
    def native_value(self) -> float | None:
        series = self._series()
        return series[0] if series else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data
        if data is None:
            return {}
        return {
            "horizon_start": data.horizon_start.isoformat(),
            "period_minutes": 15,
            "period_count": len(data.periods),
            "tomorrow_included": data.tomorrow_included,
            "contiguous": data.contiguous,
            "horizon": self._series(),
        }


class BatteryPlanSensor(CoordinatorEntity[BatteryCoordinator], SensorEntity):
    """Planned end-of-horizon SOC; full plan in attributes."""

    _attr_has_entity_name = True
    _attr_name = "Battery plan"
    _attr_native_unit_of_measurement = "%"
    _attr_icon = "mdi:battery-clock"
    _unrecorded_attributes = frozenset(
        {"actions", "charge_slots", "discharge_slots", "series"}
    )

    def __init__(self, coordinator: BatteryCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_battery_plan"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": "Home Energy Planner",
        }

    @property
    def available(self) -> bool:
        return self.coordinator.data is not None

    @property
    def native_value(self) -> float | None:
        data = self.coordinator.data
        return data.plan.end_soc_pct if data else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data
        if data is None:
            return {}
        plan = data.plan
        return {
            "mode": data.mode,
            "engine": data.engine,
            "load_forecast": data.load_info,
            "tank_milp_windows": data.tank_windows,
            "planned_cost_cents": plan.total_cost_cents,
            "baseline_cost_cents": plan.baseline_cost_cents,
            "planned_saving_cents": round(
                plan.baseline_cost_cents - plan.total_cost_cents, 2
            ),
            "period_count": len(plan.periods),
            "charge_slots": [s.as_dict() for s in data.charge_slots if s.enabled],
            "discharge_slots": [s.as_dict() for s in data.discharge_slots if s.enabled],
            "actions": [
                {
                    "start": p.start.isoformat(),
                    "action": p.action,
                    "grid_charge_kwh": p.grid_charge_kwh,
                    "discharge_kwh": p.discharge_to_load_kwh,
                    "price": p.price_cents_per_kwh,
                }
                for p in plan.periods
                if p.action != "hold" or p.grid_charge_kwh > 0
            ][:64],
            "last_apply_success": (data.applied or {}).get("success"),
            # aligned quarter-hour series for dashboard plotting
            "series": {
                "start": plan.periods[0].start.isoformat() if plan.periods else None,
                "period_minutes": 15,
                "price": [p.price_cents_per_kwh for p in plan.periods],
                "soc_pct": [
                    round(
                        self.coordinator.battery_params().soc_from_buffer_kwh(
                            p.buffer_end_kwh
                        ),
                        1,
                    )
                    for p in plan.periods
                ],
                "grid_charge_kwh": [p.grid_charge_kwh for p in plan.periods],
                "discharge_kwh": [p.discharge_to_load_kwh for p in plan.periods],
                "load_kwh": [p.load_kwh for p in data.input_periods],
                "solar_kwh": [p.solar_kwh for p in data.input_periods],
            },
        }


class ClimateTargetSensor(CoordinatorEntity[ClimateCoordinator], SensorEntity):
    """Computed space-heating target; every component in attributes."""

    _attr_has_entity_name = True
    _attr_name = "Heat pump target"
    _attr_native_unit_of_measurement = "°C"
    _attr_icon = "mdi:home-thermometer"

    def __init__(self, coordinator: ClimateCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_climate_target"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": "Home Energy Planner",
        }

    @property
    def available(self) -> bool:
        return self.coordinator.data is not None

    @property
    def native_value(self) -> float | None:
        data = self.coordinator.data
        return data.result.target if data else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data
        if data is None:
            return {}
        result = data.result
        return {
            "mode": data.mode,
            "weather_base": result.weather_base,
            "wind_bump": result.wind_bump,
            "price_offset": result.price.offset,
            "price_position": result.price.position,
            "price_now": result.price.current,
            "price_p10": round(result.price.p10, 3),
            "price_median": round(result.price.median, 3),
            "price_p90": round(result.price.p90, 3),
            "price_spread": round(result.price.spread, 3),
            "protected_price_offset": result.protected_price_offset,
            "cold_dip_boost": result.cold_dip_boost,
            "comfort_correction": result.comfort_correction,
            "lead_boost": result.lead_boost,
            "warm_correction": result.warm_correction,
            "sun_correction": result.sun_correction,
            "room_temp": data.room_temp,
            "lead_hold_until": (
                data.lead_hold_until.isoformat() if data.lead_hold_until else None
            ),
            "manual_override_until": (
                self.coordinator._override.until.isoformat()
                if self.coordinator._override.until
                else None
            ),
            "manual_override_count": self.coordinator._override.count,
            "legacy_target": data.legacy_target,
            "matches_legacy": (
                data.legacy_target is not None
                and abs(result.target - data.legacy_target) < 0.05
            ),
            "last_apply": data.applied,
        }


class WaterHeaterModeSensor(CoordinatorEntity[WaterHeaterCoordinator], SensorEntity):
    """Computed hot-water control mode; tank target in attributes."""

    _attr_has_entity_name = True
    _attr_name = "Water heater mode"
    _attr_icon = "mdi:water-boiler"

    def __init__(self, coordinator: WaterHeaterCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_water_heater_mode"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": "Home Energy Planner",
        }

    @property
    def available(self) -> bool:
        return self.coordinator.data is not None

    @property
    def native_value(self) -> str | None:
        data = self.coordinator.data
        return data.effective_mode if data else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data
        if data is None:
            return {}
        result = data.result
        return {
            "mode": data.mode,
            "computed_mode": result.mode,
            "target_temp": data.effective_target,
            "actual_surplus": result.actual_surplus,
            "buffer_preserve": result.buffer_preserve,
            "cheap_windows": result.cheap_windows,
            "manual_override_until": (
                self.coordinator._override.until.isoformat()
                if self.coordinator._override.until
                else None
            ),
            "manual_override_count": self.coordinator._override.count,
            "price_median": result.price_median,
            "price_delta": result.price_delta,
            "legacy_mode": data.legacy_mode,
            "matches_legacy": (
                data.legacy_mode is not None
                and data.effective_mode == data.legacy_mode
            ),
            "last_apply": data.applied,
        }


class ExportCurtailmentSensor(CoordinatorEntity[PricingCoordinator], SensorEntity):
    """Export-curtailment intent from raw spot prices (observe only).

    Publishes when the planner would disable grid export (raw spot at or
    below zero); the control side lands after the export-switch semantics
    probe (todo 002).
    """

    _attr_has_entity_name = True
    _attr_name = "Export curtailment"
    _attr_icon = "mdi:transmission-tower-off"
    _unrecorded_attributes = frozenset({"windows"})

    def __init__(self, coordinator: PricingCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_export_curtailment"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": "Home Energy Planner",
        }

    @property
    def native_value(self) -> str | None:
        data = self.coordinator.data
        if data is None or not data.periods:
            return None
        return "curtail" if data.periods[0].raw_cents_per_kwh <= 0 else "export"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data
        if data is None:
            return {}
        windows = []
        run_start = None
        previous = None
        for period in data.periods:
            if period.raw_cents_per_kwh <= 0:
                if run_start is None:
                    run_start = period.start
                previous = period.start
            elif run_start is not None:
                windows.append({"start": run_start.isoformat(), "end": period.start.isoformat()})
                run_start = None
        if run_start is not None and previous is not None:
            windows.append({"start": run_start.isoformat(), "end": None})
        return {
            "raw_now": data.periods[0].raw_cents_per_kwh if data.periods else None,
            "windows": windows,
            "curtailed_period_count": sum(
                1 for p in data.periods if p.raw_cents_per_kwh <= 0
            ),
        }


class IlpRecommendationSensor(CoordinatorEntity[IlpCoordinator], SensorEntity):
    """Recommended ILP (air-to-air) action; reasons in attributes."""

    _attr_has_entity_name = True
    _attr_name = "ILP recommendation"
    _attr_icon = "mdi:heat-pump"

    def __init__(self, coordinator: IlpCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_ilp_recommendation"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": "Home Energy Planner",
        }

    @property
    def available(self) -> bool:
        return self.coordinator.data is not None

    @property
    def native_value(self) -> str | None:
        data = self.coordinator.data
        return data.effective_action if data else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data
        if data is None:
            return {}
        result = data.result
        return {
            "mode": data.mode,
            "computed_action": result.action,
            "reason": result.reason,
            "target_temp": result.target_temp,
            "room_temp": data.room_temp,
            "room_humidity": data.room_humidity,
            "actual_surplus": result.actual_surplus,
            "price_delta": result.price_delta,
            "manual_override_until": (
                self.coordinator._override.until.isoformat()
                if self.coordinator._override.until
                else None
            ),
            "manual_override_count": self.coordinator._override.count,
            "last_apply": data.applied,
        }
