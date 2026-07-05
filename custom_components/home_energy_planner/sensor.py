"""Price sensors exposing the current value and the full horizon."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .battery_coordinator import BatteryCoordinator
from .const import DOMAIN
from .coordinator import PricingCoordinator

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
    _unrecorded_attributes = frozenset({"actions", "charge_slots", "discharge_slots"})

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
        }
