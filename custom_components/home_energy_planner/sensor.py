"""Price sensors exposing the current value and the full horizon."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

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
    coordinator: PricingCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        [
            EnergyPriceSensor(coordinator, entry, PRICE_KIND_RAW),
            EnergyPriceSensor(coordinator, entry, PRICE_KIND_VAT),
            EnergyPriceSensor(coordinator, entry, PRICE_KIND_ALL_IN),
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
