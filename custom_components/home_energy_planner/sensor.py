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
            PlannerSummarySensor(coordinator, entry, hass),
            AllInPriceEurSensor(coordinator, entry),
            SelfSufficiencySensor(coordinator, entry, hass),
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
            "learned": self._learned_attrs(),
            "cooling": data.cooling,
            "legacy_target": data.legacy_target,
            "matches_legacy": (
                data.legacy_target is not None
                and abs(result.target - data.legacy_target) < 0.05
            ),
            "last_apply": data.applied,
            # aligned quarter-hour series for dashboard plotting
            "projection": data.projection,
        }

    def _learned_attrs(self) -> dict[str, Any]:
        preferences = getattr(self.coordinator, "preferences", None)
        if preferences is None:
            return {}
        summary = preferences.summary()
        return {
            "target_offset": summary["adjustments"].get("climate_target_offset", 0.0),
            "event_counts": summary["event_counts"],
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
            "learned": self._learned_attrs(),
            "last_apply": data.applied,
        }

    def _learned_attrs(self) -> dict[str, Any]:
        preferences = getattr(self.coordinator, "preferences", None)
        if preferences is None:
            return {}
        summary = preferences.summary()
        adjustments = summary["adjustments"]
        weekdays = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
        return {
            "target_offset_by_weekday": {
                name: adjustments.get(f"water_weekday_{day}", 0.0)
                for day, name in enumerate(weekdays)
            },
            "event_counts": summary["event_counts"],
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


class AllInPriceEurSensor(CoordinatorEntity[PricingCoordinator], SensorEntity):
    """All-in price in EUR/kWh, for the HA Energy dashboard's price entity.

    The Energy dashboard integrates cost as (energy delta x price) over each
    short recorder interval, so pointing a grid source's `entity_energy_price`
    here gives true spot-priced cost — the near-realtime integration a flat
    tariff cannot. Our other price sensors are c/kWh; HA wants currency/kWh.
    """

    _attr_has_entity_name = True
    _attr_name = "All-in price EUR"
    _attr_native_unit_of_measurement = "EUR/kWh"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:cash-clock"
    _attr_suggested_display_precision = 4

    def __init__(self, coordinator: PricingCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_all_in_price_eur"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": "Home Energy Planner",
        }

    @property
    def native_value(self) -> float | None:
        data = self.coordinator.data
        if data is None or not data.periods:
            return None
        return round(data.periods[0].all_in_cents_per_kwh / 100.0, 4)


class SelfSufficiencySensor(CoordinatorEntity[PricingCoordinator], SensorEntity):
    """Share of the current house load covered by sun + battery (0–100 %).

    Instantaneous, from live power; recorded as a measurement so HA's
    statistics give the month-on-month mean (a spot-price house can't read
    self-sufficiency off a monthly kWh counter). Grid power sign convention:
    positive = export, negative = import (as used across the writers).
    """

    _attr_has_entity_name = True
    _attr_name = "Self-sufficiency"
    _attr_native_unit_of_measurement = "%"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:solar-power-variant"
    _attr_suggested_display_precision = 0

    _CONSUMPTION_ENTITY = "sensor.solis_total_consumption_power"
    _GRID_POWER_ENTITY = "sensor.solis_power_grid_total_power"

    def __init__(
        self, coordinator: PricingCoordinator, entry: ConfigEntry, hass: HomeAssistant
    ) -> None:
        super().__init__(coordinator)
        self._hass = hass
        self._attr_unique_id = f"{entry.entry_id}_self_sufficiency"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": "Home Energy Planner",
        }

    def _watt(self, entity_id: str) -> float | None:
        state = self._hass.states.get(entity_id)
        try:
            return float(state.state)  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            return None

    @property
    def native_value(self) -> float | None:
        from .summary import self_sufficiency_pct

        grid = self._watt(self._GRID_POWER_ENTITY)
        import_w = -grid if grid is not None else None
        return self_sufficiency_pct(self._watt(self._CONSUMPTION_ENTITY), import_w)


class PlannerSummarySensor(CoordinatorEntity[PricingCoordinator], SensorEntity):
    """Human-facing rollup: is it in control, is it cheap now, what's it doing.

    Reads the other coordinators' already-published state (no device writes)
    and reuses the watchdog's pure health logic read-only, so the dashboard
    can lead with what a person cares about instead of optimizer internals.
    """

    _attr_has_entity_name = True
    _attr_name = "Summary"
    _attr_icon = "mdi:home-lightning-bolt"
    _unrecorded_attributes = frozenset(
        {"assets", "issues", "info", "next_cheap_window", "peak_window", "coming_up"}
    )

    def __init__(
        self, coordinator: PricingCoordinator, entry: ConfigEntry, hass: HomeAssistant
    ) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._hass = hass
        self._attr_unique_id = f"{entry.entry_id}_summary"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": "Home Energy Planner",
        }

    def _bundle(self) -> dict[str, Any]:
        return self._hass.data.get(DOMAIN, {}).get(self._entry.entry_id, {})

    def _float_state(self, entity_id: str, attribute: str | None = None) -> float | None:
        state = self._hass.states.get(entity_id)
        if state is None:
            return None
        raw = state.attributes.get(attribute) if attribute else state.state
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _health(self, now, bundle: dict[str, Any]) -> list[str]:
        """Attention-level issues, computed read-only (no watchdog mutation)."""
        from datetime import timedelta

        from .watchdog import (
            APPLY_FAILING_HOURS,
            CRITICAL_INPUTS,
            ENGINE_FALLBACK_HOURS,
            WatchdogSnapshot,
            evaluate_issues,
            input_is_stale,
        )

        pdata = self.coordinator.data
        prices_age = None
        horizon_hours = 0.0
        if pdata is not None and pdata.periods:
            prices_age = max(0.0, (now - pdata.horizon_start).total_seconds() / 3600.0)
            horizon_hours = len(pdata.periods) * 0.25

        watchdog = bundle.get("watchdog")
        started = getattr(watchdog, "_started", None)
        in_grace = started is not None and (now - started) < timedelta(minutes=15)
        stale = (
            []
            if in_grace
            else [e for e in CRITICAL_INPUTS if input_is_stale(self._hass.states.get(e), now)]
        )

        battery = bundle.get("battery")
        bd = getattr(battery, "data", None)
        failing = (
            bd is not None
            and bd.mode == "control"
            and bd.applied is not None
            and not bd.applied.get("success")
        )
        fallback = (
            battery is not None
            and bd is not None
            and bd.engine != "lp"
            and str(battery._option("battery_engine")) == "lp"  # noqa: SLF001
        )
        snapshot = WatchdogSnapshot(
            prices_age_hours=prices_age,
            horizon_hours=horizon_hours,
            stale_inputs=stale,
            battery_apply_failing_hours=APPLY_FAILING_HOURS if failing else 0.0,
            engine_fallback_hours=ENGINE_FALLBACK_HOURS if fallback else 0.0,
        )
        return [msg for _key, msg in evaluate_issues(snapshot)]

    def _compute(self) -> dict[str, Any]:
        from homeassistant.util import dt as dt_util

        from .summary import SummaryInputs, build_summary

        now = dt_util.now()
        bundle = self._bundle()
        pdata = self.coordinator.data
        horizon = (
            [p.all_in_cents_per_kwh for p in pdata.periods] if pdata else []
        )

        battery = bundle.get("battery")
        bd = getattr(battery, "data", None)
        charging = discharging = False
        if bd is not None and bd.plan.periods:
            first = bd.plan.periods[0]
            charging = first.grid_charge_kwh > 0
            discharging = first.discharge_to_load_kwh > 0

        wc = bundle.get("water_heater")
        wdata = getattr(wc, "data", None)
        tank_temp = None
        water_override = None
        if wc is not None:
            tank_temp = self._float_state(
                str(wc._option("water_heater_entity")), "current_temperature"  # noqa: SLF001
            )
            water_override = wc._override.until or wc._power_override.until  # noqa: SLF001

        cc = bundle.get("climate")
        cdata = getattr(cc, "data", None)
        ic = bundle.get("ilp")
        idata = getattr(ic, "data", None)

        info: list[str] = []

        def _note(label: str, until) -> None:
            if until is not None and until > now:
                info.append(f"{label} held manually until {until.strftime('%H:%M')}")

        if wc is not None:
            _note("Hot water", wc._override.until)  # noqa: SLF001
            _note("Hot water on/off", wc._power_override.until)  # noqa: SLF001
        if ic is not None:
            _note("Air-air", ic._override.until)  # noqa: SLF001
        if cc is not None:
            _note("Heat pump", cc._override.until)  # noqa: SLF001

        inputs = SummaryInputs(
            horizon=horizon,
            # window labels are formatted with strftime downstream, so hand
            # over a local-time datetime (horizon_start is UTC-aware)
            horizon_start=dt_util.as_local(pdata.horizon_start) if pdata else None,
            period_minutes=15,
            soc_pct=self._float_state("sensor.solis_remaining_battery_capacity"),
            battery_charging_now=charging,
            battery_discharging_now=discharging,
            water_mode=wdata.effective_mode if wdata else None,
            tank_temp=tank_temp,
            water_override_until=water_override,
            climate_regime=getattr(cc, "regime", None),
            climate_target=cdata.result.target if cdata else None,
            room_temp=cdata.room_temp if cdata else None,
            ilp_action=idata.effective_action if idata else None,
            ilp_reason=idata.result.reason if idata else None,
            issues=self._health(now, bundle),
            info=info,
        )
        return build_summary(inputs)

    @property
    def native_value(self) -> str | None:
        return self._compute()["headline"]

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return self._compute()


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
            "learned": self._learned_attrs(),
            "last_apply": data.applied,
        }

    def _learned_attrs(self) -> dict[str, Any]:
        preferences = getattr(self.coordinator, "preferences", None)
        if preferences is None:
            return {}
        summary = preferences.summary()
        adjustments = summary["adjustments"]
        return {
            "cool_room_above": adjustments.get("ilp_cool_room_above", 0.0),
            "dry_humidity_above": adjustments.get("ilp_dry_humidity_above", 0.0),
            "dry_room_floor": adjustments.get("ilp_dry_room_floor", 0.0),
            "heat_room_below": adjustments.get("ilp_heat_room_below", 0.0),
            "event_counts": summary["event_counts"],
        }
