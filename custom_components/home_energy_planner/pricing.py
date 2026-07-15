"""Pure pricing math for the Home Energy Planner.

Builds the quarter-hour all-in price horizon from Nord Pool spot slots.
No Home Assistant imports; unit-testable standalone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo


PERIOD_MINUTES = 15


@dataclass(frozen=True)
class PricingConfig:
    """Fees and taxes applied on top of the raw spot price.

    All monetary values are cents/kWh. ``vat_rate`` is a fraction (0.255).
    Transfer fee follows the local-time day window
    [day_start_hour, day_end_hour).
    """

    vat_rate: float = 0.255
    margin_cents_per_kwh: float = 2.82752
    day_transfer_cents_per_kwh: float = 5.11
    night_transfer_cents_per_kwh: float = 3.12
    day_start_hour: int = 7
    day_end_hour: int = 22


@dataclass(frozen=True)
class Contract:
    """A date-ranged energy contract overriding the default spot pricing.

    Types (Finnish consumer market, researched 2026-07-06):
    - "fixed": energy price is flat for the term; marginal all-in price
      per period = energy_cents_vat_incl + transfer. Price-shifting
      stops paying by construction; solar self-use still does.
    - "flexible" (joustosähkö): fixed baseline ± monthly consumption
      effect = (consumption-weighted avg spot) - (calendar-month mean
      spot). The marginal price of period t is therefore
      baseline + (spot_vat_t - monthly_mean_vat) + transfer — the spot
      *shape* still drives shifting, anchored to the baseline. The
      monthly mean is approximated by the mean over the available
      horizon (settlement uses the calendar month).

    Dates are inclusive, evaluated in local time.
    """

    start: date
    end: date
    type: str  # "fixed" | "flexible"
    energy_cents_vat_incl: float


def contract_for(contracts: list[Contract], day: date) -> Contract | None:
    for contract in contracts:
        if contract.start <= day <= contract.end:
            return contract
    return None


@dataclass(frozen=True)
class ExportContract:
    """A date-ranged export compensation contract.

    Types:
    - "spot": raw Nord Pool spot, no deduction (the owner's current deal)
    - "spot_minus_margin": raw spot minus ``margin_cents``
    - "fixed": flat ``price_cents`` regardless of spot

    No VAT or transfer applies to export. Values may go negative on
    negative spot — never clamp; paying to export is what makes battery
    absorption free money on those quarters. Dates inclusive, local time.
    """

    start: date
    end: date
    type: str  # "spot" | "spot_minus_margin" | "fixed"
    margin_cents: float = 0.0
    price_cents: float = 0.0


def export_contract_for(
    contracts: list[ExportContract], day: date
) -> ExportContract | None:
    for contract in contracts:
        if contract.start <= day <= contract.end:
            return contract
    return None


def export_cents_for(
    contract: ExportContract | None, raw_cents: float
) -> float:
    """Export compensation for one period; plain spot when uncontracted."""
    if contract is None or contract.type == "spot":
        return raw_cents
    if contract.type == "fixed":
        return contract.price_cents
    return round(raw_cents - contract.margin_cents, 4)


@dataclass(frozen=True)
class RawSlot:
    """One Nord Pool slot as returned by the provider (EUR/MWh)."""

    start: datetime
    price_eur_per_mwh: float


@dataclass(frozen=True)
class PricePeriod:
    start: datetime
    raw_cents_per_kwh: float
    vat_cents_per_kwh: float
    all_in_cents_per_kwh: float
    # export compensation (default: plain spot); no VAT/transfer on export
    export_cents_per_kwh: float = 0.0


def floor_to_period(value: datetime) -> datetime:
    minute = (value.minute // PERIOD_MINUTES) * PERIOD_MINUTES
    return value.replace(minute=minute, second=0, microsecond=0)


def transfer_cents_for_hour(config: PricingConfig, local_hour: int) -> float:
    if config.day_start_hour <= local_hour < config.day_end_hour:
        return config.day_transfer_cents_per_kwh
    return config.night_transfer_cents_per_kwh


def raw_to_cents_per_kwh(price_eur_per_mwh: float) -> float:
    return price_eur_per_mwh / 10.0


def build_price_horizon(
    raw_slots: list[RawSlot],
    *,
    now: datetime,
    config: PricingConfig,
    local_tz: tzinfo,
    contracts: list[Contract] | None = None,
    export_contracts: list[ExportContract] | None = None,
) -> list[PricePeriod]:
    """Compute the forward horizon from the current quarter-hour onward.

    Slots are deduplicated by start (first occurrence wins) and sorted.
    The transfer fee window is evaluated in ``local_tz``. When a contract
    covers a slot's local date, the all-in energy component follows the
    contract instead of spot+margin; raw/vat fields always carry the spot
    values (export revenue and curtailment stay spot-based regardless of
    the purchase contract).
    """

    cutoff = floor_to_period(now)
    seen: set[datetime] = set()
    prepared: list[
        tuple[datetime, float, float, int, Contract | None, ExportContract | None]
    ] = []
    vat_sum = 0.0
    for slot in sorted(raw_slots, key=lambda item: item.start):
        if slot.start < cutoff or slot.start in seen:
            continue
        seen.add(slot.start)
        raw_cents = round(raw_to_cents_per_kwh(slot.price_eur_per_mwh), 4)
        vat_cents = round(raw_cents * (1 + config.vat_rate), 4)
        local = slot.start.astimezone(local_tz)
        prepared.append(
            (
                slot.start,
                raw_cents,
                vat_cents,
                local.hour,
                contract_for(contracts, local.date()) if contracts else None,
                export_contract_for(export_contracts, local.date())
                if export_contracts
                else None,
            )
        )
        vat_sum += vat_cents
    mean_vat = vat_sum / len(prepared) if prepared else 0.0

    periods: list[PricePeriod] = []
    for start, raw_cents, vat_cents, local_hour, contract, export in prepared:
        transfer = transfer_cents_for_hour(config, local_hour)
        if contract is None:
            energy = vat_cents + config.margin_cents_per_kwh
        elif contract.type == "fixed":
            energy = contract.energy_cents_vat_incl
        else:  # flexible: baseline anchored, spot shape preserved
            energy = contract.energy_cents_vat_incl + (vat_cents - mean_vat)
        periods.append(
            PricePeriod(
                start=start,
                raw_cents_per_kwh=raw_cents,
                vat_cents_per_kwh=vat_cents,
                all_in_cents_per_kwh=round(energy + transfer, 4),
                export_cents_per_kwh=export_cents_for(export, raw_cents),
            )
        )
    return periods


def horizon_is_contiguous(periods: list[PricePeriod]) -> bool:
    """True when every period starts one quarter after the previous one."""

    step = timedelta(minutes=PERIOD_MINUTES)
    return all(
        later.start - earlier.start == step
        for earlier, later in zip(periods, periods[1:])
    )
