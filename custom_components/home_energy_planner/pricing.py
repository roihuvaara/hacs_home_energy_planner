"""Pure pricing math for the Home Energy Planner.

Builds the quarter-hour all-in price horizon from Nord Pool spot slots.
No Home Assistant imports; unit-testable standalone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo


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
) -> list[PricePeriod]:
    """Compute the forward horizon from the current quarter-hour onward.

    Slots are deduplicated by start (first occurrence wins) and sorted.
    The transfer fee window is evaluated in ``local_tz``.
    """

    cutoff = floor_to_period(now)
    seen: set[datetime] = set()
    periods: list[PricePeriod] = []
    for slot in sorted(raw_slots, key=lambda item: item.start):
        if slot.start < cutoff or slot.start in seen:
            continue
        seen.add(slot.start)
        raw_cents = round(raw_to_cents_per_kwh(slot.price_eur_per_mwh), 4)
        vat_cents = round(raw_cents * (1 + config.vat_rate), 4)
        local_hour = slot.start.astimezone(local_tz).hour
        all_in = round(
            vat_cents
            + config.margin_cents_per_kwh
            + transfer_cents_for_hour(config, local_hour),
            4,
        )
        periods.append(
            PricePeriod(
                start=slot.start,
                raw_cents_per_kwh=raw_cents,
                vat_cents_per_kwh=vat_cents,
                all_in_cents_per_kwh=all_in,
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
