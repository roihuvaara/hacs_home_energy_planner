from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "custom_components" / "home_energy_planner"),
)

from pricing import (  # noqa: E402
    PricePeriod,
    PricingConfig,
    RawSlot,
    build_price_horizon,
    floor_to_period,
    horizon_is_contiguous,
    raw_to_cents_per_kwh,
    transfer_cents_for_hour,
)

HELSINKI = ZoneInfo("Europe/Helsinki")
CONFIG = PricingConfig()


def local(year, month, day, hour, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=HELSINKI)


def make_slots(start, count, price_eur_per_mwh=100.0):
    return [
        RawSlot(start=start + timedelta(minutes=15 * i), price_eur_per_mwh=price_eur_per_mwh)
        for i in range(count)
    ]


def test_floor_to_period():
    assert floor_to_period(local(2026, 7, 5, 12, 14)) == local(2026, 7, 5, 12, 0)
    assert floor_to_period(local(2026, 7, 5, 12, 15)) == local(2026, 7, 5, 12, 15)
    assert floor_to_period(local(2026, 7, 5, 12, 59)) == local(2026, 7, 5, 12, 45)


def test_raw_conversion_eur_mwh_to_cents_kwh():
    assert raw_to_cents_per_kwh(100.0) == 10.0
    assert raw_to_cents_per_kwh(-5.0) == -0.5


def test_transfer_day_night_window():
    assert transfer_cents_for_hour(CONFIG, 6) == CONFIG.night_transfer_cents_per_kwh
    assert transfer_cents_for_hour(CONFIG, 7) == CONFIG.day_transfer_cents_per_kwh
    assert transfer_cents_for_hour(CONFIG, 21) == CONFIG.day_transfer_cents_per_kwh
    assert transfer_cents_for_hour(CONFIG, 22) == CONFIG.night_transfer_cents_per_kwh


def test_all_in_matches_legacy_automation_formula():
    """Values must reproduce proxy_energy_consumer_price_helpers exactly.

    Legacy: all_in = raw_cents * 1.255 + 2.82752 + (5.11 day / 3.12 night).
    """
    day_slot = RawSlot(start=local(2026, 7, 5, 12, 0), price_eur_per_mwh=100.0)
    night_slot = RawSlot(start=local(2026, 7, 5, 23, 0), price_eur_per_mwh=100.0)
    periods = build_price_horizon(
        [day_slot, night_slot],
        now=local(2026, 7, 5, 12, 0),
        config=CONFIG,
        local_tz=HELSINKI,
    )
    assert periods[0].raw_cents_per_kwh == 10.0
    assert periods[0].vat_cents_per_kwh == 12.55
    assert periods[0].all_in_cents_per_kwh == round(12.55 + 2.82752 + 5.11, 4)
    assert periods[1].all_in_cents_per_kwh == round(12.55 + 2.82752 + 3.12, 4)


def test_horizon_starts_at_current_period_and_drops_past():
    slots = make_slots(local(2026, 7, 5, 11, 0), 8)
    periods = build_price_horizon(
        slots,
        now=local(2026, 7, 5, 12, 7),
        config=CONFIG,
        local_tz=HELSINKI,
    )
    assert periods[0].start == local(2026, 7, 5, 12, 0)
    assert len(periods) == 4


def test_duplicate_starts_deduplicated_first_wins():
    slot = local(2026, 7, 5, 12, 0)
    periods = build_price_horizon(
        [
            RawSlot(start=slot, price_eur_per_mwh=100.0),
            RawSlot(start=slot, price_eur_per_mwh=200.0),
        ],
        now=local(2026, 7, 5, 12, 0),
        config=CONFIG,
        local_tz=HELSINKI,
    )
    assert len(periods) == 1
    assert periods[0].raw_cents_per_kwh == 10.0


def test_two_day_horizon_is_contiguous_and_ordered():
    today = make_slots(local(2026, 7, 5, 0, 0), 96)
    tomorrow = make_slots(local(2026, 7, 6, 0, 0), 96)
    periods = build_price_horizon(
        tomorrow + today,  # out-of-order input on purpose
        now=local(2026, 7, 5, 14, 3),
        config=CONFIG,
        local_tz=HELSINKI,
    )
    assert periods[0].start == local(2026, 7, 5, 14, 0)
    assert periods[-1].start == local(2026, 7, 6, 23, 45)
    assert horizon_is_contiguous(periods)


def test_gap_detected_as_non_contiguous():
    slots = make_slots(local(2026, 7, 5, 12, 0), 2) + make_slots(
        local(2026, 7, 5, 13, 0), 2
    )
    periods = build_price_horizon(
        slots,
        now=local(2026, 7, 5, 12, 0),
        config=CONFIG,
        local_tz=HELSINKI,
    )
    assert not horizon_is_contiguous(periods)


def test_transfer_window_uses_local_time_for_utc_slots():
    # 05:00 UTC = 08:00 Helsinki in summer -> day transfer applies.
    slot = RawSlot(
        start=datetime(2026, 7, 5, 5, 0, tzinfo=timezone.utc),
        price_eur_per_mwh=100.0,
    )
    periods = build_price_horizon(
        [slot],
        now=datetime(2026, 7, 5, 5, 0, tzinfo=timezone.utc),
        config=CONFIG,
        local_tz=HELSINKI,
    )
    assert periods[0].all_in_cents_per_kwh == round(12.55 + 2.82752 + 5.11, 4)


def test_negative_prices_supported():
    slot = RawSlot(start=local(2026, 7, 5, 12, 0), price_eur_per_mwh=-20.0)
    periods = build_price_horizon(
        [slot],
        now=local(2026, 7, 5, 12, 0),
        config=CONFIG,
        local_tz=HELSINKI,
    )
    assert periods[0].raw_cents_per_kwh == -2.0
    assert periods[0].vat_cents_per_kwh == round(-2.0 * 1.255, 4)


def test_empty_input_yields_empty_horizon():
    assert (
        build_price_horizon(
            [], now=local(2026, 7, 5, 12, 0), config=CONFIG, local_tz=HELSINKI
        )
        == []
    )


# --- export contracts ---------------------------------------------------------

from datetime import date  # noqa: E402

from pricing import ExportContract, export_cents_for, export_contract_for  # noqa: E402


def test_export_defaults_to_plain_spot():
    slots = make_slots(local(2026, 7, 5, 12, 0), 4, price_eur_per_mwh=80.0)
    periods = build_price_horizon(
        slots, now=local(2026, 7, 5, 12, 0), config=CONFIG, local_tz=HELSINKI
    )
    assert all(p.export_cents_per_kwh == 8.0 for p in periods)


def test_export_contract_selected_by_date():
    contracts = [
        ExportContract(
            start=date(2026, 10, 1),
            end=date(2027, 3, 31),
            type="spot_minus_margin",
            margin_cents=0.3,
        )
    ]
    assert export_contract_for(contracts, date(2026, 7, 5)) is None
    winter = export_contract_for(contracts, date(2026, 12, 1))
    assert winter is not None and winter.margin_cents == 0.3


def test_export_contract_types():
    spot = ExportContract(date(2026, 1, 1), date(2026, 12, 31), "spot")
    margin = ExportContract(
        date(2026, 1, 1), date(2026, 12, 31), "spot_minus_margin", margin_cents=0.5
    )
    fixed = ExportContract(
        date(2026, 1, 1), date(2026, 12, 31), "fixed", price_cents=4.0
    )
    assert export_cents_for(None, 8.0) == 8.0
    assert export_cents_for(spot, 8.0) == 8.0
    assert export_cents_for(margin, 8.0) == 7.5
    assert export_cents_for(fixed, 8.0) == 4.0
    # negative spot flows through unclamped
    assert export_cents_for(margin, -1.0) == -1.5


def test_export_contract_applied_in_horizon():
    contracts = [
        ExportContract(
            start=date(2026, 7, 1),
            end=date(2026, 7, 31),
            type="spot_minus_margin",
            margin_cents=0.3,
        )
    ]
    slots = make_slots(local(2026, 7, 5, 12, 0), 4, price_eur_per_mwh=80.0)
    periods = build_price_horizon(
        slots,
        now=local(2026, 7, 5, 12, 0),
        config=CONFIG,
        local_tz=HELSINKI,
        export_contracts=contracts,
    )
    assert all(p.export_cents_per_kwh == 7.7 for p in periods)
    # import side untouched by the export contract
    assert all(p.raw_cents_per_kwh == 8.0 for p in periods)
