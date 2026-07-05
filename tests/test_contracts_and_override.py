from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "custom_components"))

from home_energy_planner.manual_override import ManualOverrideTracker  # noqa: E402
from home_energy_planner.pricing import (  # noqa: E402
    Contract,
    PricingConfig,
    RawSlot,
    build_price_horizon,
)

TZ = ZoneInfo("Europe/Helsinki")
CONFIG = PricingConfig()
HOLD = timedelta(hours=4)


def slots(day, prices_eur_mwh):
    start = datetime(2026, 10, day, 0, 0, tzinfo=TZ)
    return [
        RawSlot(start=start + timedelta(minutes=15 * i), price_eur_per_mwh=p)
        for i, p in enumerate(prices_eur_mwh)
    ]


def test_fixed_contract_flattens_energy_component():
    contract = Contract(date(2026, 9, 15), date(2027, 3, 14), "fixed", 8.5)
    periods = build_price_horizon(
        slots(1, [50.0, 150.0, 250.0, 350.0]),
        now=datetime(2026, 10, 1, 0, 0, tzinfo=TZ),
        config=CONFIG,
        local_tz=TZ,
        contracts=[contract],
    )
    # night transfer 3.12 applies; energy part identical despite spot swings
    assert all(abs(p.all_in_cents_per_kwh - (8.5 + 3.12)) < 1e-6 for p in periods)
    # spot fields keep the real spot for export/curtailment decisions
    assert periods[0].raw_cents_per_kwh == 5.0


def test_flexible_contract_keeps_spot_shape_anchored_to_baseline():
    contract = Contract(date(2026, 9, 15), date(2027, 3, 14), "flexible", 8.5)
    periods = build_price_horizon(
        slots(1, [50.0, 150.0, 250.0, 350.0]),
        now=datetime(2026, 10, 1, 0, 0, tzinfo=TZ),
        config=CONFIG,
        local_tz=TZ,
        contracts=[contract],
    )
    all_in = [p.all_in_cents_per_kwh for p in periods]
    vat = [p.vat_cents_per_kwh for p in periods]
    # deltas between periods equal the spot(vat) deltas exactly
    for i in range(1, 4):
        assert abs((all_in[i] - all_in[0]) - (vat[i] - vat[0])) < 1e-6
    # mean energy component equals the baseline
    mean_energy = sum(all_in) / 4 - 3.12
    assert abs(mean_energy - 8.5) < 1e-6


def test_contract_only_applies_inside_dates():
    contract = Contract(date(2026, 10, 2), date(2026, 10, 3), "fixed", 8.5)
    periods = build_price_horizon(
        slots(1, [100.0] * 4),
        now=datetime(2026, 10, 1, 0, 0, tzinfo=TZ),
        config=CONFIG,
        local_tz=TZ,
        contracts=[contract],
    )
    # Oct 1 is outside the contract: spot + margin applies
    expected = round(10.0 * 1.255 + CONFIG.margin_cents_per_kwh + 3.12, 4)
    assert periods[0].all_in_cents_per_kwh == expected


def now_at(hour):
    return datetime(2026, 7, 6, hour, 0, tzinfo=timezone.utc)


def test_manual_change_suppresses_until_expiry():
    tracker = ManualOverrideTracker()
    tracker.record_write(55)
    # human turned the tank to 60; planner wants 51
    assert tracker.suppressed(60, 51, now_at(10), HOLD)
    assert tracker.count == 1
    assert tracker.suppressed(60, 51, now_at(13), HOLD)  # still inside window
    assert not tracker.suppressed(60, 51, now_at(15), HOLD)  # expired: free game


def test_no_override_when_device_holds_our_value_or_agrees():
    tracker = ManualOverrideTracker()
    tracker.record_write(55)
    assert not tracker.suppressed(55, 60, now_at(10), HOLD)  # normal write path
    assert not tracker.suppressed(51, 51, now_at(10), HOLD)  # already at target
    assert tracker.count == 0


def test_unknown_provenance_gets_one_grace_window():
    tracker = ManualOverrideTracker()
    assert tracker.suppressed("dry", "off", now_at(10), HOLD)
    assert not tracker.suppressed("dry", "off", now_at(15), HOLD)


def test_zero_hold_disables_override():
    tracker = ManualOverrideTracker()
    tracker.record_write("off")
    assert not tracker.suppressed("dry", "cool", now_at(10), timedelta(0))
