from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "custom_components"))

from types import SimpleNamespace  # noqa: E402

from home_energy_planner.watchdog import (  # noqa: E402
    Debouncer,
    WatchdogSnapshot,
    evaluate_issues,
    input_is_stale,
)


def snap(**overrides):
    defaults = dict(
        prices_age_hours=0.1,
        horizon_hours=24.0,
        stale_inputs=[],
        battery_apply_failing_hours=0.0,
        engine_fallback_hours=0.0,
    )
    defaults.update(overrides)
    return WatchdogSnapshot(**defaults)


def keys(snapshot):
    return [key for key, _ in evaluate_issues(snapshot)]


def test_healthy_snapshot_has_no_issues():
    assert keys(snap()) == []


def test_price_issues():
    assert keys(snap(prices_age_hours=None)) == ["prices_missing"]
    assert keys(snap(prices_age_hours=3.0)) == ["prices_stale"]
    assert keys(snap(horizon_hours=4.0)) == ["horizon_short"]


def test_stale_inputs_reported_individually():
    result = keys(snap(stale_inputs=["sensor.a", "sensor.b"]))
    assert result == ["input:sensor.a", "input:sensor.b"]


def test_apply_and_engine_thresholds():
    assert keys(snap(battery_apply_failing_hours=1.0)) == []
    assert keys(snap(battery_apply_failing_hours=2.5)) == ["battery_apply"]
    assert keys(snap(engine_fallback_hours=0.5)) == []
    assert keys(snap(engine_fallback_hours=1.5)) == ["engine_fallback"]


def test_input_staleness_uses_last_reported():
    now = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
    old = now - timedelta(hours=5)
    # flat-but-alive: value unchanged for hours (last_updated stuck) while
    # the integration keeps re-reporting it -> NOT stale
    flat_alive = SimpleNamespace(
        state="13.0", last_updated=old, last_reported=now - timedelta(minutes=5)
    )
    assert not input_is_stale(flat_alive, now)
    dead = SimpleNamespace(state="18.0", last_updated=old, last_reported=old)
    assert input_is_stale(dead, now)
    assert input_is_stale(None, now)
    assert input_is_stale(SimpleNamespace(state="unavailable"), now)


def test_debouncer_once_per_day():
    debounce = Debouncer()
    t0 = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
    assert debounce.due("x", t0)
    assert not debounce.due("x", t0 + timedelta(hours=6))
    assert debounce.due("x", t0 + timedelta(hours=25))
    assert debounce.due("y", t0 + timedelta(hours=6))  # independent keys
