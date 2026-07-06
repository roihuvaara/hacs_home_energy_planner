from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "custom_components"))

from home_energy_planner.preference import (  # noqa: E402
    CAPS,
    RING_CAP,
    PreferenceEvent,
    PreferenceLog,
    derive_adjustments,
)

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)


def event(module="ilp", planner="off", manual="cool", when=NOW, **context):
    return PreferenceEvent(
        when=when,
        module=module,
        planner_value=planner,
        manual_value=manual,
        context=context,
    )


# --- event log ----------------------------------------------------------------


def test_log_dedupes_same_manual_value_within_window():
    log = PreferenceLog()
    assert log.append(event())
    # restart re-detection of the same manual value an hour later
    assert not log.append(event(when=NOW + timedelta(hours=1)))
    # same value again outside the window is a fresh signal
    assert log.append(event(when=NOW + timedelta(hours=7)))
    # a different manual value inside the window still counts
    assert log.append(event(manual="dry", when=NOW + timedelta(hours=8)))
    assert len(log.events) == 3


def test_log_dedupe_is_per_module():
    log = PreferenceLog()
    assert log.append(event(module="ilp"))
    assert log.append(
        event(module="climate", planner=23.0, manual=25.0, when=NOW)
    )


def test_log_ring_cap_and_roundtrip():
    log = PreferenceLog()
    for i in range(RING_CAP + 20):
        log.append(
            event(
                manual="cool" if i % 2 else "off",
                when=NOW + timedelta(hours=7 * i),
            )
        )
    assert len(log.events) == RING_CAP
    restored = PreferenceLog.from_dict(log.as_dict())
    assert len(restored.events) == RING_CAP
    assert restored.events[-1].when == log.events[-1].when
    assert restored.events[0].module == "ilp"


def test_from_dict_skips_corrupt_rows():
    data = {"events": [{"bogus": 1}, event().as_dict()]}
    assert len(PreferenceLog.from_dict(data).events) == 1


# --- adjustment derivation ----------------------------------------------------


def test_no_events_means_all_zero():
    adjustments = derive_adjustments([], NOW)
    assert set(adjustments) == set(CAPS)
    assert all(value == 0.0 for value in adjustments.values())


def test_climate_target_steps_are_signed_and_magnitude_blind():
    warmer_small = event(module="climate", planner=23.0, manual=23.5)
    warmer_huge = event(module="climate", planner=23.0, manual=30.0)
    a = derive_adjustments([warmer_small], NOW)
    b = derive_adjustments([warmer_huge], NOW)
    assert a["climate_target_offset"] == b["climate_target_offset"] == 0.1
    cooler = event(module="climate", planner=24.0, manual=22.0)
    assert derive_adjustments([cooler], NOW)["climate_target_offset"] == -0.1


def test_opposite_events_cancel():
    warmer = event(module="climate", planner=23.0, manual=25.0)
    cooler = event(module="climate", planner=24.0, manual=22.0)
    assert derive_adjustments([warmer, cooler], NOW)["climate_target_offset"] == 0.0


def test_adjustments_are_capped():
    warmer = [
        event(
            module="climate",
            planner=23.0,
            manual=25.0,
            when=NOW - timedelta(hours=7 * i),
        )
        for i in range(30)
    ]
    result = derive_adjustments(warmer, NOW)
    assert result["climate_target_offset"] == CAPS["climate_target_offset"]


def test_recency_decay_halves_month_old_events():
    old = event(module="climate", planner=23.0, manual=25.0, when=NOW - timedelta(days=30))
    assert derive_adjustments([old], NOW)["climate_target_offset"] == 0.05


def test_owner_scenario_manual_heat_during_dry():
    # 2026-07-06: ILP on dry, room 22.5, owner forced heat
    fix = event(module="ilp", planner="dry", manual="heat", room_temp=22.5)
    result = derive_adjustments([fix], NOW)
    assert result["ilp_heat_room_below"] == 0.1


def test_ilp_dry_off_in_cool_room_raises_floor_too():
    stop = event(module="ilp", planner="dry", manual="off", room_temp=23.2)
    result = derive_adjustments([stop], NOW)
    assert result["ilp_dry_humidity_above"] == 0.5
    assert result["ilp_dry_room_floor"] == 0.1
    # in a warm room only the humidity threshold moves
    warm_stop = event(module="ilp", planner="dry", manual="off", room_temp=24.5)
    result = derive_adjustments([warm_stop], NOW)
    assert result["ilp_dry_room_floor"] == 0.0


def test_ilp_cool_steps():
    eager = event(module="ilp", planner="off", manual="cool")
    assert derive_adjustments([eager], NOW)["ilp_cool_room_above"] == -0.1
    stop = event(module="ilp", planner="cool", manual="off")
    assert derive_adjustments([stop], NOW)["ilp_cool_room_above"] == 0.1


def test_log_only_modules_do_not_adjust():
    hvac = event(module="climate_hvac", planner="off", manual="heat")
    result = derive_adjustments([hvac], NOW)
    assert all(value == 0.0 for value in result.values())


def test_water_heater_overrides_learn_per_weekday():
    # boosting the tank on one weekday warms only that weekday
    boost = event(module="water_heater", planner=55, manual=66, when=NOW)
    result = derive_adjustments([boost], NOW)
    key = f"water_weekday_{NOW.weekday()}"
    assert result[key] == 1.0
    other_days = [
        result[f"water_weekday_{day}"] for day in range(7) if day != NOW.weekday()
    ]
    assert all(value == 0.0 for value in other_days)
    # lowering learns downward; repeated boosts cap at 6
    lower = event(module="water_heater", planner=60, manual=51, when=NOW)
    assert derive_adjustments([lower], NOW)[key] == -1.0
    many = [
        event(
            module="water_heater",
            planner=55,
            manual=66,
            when=NOW - timedelta(days=7 * i),
        )
        for i in range(12)
    ]
    assert derive_adjustments(many, NOW)[key] <= 6.0
