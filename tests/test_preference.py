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


def test_opposite_events_mostly_cancel_newer_wins_ties():
    # contradiction discounting means the newer event outweighs the
    # older one it contradicts — near-cancellation, leaning recent
    warmer = event(module="climate", planner=23.0, manual=25.0)
    cooler = event(module="climate", planner=24.0, manual=22.0, when=NOW + timedelta(hours=1))
    offset = derive_adjustments([warmer, cooler], NOW + timedelta(hours=1))[
        "climate_target_offset"
    ]
    assert -0.05 <= offset <= 0.0


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


# --- weather-state kernel: seasonal memory --------------------------------------

from home_energy_planner.preference import daylight_hours, similarity  # noqa: E402

# Koti is at ~61N: July ~19 h light, November ~6.5 h
JULY_15C = {"outdoor_temp": 15.0, "outdoor_mean_7d": 18.0, "daylight_hours": 19.0}
NOVEMBER_15C = {"outdoor_temp": 15.0, "outdoor_mean_7d": 4.0, "daylight_hours": 6.5}
WINTER = {"outdoor_temp": -5.0, "outdoor_mean_7d": -6.0, "daylight_hours": 6.0}


def test_daylight_hours_at_finnish_latitude():
    from datetime import date

    lat = 61.5
    assert daylight_hours(lat, date(2026, 6, 21)) > 18.0
    assert daylight_hours(lat, date(2026, 12, 21)) < 6.5
    assert abs(daylight_hours(lat, date(2026, 3, 20)) - 12.0) < 0.8
    # polar clamps
    assert daylight_hours(70.0, date(2026, 6, 21)) == 24.0
    assert daylight_hours(70.0, date(2026, 12, 21)) == 0.0


def test_same_temperature_different_season_is_dissimilar():
    # the owner's case: a 15 C day in July must not borrow November's
    # preferences even though the instantaneous temperature matches
    assert similarity(JULY_15C, dict(JULY_15C)) > 0.99
    assert similarity(NOVEMBER_15C, dict(JULY_15C)) < 0.01
    # events without features report None (caller falls back to age-only)
    assert similarity({}, dict(JULY_15C)) is None


def test_winter_preferences_resurface_next_winter():
    # two winters of "warmer please" events, 1 and 2 years old
    winter_events = [
        event(
            module="climate",
            planner=22.0,
            manual=23.0,
            when=NOW - timedelta(days=365 * years + drift),
            **WINTER,
        )
        for years in (1, 2)
        for drift in (0, 10, 20)
    ]
    in_winter = derive_adjustments(winter_events, NOW, dict(WINTER))
    in_summer = derive_adjustments(winter_events, NOW, dict(JULY_15C))
    # both past winters contribute when it is winter again...
    assert in_winter["climate_target_offset"] >= 0.3
    # ...and none of it leaks into summer
    assert in_summer["climate_target_offset"] == 0.0


def test_multiple_winters_accumulate_with_gentle_age_decay():
    one_year = [
        event(
            module="climate", planner=22.0, manual=23.0,
            when=NOW - timedelta(days=365), **WINTER,
        )
    ]
    two_years = [
        event(
            module="climate", planner=22.0, manual=23.0,
            when=NOW - timedelta(days=730), **WINTER,
        )
    ]
    w1 = derive_adjustments(one_year, NOW, dict(WINTER))["climate_target_offset"]
    w2 = derive_adjustments(two_years, NOW, dict(WINTER))["climate_target_offset"]
    # older winters still count, at roughly the 2-year half-life
    assert w1 > w2 > 0.0
    assert w2 >= 0.05 - 1e-9


def test_pre_capture_events_keep_the_legacy_fast_fade():
    # a featureless event from last summer must not steer this winter
    # (nor much of anything: 30-day half-life, it is long gone)
    legacy = event(
        module="climate",
        planner=23.0,
        manual=25.0,
        when=NOW - timedelta(days=180),
    )
    result = derive_adjustments([legacy], NOW, dict(WINTER))
    assert result["climate_target_offset"] == 0.0


# --- blended supersession decay: contradiction retires, confirmation keeps ------


def test_preference_flip_needs_few_counter_events_not_one_per_stale():
    # 6 winters-old "warmer" votes; the household changes its mind.
    # Pure signed summing would need 7 counter-events to flip; with
    # contradiction discounting 3 recent "cooler" events suffice.
    old_warmer = [
        event(
            module="climate", planner=22.0, manual=23.0,
            when=NOW - timedelta(days=30 + i), **WINTER,
        )
        for i in range(6)
    ]
    new_cooler = [
        event(
            module="climate", planner=23.0, manual=22.0,
            when=NOW - timedelta(days=i), **WINTER,
        )
        for i in range(3)
    ]
    offset = derive_adjustments(old_warmer + new_cooler, NOW, dict(WINTER))[
        "climate_target_offset"
    ]
    assert offset <= 0.0, offset


def test_confirmation_never_discounts():
    # two same-direction winter events a year apart: both count in full
    # (no supersession between agreeing evidence)
    confirmed = [
        event(
            module="climate", planner=22.0, manual=23.0,
            when=NOW - timedelta(days=365), **WINTER,
        ),
        event(
            module="climate", planner=22.0, manual=23.0,
            when=NOW - timedelta(days=1), **WINTER,
        ),
    ]
    offset = derive_adjustments(confirmed, NOW, dict(WINTER))["climate_target_offset"]
    # ~0.1 (fresh) + ~0.07 (one year of calendar backstop): no contradiction loss
    assert offset >= 0.15


def test_summer_contradiction_cannot_retire_winter_evidence():
    winter_warmer = [
        event(
            module="climate", planner=22.0, manual=23.0,
            when=NOW - timedelta(days=200 + i), **WINTER,
        )
        for i in range(4)
    ]
    summer_cooler = [
        event(
            module="climate", planner=24.0, manual=23.0,
            when=NOW - timedelta(days=i), **JULY_15C,
        )
        for i in range(3)
    ]
    with_summer = derive_adjustments(
        winter_warmer + summer_cooler, NOW, dict(WINTER)
    )["climate_target_offset"]
    without_summer = derive_adjustments(winter_warmer, NOW, dict(WINTER))[
        "climate_target_offset"
    ]
    # querying in winter: the dissimilar summer contradictions barely
    # dent the winter evidence (the summer events themselves weigh ~0)
    assert with_summer >= without_summer * 0.9
