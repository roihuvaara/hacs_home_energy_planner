from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "custom_components"))

from home_energy_planner.climate_core import (  # noqa: E402
    ClimateConfig,
    ClimateInputs,
    ForecastHour,
    cold_dip_boost,
    comfort_correction,
    compute_climate_target,
    price_shape,
    sun_correction,
    warm_correction,
    weather_base,
)

CONFIG = ClimateConfig()


def hours(temps, gusts=None, conditions=None, clouds=None):
    gusts = gusts or [0.0] * len(temps)
    conditions = conditions or [""] * len(temps)
    clouds = clouds or [None] * len(temps)
    return [
        ForecastHour(t, g, c, cc)
        for t, g, c, cc in zip(temps, gusts, conditions, clouds)
    ]


def make_inputs(**overrides):
    defaults = dict(
        hourly_forecast=hours([5.0] * 24),
        fallback_temp=5.0,
        room_temp=23.5,
        future_all_in=[12.0] * 96,
        solar_current_hour_kwh=0.0,
        solar_next_hour_kwh=0.0,
        lead_hold_active=False,
    )
    defaults.update(overrides)
    return ClimateInputs(**defaults)


# --- weather base ------------------------------------------------------------


def test_weather_base_bands():
    cases = [(15.0, 20.0), (11.0, 22.0), (7.0, 23.0), (3.0, 25.0), (-1.0, 26.0), (-5.0, 28.0), (-10.0, 30.0)]
    for avg_temp, expected in cases:
        base, _ = weather_base(hours([avg_temp] * 24), 0.0, CONFIG)
        assert base == expected, f"avg {avg_temp} -> {base}, expected {expected}"


def test_weather_base_uses_first_12_hours_only():
    base, _ = weather_base(hours([15.0] * 12 + [-10.0] * 12), 0.0, CONFIG)
    assert base == 20.0


def test_wind_bump_tiers():
    base_calm, bump_calm = weather_base(hours([5.0] * 24), 0.0, CONFIG)
    assert bump_calm == 0.0
    _, bump_windy = weather_base(
        hours([5.0] * 24, gusts=[40.0] * 6 + [0.0] * 18), 0.0, CONFIG
    )
    assert bump_windy == 1.0
    _, bump_very = weather_base(
        hours([5.0] * 24, gusts=[55.0] * 6 + [0.0] * 18), 0.0, CONFIG
    )
    assert bump_very == 2.0
    base_extreme, bump_extreme = weather_base(
        hours([5.0] * 24, gusts=[65.0] * 4 + [0.0] * 20), 0.0, CONFIG
    )
    assert bump_extreme == 3.0
    assert base_extreme == base_calm + 3.0


def test_weather_base_coldest_band_plus_bump():
    base, _ = weather_base(hours([-10.0] * 24, gusts=[65.0] * 24), 0.0, CONFIG)
    assert base == 33.0  # coldest band 30 + extreme wind bump 3, under the 35 cap


# --- horizon price shaping ---------------------------------------------------


def test_flat_horizon_gets_no_offset():
    # uniformly cheap or expensive days have nothing worth shifting
    assert price_shape([2.0] * 48, CONFIG).offset == 0.0
    assert price_shape([40.0] * 48, CONFIG).offset == 0.0


def test_cheap_quarter_before_spike_banks_heat():
    future = [8.0] * 12 + [45.0] * 12 + [15.0] * 24
    shape = price_shape(future, CONFIG)
    assert shape.offset >= 2.0
    assert shape.position < 0


def test_spike_quarter_coasts():
    future = [45.0] * 12 + [8.0] * 12 + [15.0] * 24
    shape = price_shape(future, CONFIG)
    assert shape.offset == -4.0


def test_median_price_neutral():
    future = [15.0] + [8.0] * 20 + [15.0] * 6 + [22.0] * 21
    assert price_shape(future, CONFIG).offset == 0.0


def test_negative_prices_max_boost():
    future = [-1.0] * 4 + [10.0] * 20 + [20.0] * 24
    assert price_shape(future, CONFIG).offset == 4.0


def test_short_horizon_no_offset():
    assert price_shape([5.0, 40.0], CONFIG).offset == 0.0
    assert price_shape([], CONFIG).offset == 0.0


# --- corrections -------------------------------------------------------------


def test_comfort_correction_bands():
    assert comfort_correction(22.0, CONFIG) == 2.0
    assert comfort_correction(23.5, CONFIG) == 1.0
    assert comfort_correction(24.5, CONFIG) == 0.0
    assert comfort_correction(None, CONFIG) == 0.0


def test_warm_correction_bands():
    assert warm_correction(25.5, CONFIG) == 2.0
    assert warm_correction(24.6, CONFIG) == 1.0
    assert warm_correction(24.0, CONFIG) == 0.0
    assert warm_correction(None, CONFIG) == 0.0


def test_sun_correction_needs_bright_forecast():
    bright = hours([10.0, 10.0], conditions=["sunny", "sunny"], clouds=[20.0, 20.0])
    overcast = hours([10.0, 10.0], conditions=["cloudy", "cloudy"])
    assert sun_correction(24.0, bright, 2.0, 0.0, CONFIG) == 2.0
    assert sun_correction(24.0, overcast, 2.0, 0.0, CONFIG) == 0.0
    # cloud coverage gate applies even on a nominally sunny condition
    hazy = hours([10.0, 10.0], conditions=["sunny", "sunny"], clouds=[80.0, 80.0])
    assert sun_correction(24.0, hazy, 2.0, 0.0, CONFIG) == 0.0


def test_sun_correction_tier_one():
    bright = hours([10.0, 10.0], conditions=["partlycloudy", "sunny"], clouds=[30.0, None])
    assert sun_correction(23.0, bright, 0.5, 0.0, CONFIG) == 1.0
    assert sun_correction(22.5, bright, 0.5, 0.0, CONFIG) == 0.0
    assert sun_correction(23.0, bright, 0.1, 0.1, CONFIG) == 0.0


def test_cold_dip_boost_next_6_hours():
    assert cold_dip_boost(hours([-5.0] + [5.0] * 23), 5.0, CONFIG) == 2.0
    assert cold_dip_boost(hours([-2.0] + [5.0] * 23), 5.0, CONFIG) == 1.0
    assert cold_dip_boost(hours([5.0] * 6 + [-10.0] * 18), 5.0, CONFIG) == 0.0


# --- combination -------------------------------------------------------------


def test_target_combination_and_clamp():
    # room 24.2 sits in the neutral band: no comfort, warm, or sun correction
    result = compute_climate_target(make_inputs(room_temp=24.2))
    # avg 5 C -> base 25, everything else neutral
    assert result.weather_base == 25.0
    assert result.target == 25.0

    cold = compute_climate_target(
        make_inputs(hourly_forecast=hours([-10.0] * 24, gusts=[65.0] * 24), room_temp=22.0)
    )
    # base clamped 35 + comfort 2 + cold dip 2 -> clamp back to 35
    assert cold.target == 35.0


def test_negative_offset_suppressed_when_room_cold():
    future = [45.0] * 12 + [8.0] * 12 + [15.0] * 72
    warm_room = compute_climate_target(make_inputs(future_all_in=future, room_temp=23.0))
    assert warm_room.price.offset == -4.0
    assert warm_room.protected_price_offset == -4.0
    cold_room = compute_climate_target(make_inputs(future_all_in=future, room_temp=22.5))
    assert cold_room.protected_price_offset == 0.0
    # both rooms sit in the same comfort tier; the only delta is the
    # suppressed setback offset
    assert cold_room.target == warm_room.target + 4.0


def test_lead_boost_requires_hold_and_cold_room():
    held = compute_climate_target(make_inputs(room_temp=22.9, lead_hold_active=True))
    idle = compute_climate_target(make_inputs(room_temp=22.9, lead_hold_active=False))
    assert held.lead_boost == 1.0
    assert idle.lead_boost == 0.0
    warm_held = compute_climate_target(make_inputs(room_temp=23.5, lead_hold_active=True))
    assert warm_held.lead_boost == 0.0
