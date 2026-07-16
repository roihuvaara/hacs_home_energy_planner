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


def test_dew_point_sanity():
    from home_energy_planner.climate_core import dew_point_c

    assert abs(dew_point_c(23.0, 50.0) - 12.0) < 0.5
    assert abs(dew_point_c(25.0, 60.0) - 16.7) < 0.6
    assert dew_point_c(23.0, 100.0) == 23.0 or abs(dew_point_c(23.0, 100.0) - 23.0) < 0.01


def test_cool_water_target_guarded():
    from home_energy_planner.climate_core import cool_water_target

    # normal day, dry room: conservative 18
    target, dew = cool_water_target(23.0, 50.0, 22.0, CONFIG)
    assert target == 18.0 and dew is not None
    # hot day, dry room: allowed down to 16
    target, _ = cool_water_target(23.0, 50.0, 28.0, CONFIG)
    assert target == 16.0
    # hot day, humid room: dew point floor wins over the hot-day target
    target, dew = cool_water_target(25.0, 70.0, 28.0, CONFIG)
    assert dew is not None and target == round(dew + CONFIG.dew_point_margin, 1)
    assert target > 16.0
    # unknown humidity: fall back to the configured target only
    target, dew = cool_water_target(23.0, None, 28.0, CONFIG)
    assert target == 16.0 and dew is None


# --- thermal regime state machine --------------------------------------------


def test_regime_signature_has_no_instantaneous_room_temp():
    import inspect

    from home_energy_planner.climate_core import decide_regime

    params = list(inspect.signature(decide_regime).parameters)
    assert "room_temp" not in params
    assert "room_mean_24h" in params  # only the slow signal
    assert "hours_in_regime" not in params  # transitions are free, no dwell


def test_regime_heat_is_forecast_predictive():
    from home_energy_planner.climate_core import decide_regime

    # 12h forecast avg below the mildest base band -> heat, room mean fine
    regime, _ = decide_regime("neutral", 10.0, 12.0, 15.0, 23.8, CONFIG)
    assert regime == "heat"
    # mild forecast, warm room mean -> no heat
    regime, _ = decide_regime("neutral", 16.0, 15.0, 20.0, 23.8, CONFIG)
    assert regime == "neutral"


def test_regime_room_mean_backup_forces_heat():
    from home_energy_planner.climate_core import decide_regime

    regime, _ = decide_regime("neutral", 16.0, 18.0, 26.0, 22.9, CONFIG)
    assert regime == "heat"


def test_regime_cool_needs_warm_stretch_and_hot_day():
    from home_energy_planner.climate_core import decide_regime

    regime, _ = decide_regime("neutral", 18.0, 18.0, 26.0, 23.8, CONFIG)
    assert regime == "cool"
    # warm stretch but no hot day
    regime, _ = decide_regime("neutral", 18.0, 18.0, 23.0, 23.8, CONFIG)
    assert regime == "neutral"
    # hot day but cold stretch
    regime, _ = decide_regime("neutral", 18.0, 15.0, 26.0, 23.8, CONFIG)
    assert regime == "neutral"


def test_regime_transitions_are_free():
    from home_energy_planner.climate_core import decide_regime

    # the 2026-07-06 failure shape: parked in neutral, forecast turns cold
    # overnight -> heat immediately, nothing may hold the pump off
    regime, _ = decide_regime("neutral", 13.0, 14.0, 15.0, 23.8, CONFIG)
    assert regime == "heat"
    # direct reversals allowed when conditions genuinely swing
    regime, _ = decide_regime("heat", 18.0, 18.0, 26.0, 23.8, CONFIG)
    assert regime == "cool"
    regime, _ = decide_regime("cool", 10.0, 12.0, 15.0, 23.8, CONFIG)
    assert regime == "heat"


def test_regime_heat_exit_hysteresis():
    from home_energy_planner.climate_core import decide_regime

    # inside the forecast margin band (14..15): heat holds, but neutral
    # does not newly enter heat
    assert decide_regime("heat", 14.5, 15.0, 20.0, 23.8, CONFIG)[0] == "heat"
    assert decide_regime("neutral", 14.5, 15.0, 20.0, 23.8, CONFIG)[0] == "neutral"
    # clear of the margin: heat releases
    assert decide_regime("heat", 15.5, 15.0, 20.0, 23.8, CONFIG)[0] == "neutral"
    # room-mean margin band (23.2..23.4) behaves the same way
    assert decide_regime("heat", 16.0, 15.0, 20.0, 23.3, CONFIG)[0] == "heat"
    assert decide_regime("neutral", 16.0, 15.0, 20.0, 23.3, CONFIG)[0] == "neutral"


def test_regime_cool_exit_hysteresis_and_heat_override():
    from home_energy_planner.climate_core import decide_regime

    # cool holds inside its exit margins (mean 16.5 >= 16, max 24.5 >= 24)
    assert decide_regime("cool", 18.0, 16.5, 24.5, 23.8, CONFIG)[0] == "cool"
    # released once clearly below the margins
    assert decide_regime("cool", 18.0, 15.5, 24.5, 23.8, CONFIG)[0] == "neutral"
    # overcooled room mean under the backup net breaks cool immediately
    assert decide_regime("cool", 18.0, 18.0, 26.0, 23.0, CONFIG)[0] == "heat"


def test_cool_active_night_block_fallback_and_surplus():
    from home_energy_planner.climate_core import cool_active_now

    # no price horizon (cool_windows=None): the fixed night block applies
    assert cool_active_now(23, 0.0, CONFIG)  # night
    assert cool_active_now(3, 0.0, CONFIG)  # night wraps past midnight
    assert not cool_active_now(14, 0.0, CONFIG)  # afternoon, no surplus
    assert cool_active_now(14, 900.0, CONFIG)  # afternoon on solar surplus
    assert not cool_active_now(9, 0.0, CONFIG)  # block ends at 09


def test_cool_active_follows_ranked_windows_over_night_block():
    from home_energy_planner.climate_core import cool_active_now

    # with a horizon, windows replace the clock entirely: active iff a
    # window covers the current quarter (index 0), whatever the hour
    assert cool_active_now(14, 0.0, CONFIG, cool_windows=[(0, 8)])
    assert not cool_active_now(23, 0.0, CONFIG, cool_windows=[(4, 12)])
    # empty horizon-backed plan means no cheap quarters right now
    assert not cool_active_now(23, 0.0, CONFIG, cool_windows=[])
    # surplus overrides regardless of windows
    assert cool_active_now(14, 900.0, CONFIG, cool_windows=[])


def test_plan_cool_windows_prefers_cheap_and_cool_quarters():
    from home_energy_planner.climate_core import plan_cool_windows

    # flat prices, hot afternoon (hours 0-11) then cool night (12-23):
    # the COP penalty pushes the whole budget into the cool half
    prices = [10.0] * 96
    forecast = hours([26.0] * 12 + [15.0] * 12)
    windows = plan_cool_windows(prices, forecast, CONFIG)
    assert windows == [(48, 96)]  # exactly the cool night half

    # a deep price dip mid-afternoon outweighs the warm-air penalty:
    # 26 C costs 26*0.3 = 7.8 c over 15 C air, a 9 c dip beats it
    dipped = list(prices)
    for i in range(16, 24):  # 04:00-06:00 into the hot half
        dipped[i] = 1.0
    windows = plan_cool_windows(dipped, forecast, CONFIG)
    assert any(start <= 16 < end for start, end in windows)

    # no forecast: pure price ranking still works
    assert plan_cool_windows(dipped, [], CONFIG) != []
    # unusably short horizon
    assert plan_cool_windows([10.0, 10.0], forecast, CONFIG) == []


def test_plan_budget_windows_fills_budget_on_flat_prices():
    from home_energy_planner.price_windows import (
        plan_budget_windows,
        plan_cheap_windows,
    )

    flat = [10.0] * 48
    # the gated planner refuses a flat day; the budget planner must not —
    # it places required runtime, it does not decide whether to run
    assert plan_cheap_windows(
        flat, min_run_quarters=4, budget_quarters=24, margin_cents=1.0
    ) == []
    placed = plan_budget_windows(flat, min_run_quarters=4, budget_quarters=24)
    assert sum(end - start for start, end in placed) == 24


# --- projection ---------------------------------------------------------------


def test_projection_quarter_zero_matches_live_target():
    from home_energy_planner.climate_core import project_targets

    inputs = make_inputs(
        hourly_forecast=hours([5.0] * 48),
        future_all_in=[12.0] * 60 + [6.0] * 36,
    )
    live = compute_climate_target(inputs, CONFIG)
    projected = project_targets(inputs, CONFIG)
    assert len(projected) == len(inputs.future_all_in)
    assert projected[0] == live.target


def test_projection_follows_price_shape_forward():
    from home_energy_planner.climate_core import project_targets

    # expensive first 24 quarters, cheap after: the projection should ask
    # for more heat once the window it sees turns cheap-relative
    prices = [15.0] * 24 + [5.0] * 72
    inputs = make_inputs(
        hourly_forecast=hours([5.0] * 48), future_all_in=prices
    )
    projected = project_targets(inputs, CONFIG)
    assert projected[0] < projected[40]


def test_projection_tracks_forecast_temperature():
    from home_energy_planner.climate_core import project_targets

    # mild now, cold from hour 12 on: later quarters land in a colder
    # 12 h window and get a higher weather base
    inputs = make_inputs(
        hourly_forecast=hours([15.0] * 12 + [-5.0] * 36),
        future_all_in=[12.0] * 96,
    )
    projected = project_targets(inputs, CONFIG)
    assert projected[-1] > projected[0]


def test_projection_includes_extra_offset():
    from home_energy_planner.climate_core import project_targets

    inputs = make_inputs(hourly_forecast=hours([5.0] * 48))
    plain = project_targets(inputs, CONFIG)
    shifted = project_targets(inputs, CONFIG, extra_offset=1.0)
    assert shifted[0] == plain[0] + 1.0


def test_projection_empty_price_horizon():
    from home_energy_planner.climate_core import project_targets

    inputs = make_inputs(future_all_in=[])
    assert project_targets(inputs, CONFIG) == []
