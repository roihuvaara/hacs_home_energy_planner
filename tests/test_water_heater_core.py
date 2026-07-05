from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "custom_components"))

from home_energy_planner.water_heater_core import (  # noqa: E402
    WaterHeaterInputs,
    compute_water_heater_mode,
)


def make_inputs(**overrides):
    defaults = dict(
        current_vat=6.0,
        current_all_in=12.0,
        future_all_in=[12.0] * 96,
        solar_now_w=0.0,
        solar_next_hour_w=0.0,
        solar_remaining_today_kwh=0.0,
        battery_soc_pct=50.0,
    )
    defaults.update(overrides)
    return WaterHeaterInputs(**defaults)


def test_live_surplus_selects_solar_boost():
    result = compute_water_heater_mode(make_inputs(solar_now_w=2000.0))
    assert result.mode == "solar_boost"
    assert result.target_temp == 66


def test_moderate_surplus_needs_full_battery():
    partial = compute_water_heater_mode(make_inputs(solar_now_w=1400.0, battery_soc_pct=80.0))
    assert partial.mode != "solar_boost"
    full = compute_water_heater_mode(make_inputs(solar_now_w=1400.0, battery_soc_pct=96.0))
    assert full.mode == "solar_boost"


def test_strong_solar_day_preserves_buffer_at_normal():
    result = compute_water_heater_mode(
        make_inputs(solar_remaining_today_kwh=20.0, current_vat=3.0)
    )
    # would be cheap_boost on vat alone, but the sunny forecast preserves headroom
    assert result.mode == "normal"
    assert result.buffer_preserve
    assert result.target_temp == 55


def test_no_preserve_when_price_expensive():
    result = compute_water_heater_mode(
        make_inputs(
            solar_remaining_today_kwh=20.0,
            current_vat=9.0,
            current_all_in=20.0,
            future_all_in=[12.0] * 96,
        )
    )
    assert not result.buffer_preserve
    assert result.mode == "hold"


def test_cheap_boost_branches():
    # very cheap VAT price alone
    assert compute_water_heater_mode(make_inputs(current_vat=3.0)).mode == "cheap_boost"
    # well below horizon average
    below_avg = make_inputs(current_all_in=10.0, future_all_in=[10.0] + [14.0] * 95)
    assert compute_water_heater_mode(below_avg).mode == "cheap_boost"
    # at the horizon minimum
    at_min = make_inputs(current_all_in=12.0, future_all_in=[12.0] + [12.5] * 40 + [18.0] * 55)
    assert compute_water_heater_mode(at_min).mode == "cheap_boost"


def test_normal_and_hold_fallthrough():
    # current must sit clear of the horizon minimum + 0.75 margin or the
    # cheap-boost min rule catches it
    normal = compute_water_heater_mode(
        make_inputs(current_vat=6.5, current_all_in=14.0, future_all_in=[14.0] * 40 + [12.0] + [14.5] * 55)
    )
    assert normal.mode == "normal"
    hold = compute_water_heater_mode(
        make_inputs(current_vat=15.0, current_all_in=25.0, future_all_in=[25.0] + [10.0] * 95)
    )
    assert hold.mode == "hold"
    assert hold.target_temp == 51
