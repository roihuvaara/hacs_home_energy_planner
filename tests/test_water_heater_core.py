from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "custom_components"))

from home_energy_planner.water_heater_core import (  # noqa: E402
    WaterHeaterInputs,
    compute_water_heater_mode,
    normalized_power,
)


def make_inputs(**overrides):
    defaults = dict(
        future_all_in=[12.0] * 96,
        grid_export_w=0.0,
        upcoming_solar_kwh=0.0,
    )
    defaults.update(overrides)
    return WaterHeaterInputs(**defaults)


def test_measured_export_selects_solar_boost():
    result = compute_water_heater_mode(make_inputs(grid_export_w=800.0))
    assert result.mode == "solar_boost"
    assert result.target_temp == 66
    below = compute_water_heater_mode(make_inputs(grid_export_w=300.0))
    assert below.mode != "solar_boost"


def test_upcoming_solar_preserves_headroom_over_cheap_boost():
    # current quarter is the cheapest of the day, but strong solar is coming:
    # keep the tank at normal so the surplus has somewhere to go (ADR 0007)
    future = [5.0] * 4 + [15.0] * 92
    preserved = compute_water_heater_mode(
        make_inputs(future_all_in=future, upcoming_solar_kwh=10.0)
    )
    assert preserved.mode == "normal"
    assert preserved.buffer_preserve
    boosted = compute_water_heater_mode(
        make_inputs(future_all_in=future, upcoming_solar_kwh=1.0)
    )
    assert boosted.mode == "cheap_boost"


def test_surplus_beats_preserve():
    result = compute_water_heater_mode(
        make_inputs(grid_export_w=1000.0, upcoming_solar_kwh=10.0)
    )
    assert result.mode == "solar_boost"
    assert not result.buffer_preserve


def test_cheap_window_boosts_now_with_min_run_length():
    future = [6.0] * 6 + [14.0] * 90
    result = compute_water_heater_mode(make_inputs(future_all_in=future))
    assert result.mode == "cheap_boost"
    assert result.cheap_windows
    assert all(end - start >= 3 for start, end in result.cheap_windows)
    assert result.cheap_windows[0][0] == 0


def test_cheap_window_later_does_not_boost_now():
    future = [14.0] * 20 + [6.0] * 8 + [14.0] * 68
    result = compute_water_heater_mode(make_inputs(future_all_in=future))
    assert result.mode == "normal"
    # the window is planned, just not active yet
    assert result.cheap_windows and result.cheap_windows[0][0] == 20


def test_zigzag_prices_produce_no_boost_windows():
    # alternating cheap/expensive quarters: no min-run window clears the
    # margin, so the compressor is never asked to chase 15-minute dips
    future = [6.0, 14.0] * 48
    result = compute_water_heater_mode(make_inputs(future_all_in=future))
    assert result.mode == "normal"
    assert result.cheap_windows == []


def test_single_cheap_dip_becomes_full_hour_run():
    # one 15-minute dip is worth using only as part of a full-length run
    future = [7.0] + [11.0] * 95
    result = compute_water_heater_mode(make_inputs(future_all_in=future))
    if result.cheap_windows:
        assert all(end - start >= 3 for start, end in result.cheap_windows)


def test_expensive_stretch_holds():
    future = [30.0] * 8 + [10.0] * 88
    result = compute_water_heater_mode(make_inputs(future_all_in=future))
    assert result.mode == "hold"
    assert result.target_temp == 51


def test_flat_prices_run_normal():
    # spread below the deadband: no boosting, no holding, whatever the level
    for level in (3.0, 30.0):
        result = compute_water_heater_mode(make_inputs(future_all_in=[level] * 96))
        assert result.mode == "normal"


def test_negative_price_quarter_boosts():
    future = [-1.0] * 4 + [8.0] * 44 + [15.0] * 48
    result = compute_water_heater_mode(make_inputs(future_all_in=future))
    assert result.mode == "cheap_boost"


def test_empty_horizon_runs_normal():
    result = compute_water_heater_mode(make_inputs(future_all_in=[]))
    assert result.mode == "normal"


def test_normalized_power_maps_boost_to_on_and_unknown_to_none():
    assert normalized_power("off") == "off"
    assert normalized_power("heat_pump") == "on"
    # performance is the boost mode: a running state, never fought as an off
    assert normalized_power("performance") == "on"
    # unavailable / unknown modes leave the heater alone
    assert normalized_power(None) is None
    assert normalized_power("eco") is None


# --- MILP setpoint actuation (thermal-battery mode) ---------------------------

from home_energy_planner.water_heater_core import milp_setpoint  # noqa: E402


def test_milp_setpoint_maps_window_to_ceiling_and_coast_to_floor():
    assert milp_setpoint(True, 66.0, 50.0) == ("milp_heat", 66)
    assert milp_setpoint(False, 66.0, 50.0) == ("milp_coast", 50)
    # rounds, never truncates
    assert milp_setpoint(True, 65.6, 50.4) == ("milp_heat", 66)
