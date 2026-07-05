from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "custom_components"))

from home_energy_planner.ilp_core import (  # noqa: E402
    IlpInputs,
    compute_ilp_action,
)


def make_inputs(**overrides):
    defaults = dict(
        room_temp=23.0,
        room_humidity=50.0,
        grid_export_w=0.0,
        future_all_in=[12.0] * 96,
        outdoor_forecast_max_24h=20.0,
        currently_cooling=False,
        currently_drying=False,
    )
    defaults.update(overrides)
    return IlpInputs(**defaults)


def test_hard_max_cools_at_any_price():
    expensive = [30.0] + [10.0] * 95
    result = compute_ilp_action(
        make_inputs(room_temp=25.8, future_all_in=expensive)
    )
    assert result.action == "cool"
    assert "hard max" in result.reason


def test_warm_room_cools_on_surplus_or_cheap_only():
    surplus = compute_ilp_action(make_inputs(room_temp=24.8, grid_export_w=900.0))
    assert surplus.action == "cool"
    cheap = compute_ilp_action(
        make_inputs(room_temp=24.8, future_all_in=[10.0] + [14.0] * 95)
    )
    assert cheap.action == "cool"
    expensive = compute_ilp_action(
        make_inputs(room_temp=24.8, future_all_in=[20.0] + [10.0] * 95)
    )
    assert expensive.action == "off"


def test_precool_needs_surplus_and_hot_forecast():
    ready = make_inputs(
        room_temp=24.0, grid_export_w=900.0, outdoor_forecast_max_24h=28.0
    )
    assert compute_ilp_action(ready).action == "cool"
    no_surplus = make_inputs(room_temp=24.0, outdoor_forecast_max_24h=28.0)
    assert compute_ilp_action(no_surplus).action == "off"
    mild_day = make_inputs(
        room_temp=24.0, grid_export_w=900.0, outdoor_forecast_max_24h=20.0
    )
    assert compute_ilp_action(mild_day).action == "off"


def test_running_cooldown_finishes_to_stop_threshold():
    running = compute_ilp_action(
        make_inputs(room_temp=24.0, currently_cooling=True)
    )
    assert running.action == "cool"
    done = compute_ilp_action(make_inputs(room_temp=23.3, currently_cooling=True))
    assert done.action == "off"


def test_comfortable_room_stays_off():
    result = compute_ilp_action(make_inputs(room_temp=22.5, grid_export_w=2000.0))
    assert result.action == "off"
    unknown = compute_ilp_action(make_inputs(room_temp=None, grid_export_w=2000.0))
    assert unknown.action == "off"


def test_very_humid_room_dries_at_any_price():
    expensive = [30.0] + [10.0] * 95
    result = compute_ilp_action(
        make_inputs(room_humidity=72.0, future_all_in=expensive)
    )
    assert result.action == "dry"
    assert "hard max" in result.reason


def test_humid_room_dries_on_surplus_or_cheap_only():
    surplus = compute_ilp_action(make_inputs(room_humidity=65.0, grid_export_w=900.0))
    assert surplus.action == "dry"
    expensive = compute_ilp_action(
        make_inputs(room_humidity=65.0, future_all_in=[20.0] + [10.0] * 95)
    )
    assert expensive.action == "off"


def test_cooling_takes_priority_over_dry():
    result = compute_ilp_action(
        make_inputs(room_temp=26.0, room_humidity=75.0)
    )
    assert result.action == "cool"


def test_dry_run_finishes_to_stop_threshold():
    running = compute_ilp_action(make_inputs(room_humidity=58.0, currently_drying=True))
    assert running.action == "dry"
    done = compute_ilp_action(make_inputs(room_humidity=53.0, currently_drying=True))
    assert done.action == "off"


def test_unknown_humidity_never_dries():
    result = compute_ilp_action(make_inputs(room_humidity=None, grid_export_w=2000.0))
    assert result.action == "off"
