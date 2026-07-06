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
        room_humidity=45.0,
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
    result = compute_ilp_action(make_inputs(room_temp=23.1, grid_export_w=2000.0))
    assert result.action == "off"
    unknown = compute_ilp_action(make_inputs(room_temp=None, grid_export_w=2000.0))
    assert unknown.action == "off"


def test_very_humid_room_dries_at_any_price():
    expensive = [30.0] + [10.0] * 95
    result = compute_ilp_action(
        make_inputs(room_humidity=60.0, future_all_in=expensive)
    )
    assert result.action == "dry"
    assert "hard max" in result.reason


def test_humid_room_dries_on_surplus_or_cheap_only():
    surplus = compute_ilp_action(make_inputs(room_humidity=52.0, grid_export_w=900.0))
    assert surplus.action == "dry"
    expensive = compute_ilp_action(
        make_inputs(room_humidity=52.0, future_all_in=[20.0] + [10.0] * 95)
    )
    assert expensive.action == "off"


def test_cooling_takes_priority_over_dry():
    result = compute_ilp_action(
        make_inputs(room_temp=26.0, room_humidity=75.0)
    )
    assert result.action == "cool"


def test_dry_run_finishes_to_stop_threshold():
    running = compute_ilp_action(make_inputs(room_humidity=47.0, currently_drying=True))
    assert running.action == "dry"
    done = compute_ilp_action(make_inputs(room_humidity=43.0, currently_drying=True))
    assert done.action == "off"


def test_unknown_humidity_never_dries():
    result = compute_ilp_action(make_inputs(room_humidity=None, grid_export_w=2000.0))
    assert result.action == "off"


def test_dry_never_runs_below_room_floor():
    # 22.9: below the dry floor (23.0) but above heat entry (22.8) —
    # isolates the floor from the heat ladder
    cool_cheap = compute_ilp_action(
        make_inputs(
            room_temp=22.9,
            room_humidity=53.0,
            future_all_in=[10.0] + [14.0] * 95,
        )
    )
    assert cool_cheap.action == "off"
    assert "below dry floor" in cool_cheap.reason
    # even past the hard humidity limit (owner decision: no bypass)
    cool_humid = compute_ilp_action(
        make_inputs(room_temp=22.9, room_humidity=60.0, future_all_in=[20.0] + [10.0] * 95)
    )
    assert cool_humid.action == "off"
    # a running dry stops when the room falls through the floor
    running = compute_ilp_action(
        make_inputs(
            room_temp=22.9,
            room_humidity=47.0,
            currently_drying=True,
            future_all_in=[20.0] + [10.0] * 95,
        )
    )
    assert running.action == "off"
    # unknown room temperature is conservative: no dry
    unknown = compute_ilp_action(make_inputs(room_temp=None, room_humidity=60.0))
    assert unknown.action == "off"


def test_cold_room_heats_instead_of_dry():
    # the full 2026-07-06 morning: 22.5 C, humid, cheap -> the right
    # answer is heat, never dry
    result = compute_ilp_action(
        make_inputs(
            room_temp=22.5,
            room_humidity=53.0,
            future_all_in=[10.0] + [14.0] * 95,
        )
    )
    assert result.action == "heat"
    assert result.target_temp == 23.5


def test_heat_assist_ladder():
    # hard comfort net: heat at any price
    expensive = [30.0] + [10.0] * 95
    hard = compute_ilp_action(make_inputs(room_temp=21.8, future_all_in=expensive))
    assert hard.action == "heat"
    assert "hard min" in hard.reason
    # assist band needs cheap or surplus
    cheap = compute_ilp_action(
        make_inputs(room_temp=22.6, future_all_in=[10.0] + [14.0] * 95)
    )
    assert cheap.action == "heat"
    surplus = compute_ilp_action(
        make_inputs(room_temp=22.6, grid_export_w=900.0, future_all_in=expensive)
    )
    assert surplus.action == "heat"
    pricey = compute_ilp_action(make_inputs(room_temp=22.6, future_all_in=expensive))
    assert pricey.action == "off"
    # finishing: keep heating up to the stop line, then off
    running = compute_ilp_action(
        make_inputs(room_temp=23.1, currently_heating=True, future_all_in=expensive)
    )
    assert running.action == "heat"
    done = compute_ilp_action(
        make_inputs(room_temp=23.4, currently_heating=True, future_all_in=expensive)
    )
    assert done.action == "off"


def test_heat_assist_never_fights_slab_cooling():
    result = compute_ilp_action(
        make_inputs(room_temp=22.5, slab_cooling=True, future_all_in=[10.0] + [14.0] * 95)
    )
    assert result.action == "off"


def test_slab_cooling_raises_ilp_threshold():
    # 24.8 normally cools on surplus; with the slab regime active the
    # bumped threshold (25.0) keeps the ILP out of the slab's way
    normal = compute_ilp_action(make_inputs(room_temp=24.8, grid_export_w=900.0))
    assert normal.action == "cool"
    deferred = compute_ilp_action(
        make_inputs(room_temp=24.8, grid_export_w=900.0, slab_cooling=True)
    )
    assert deferred.action == "off"
    # the hard max still overrides regardless of the slab
    hard = compute_ilp_action(make_inputs(room_temp=25.6, slab_cooling=True))
    assert hard.action == "cool"
