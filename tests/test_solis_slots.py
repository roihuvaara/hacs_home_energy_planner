from pathlib import Path
import sys

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "custom_components" / "home_energy_planner"),
)

from solis_slots import (  # noqa: E402
    SLOT_COUNT,
    SlotSpec,
    WriteOp,
    clamp_slot_values,
    diff_tables,
    diff_write_ops,
    find_cross_side_overlaps,
    normalize_table,
    validate_table,
    wall_clock_ranges,
)


def slot(time="00:00-00:00", enabled=False, current=0, soc=19):
    return SlotSpec(time=time, enabled=enabled, current=current, soc=soc)


def table(*slots):
    return list(slots) + [SlotSpec()] * (SLOT_COUNT - len(slots))


def test_normalize_pads_to_six_disabled_slots():
    result = normalize_table([{"time": "12:00-13:00", "enabled": True, "current": 8, "soc": 30}])
    assert len(result) == SLOT_COUNT
    assert result[0] == slot("12:00-13:00", True, 8, 30)
    assert all(item == SlotSpec() for item in result[1:])


def test_normalize_rejects_more_than_six():
    with pytest.raises(ValueError):
        normalize_table([{}] * 7)


def test_validate_flags_bad_time_and_enabled_empty_window():
    problems = validate_table(table(slot("25:00-26:00"), slot(enabled=True)), "charge")
    assert any("invalid time" in problem for problem in problems)
    assert any("enabled but time" in problem for problem in problems)


def test_validate_accepts_normal_table():
    assert validate_table(table(slot("14:00-19:45", True, 3, 98)), "charge") == []


def test_wall_clock_ranges_wraps_midnight():
    assert wall_clock_ranges("22:45-01:15") == [(22 * 60 + 45, 24 * 60), (0, 75)]
    assert wall_clock_ranges("12:00-13:00") == [(12 * 60, 13 * 60)]


def test_cross_side_overlap_detected_including_wrap():
    charge = table(slot("23:30-00:30", True, 8, 60))
    discharge = table(slot("00:00-01:00", True, 0, 30))
    conflicts = find_cross_side_overlaps(charge, discharge)
    assert len(conflicts) == 1
    assert "charge slot 1" in conflicts[0]


def test_cross_side_overlap_ignores_disabled_and_disjoint():
    charge = table(slot("02:00-04:00", True, 8, 60))
    discharge = table(
        slot("02:00-04:00", False, 0, 30),  # disabled: no conflict
        slot("04:00-06:00", True, 0, 30),  # shares boundary only: no conflict
    )
    assert find_cross_side_overlaps(charge, discharge) == []


def test_diff_noop_produces_no_ops():
    current = table(slot("14:00-19:45", True, 3, 98))
    ops = diff_write_ops(
        current_charge=current,
        current_discharge=table(),
        desired_charge=[SlotSpec(**item.as_dict()) for item in current],
        desired_discharge=table(),
    )
    assert ops == []


def test_diff_orders_disable_then_fields_then_enable():
    current_charge = table(slot("10:00-11:00", True, 8, 50))
    desired_charge = table()  # remove the window entirely
    current_discharge = table()
    desired_discharge = table(slot("20:00-21:00", True, 0, 30))

    ops = diff_write_ops(
        current_charge=current_charge,
        current_discharge=current_discharge,
        desired_charge=desired_charge,
        desired_discharge=desired_discharge,
    )
    kinds = [(op.side, op.field, op.value) for op in ops]
    # Disable of the removed charge slot must come first...
    assert kinds[0] == ("charge", "enabled", False)
    # ...and the discharge enable must come last, after its field writes.
    assert kinds[-1] == ("discharge", "enabled", True)
    field_ops = kinds[1:-1]
    assert ("discharge", "time", "20:00-21:00") in field_ops
    assert all(field != "enabled" for _, field, _ in field_ops)


def test_diff_writes_only_changed_fields():
    current = table(slot("14:00-19:45", True, 3, 98))
    desired = table(slot("14:00-19:45", True, 3, 95))  # soc change only
    ops = diff_write_ops(
        current_charge=current,
        current_discharge=table(),
        desired_charge=desired,
        desired_discharge=table(),
    )
    assert ops == [WriteOp("charge", 1, "soc", 95)]


def test_verification_diff_reports_field_level_mismatch():
    expected = table(slot("12:00-13:00", True, 8, 30))
    actual = table(slot("00:00-00:00", True, 8, 30))  # device reverted the time
    mismatches = diff_tables(expected, actual, "charge")
    assert mismatches == [
        {
            "side": "charge",
            "slot": 1,
            "field": "time",
            "expected": "12:00-13:00",
            "actual": "00:00-00:00",
        }
    ]


def test_clamp_raises_soc_to_device_minimum():
    # live incident 2026-07-14: hold slot soc 18 (reserve floor) rejected by
    # slot1_discharge_soc whose dynamic min is over-discharge SOC + 1 = 19
    ranges = {(1, "soc"): (19.0, 100.0)}
    clamped, notes = clamp_slot_values(
        table(slot("22:00-06:00", True, 0, 18)),
        lambda s, f: ranges.get((s, f)),
        "discharge",
    )
    assert clamped[0].soc == 19
    assert clamped[0].current == 0
    assert notes == ["discharge slot 1 soc: 18 -> 19 (device range 19.0-100.0)"]


def test_clamp_lowers_current_to_device_maximum():
    ranges = {(1, "current"): (0.0, 50.0)}
    clamped, notes = clamp_slot_values(
        table(slot("02:00-04:00", True, 62, 90)),
        lambda s, f: ranges.get((s, f)),
        "charge",
    )
    assert clamped[0].current == 50
    assert clamped[0].soc == 90
    assert len(notes) == 1


def test_clamp_in_range_and_unknown_range_untouched():
    original = table(slot("02:00-04:00", True, 20, 90), slot("12:00-13:00", True, 5, 40))
    clamped, notes = clamp_slot_values(
        original,
        lambda s, f: (0.0, 100.0) if s == 1 else None,
        "charge",
    )
    assert clamped == original
    assert notes == []


def test_clamp_rounds_inward_on_fractional_bounds():
    ranges = {(1, "soc"): (18.5, 99.5)}
    low, _ = clamp_slot_values(
        table(slot("22:00-06:00", True, 0, 18)),
        lambda s, f: ranges.get((s, f)),
        "discharge",
    )
    high, _ = clamp_slot_values(
        table(slot("22:00-06:00", True, 0, 100)),
        lambda s, f: ranges.get((s, f)),
        "discharge",
    )
    assert low[0].soc == 19
    assert high[0].soc == 99
