"""Pure Solis slot-table logic: model, validation, diffing, verification.

Solis charge/discharge slots are date-less daily-recurring wall-clock
windows. This module knows nothing about Home Assistant; the writer glue
in ``solis_writer.py`` executes the ops it produces.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

SLOT_COUNT = 6
SLOT_FIELDS = ("time", "current", "soc", "enabled")
EMPTY_TIME = "00:00-00:00"
DEFAULT_SOC = 19

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]|24):([0-5]\d)$")


@dataclass(frozen=True)
class SlotSpec:
    time: str = EMPTY_TIME
    enabled: bool = False
    current: int = 0
    soc: int = DEFAULT_SOC

    def as_dict(self) -> dict[str, object]:
        return {
            "time": self.time,
            "enabled": self.enabled,
            "current": self.current,
            "soc": self.soc,
        }


@dataclass(frozen=True)
class WriteOp:
    side: str  # "charge" | "discharge"
    slot: int  # 1-based
    field: str  # one of SLOT_FIELDS
    value: object


def normalize_table(raw: Sequence[Mapping[str, object]] | None) -> list[SlotSpec]:
    """Coerce a service payload into exactly SLOT_COUNT SlotSpecs."""

    table = [SlotSpec() for _ in range(SLOT_COUNT)]
    if not raw:
        return table
    if len(raw) > SLOT_COUNT:
        raise ValueError(f"At most {SLOT_COUNT} slots per side, got {len(raw)}")
    for index, item in enumerate(raw):
        table[index] = SlotSpec(
            time=str(item.get("time", EMPTY_TIME)),
            enabled=bool(item.get("enabled", False)),
            current=int(float(item.get("current", 0))),
            soc=int(float(item.get("soc", DEFAULT_SOC))),
        )
    return table


def clamp_slot_values(
    table: Sequence[SlotSpec],
    get_range,
    side: str,
) -> tuple[list[SlotSpec], list[str]]:
    """Clamp current/soc to the device's advertised numeric ranges.

    The inverter narrows some ranges dynamically (discharge SOC min is
    over-discharge SOC + 1), so a planner value can be deterministically
    rejected no matter how often it is retried. ``get_range(slot, field)``
    returns (min, max) — either bound may be None — or None when unknown.
    """
    import math
    from dataclasses import replace

    out: list[SlotSpec] = []
    notes: list[str] = []
    for index, slot in enumerate(table, start=1):
        values = {"current": slot.current, "soc": slot.soc}
        for field_name, value in list(values.items()):
            bounds = get_range(index, field_name)
            if bounds is None:
                continue
            low, high = bounds
            clamped = value
            if low is not None and clamped < low:
                clamped = int(math.ceil(low))
            if high is not None and clamped > high:
                clamped = int(math.floor(high))
            if clamped != value:
                values[field_name] = clamped
                notes.append(
                    f"{side} slot {index} {field_name}: {value} -> {clamped} "
                    f"(device range {low}-{high})"
                )
        out.append(replace(slot, current=values["current"], soc=values["soc"]))
    return out, notes


def validate_table(table: Sequence[SlotSpec], side: str) -> list[str]:
    """Return human-readable problems; empty list means valid."""

    problems: list[str] = []
    for index, slot in enumerate(table, start=1):
        if not _TIME_RE.match(slot.time):
            problems.append(f"{side} slot {index}: invalid time '{slot.time}'")
            continue
        if slot.enabled and slot.time == EMPTY_TIME:
            problems.append(f"{side} slot {index}: enabled but time is {EMPTY_TIME}")
        if not 0 <= slot.current <= 100:
            problems.append(f"{side} slot {index}: current {slot.current} out of range")
        if not 0 <= slot.soc <= 100:
            problems.append(f"{side} slot {index}: soc {slot.soc} out of range")
    return problems


def wall_clock_ranges(time_window: str) -> list[tuple[int, int]]:
    """Minute-of-day ranges for a window; wrap-around yields two ranges."""

    start_raw, end_raw = time_window.split("-")
    start_h, start_m = (int(part) for part in start_raw.split(":"))
    end_h, end_m = (int(part) for part in end_raw.split(":"))
    start = start_h * 60 + start_m
    end = end_h * 60 + end_m
    if end <= start:
        return [(start, 24 * 60), (0, end)]
    return [(start, end)]


def _active_windows(table: Sequence[SlotSpec]) -> list[tuple[int, str]]:
    return [
        (index, slot.time)
        for index, slot in enumerate(table, start=1)
        if slot.enabled and slot.time != EMPTY_TIME
    ]


def find_cross_side_overlaps(
    charge_table: Sequence[SlotSpec],
    discharge_table: Sequence[SlotSpec],
) -> list[str]:
    """Wall-clock collisions between enabled charge and discharge windows.

    Slots recur daily with no date, so any overlap in minute-of-day space is
    a real simultaneous-activation conflict on the inverter.
    """

    conflicts: list[str] = []
    for charge_index, charge_time in _active_windows(charge_table):
        for discharge_index, discharge_time in _active_windows(discharge_table):
            for c_start, c_end in wall_clock_ranges(charge_time):
                if any(
                    c_start < d_end and d_start < c_end
                    for d_start, d_end in wall_clock_ranges(discharge_time)
                ):
                    conflicts.append(
                        f"charge slot {charge_index} ({charge_time}) overlaps "
                        f"discharge slot {discharge_index} ({discharge_time})"
                    )
                    break
    return conflicts


def diff_write_ops(
    *,
    current_charge: Sequence[SlotSpec],
    current_discharge: Sequence[SlotSpec],
    desired_charge: Sequence[SlotSpec],
    desired_discharge: Sequence[SlotSpec],
) -> list[WriteOp]:
    """Field-level diff as an ordered op list.

    Order matters on the inverter: windows being removed are disabled before
    any new times land, and switches are enabled only after their window
    fields are in place.
    """

    disables: list[WriteOp] = []
    field_updates: list[WriteOp] = []
    enables: list[WriteOp] = []

    for side, current_table, desired_table in (
        ("charge", current_charge, desired_charge),
        ("discharge", current_discharge, desired_discharge),
    ):
        for index, (current, desired) in enumerate(
            zip(current_table, desired_table), start=1
        ):
            if current.enabled and not desired.enabled:
                disables.append(WriteOp(side, index, "enabled", False))
            for field in ("time", "current", "soc"):
                if getattr(current, field) != getattr(desired, field):
                    field_updates.append(
                        WriteOp(side, index, field, getattr(desired, field))
                    )
            if desired.enabled and not current.enabled:
                enables.append(WriteOp(side, index, "enabled", True))

    return disables + field_updates + enables


def diff_tables(
    expected: Sequence[SlotSpec],
    actual: Sequence[SlotSpec],
    side: str,
) -> list[dict[str, object]]:
    """Verification mismatches between an intended and an observed table."""

    mismatches: list[dict[str, object]] = []
    for index, (want, got) in enumerate(zip(expected, actual), start=1):
        for field in SLOT_FIELDS:
            if getattr(want, field) != getattr(got, field):
                mismatches.append(
                    {
                        "side": side,
                        "slot": index,
                        "field": field,
                        "expected": getattr(want, field),
                        "actual": getattr(got, field),
                    }
                )
    return mismatches
