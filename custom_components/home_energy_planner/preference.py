"""Preference learning from manual overrides.

Every respected manual override is also evidence about the household's
real comfort preferences. This module keeps a persistent event log
(each override with the context it happened in) and folds it into
small, bounded threshold adjustments:

- each event contributes one fixed step in the direction it points
  (magnitude of the user's change is deliberately ignored — robust to
  one-off extremes);
- opposite events cancel;
- events fade with a 30-day half-life so old habits stop steering;
- the folded sum is clipped to a hard cap per parameter, so learning
  can trim a threshold but never walk it somewhere absurd.

Only mappings whose direction is unambiguous adjust anything; the rest
(Versati hvac flips) are logged for the monthly report and manual
tuning.

No Home Assistant imports; unit-testable standalone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

RING_CAP = 200
DEDUPE_HOURS = 6.0
HALF_LIFE_DAYS = 30.0
VALUE_TOLERANCE = 0.11

# adjustment key -> hard cap (offset is always within [-cap, +cap])
CAPS = {
    "climate_target_offset": 1.0,
    "ilp_cool_room_above": 1.0,
    "ilp_dry_humidity_above": 4.0,
    "ilp_dry_room_floor": 0.5,
    "ilp_heat_room_below": 0.5,
    # tank target per weekday (0=Mon..6=Sun): repeated overrides on the
    # same weekday reveal weekly rhythms (gym days, laundry, sauna-ish
    # habits) without assuming which days they are. A dedicated sauna
    # sensor is the planned better signal for event-driven pre-heating.
    **{f"water_weekday_{day}": 6.0 for day in range(7)},
}

MODULE_CLIMATE = "climate"
MODULE_CLIMATE_HVAC = "climate_hvac"
MODULE_ILP = "ilp"
MODULE_WATER_HEATER = "water_heater"

_ILP_OFF_MODES = ("off", "fan_only")


@dataclass(frozen=True)
class PreferenceEvent:
    when: datetime
    module: str  # climate | climate_hvac | ilp | water_heater
    planner_value: object
    manual_value: object
    context: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "when": self.when.isoformat(),
            "module": self.module,
            "planner_value": self.planner_value,
            "manual_value": self.manual_value,
            "context": dict(self.context),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PreferenceEvent":
        return cls(
            when=datetime.fromisoformat(str(data["when"])),
            module=str(data["module"]),
            planner_value=data.get("planner_value"),
            manual_value=data.get("manual_value"),
            context=dict(data.get("context") or {}),
        )


def _same_value(a: object, b: object) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= VALUE_TOLERANCE
    return a == b


class PreferenceLog:
    def __init__(self, events: list[PreferenceEvent] | None = None) -> None:
        self.events: list[PreferenceEvent] = list(events or [])

    def append(self, event: PreferenceEvent) -> bool:
        """Store the event; False when it repeats the most recent manual
        value for the same module inside the dedupe window (restart
        re-detections and back-to-back tweaks are one preference signal,
        not several)."""
        last = next(
            (e for e in reversed(self.events) if e.module == event.module), None
        )
        if (
            last is not None
            and event.when - last.when <= timedelta(hours=DEDUPE_HOURS)
            and _same_value(last.manual_value, event.manual_value)
        ):
            return False
        self.events.append(event)
        del self.events[:-RING_CAP]
        return True

    def counts_by_module(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.module] = counts.get(event.module, 0) + 1
        return counts

    def as_dict(self) -> dict:
        return {"events": [event.as_dict() for event in self.events]}

    @classmethod
    def from_dict(cls, data: dict) -> "PreferenceLog":
        events = []
        for row in data.get("events") or []:
            try:
                events.append(PreferenceEvent.from_dict(row))
            except (KeyError, TypeError, ValueError):
                continue  # one corrupt row must not lose the log
        return cls(events)


def _weight(event: PreferenceEvent, now: datetime) -> float:
    age_days = max(0.0, (now - event.when).total_seconds() / 86400.0)
    return math.pow(0.5, age_days / HALF_LIFE_DAYS)


def _climate_step(event: PreferenceEvent) -> dict[str, float]:
    try:
        delta = float(event.manual_value) - float(event.planner_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}
    if abs(delta) <= VALUE_TOLERANCE:
        return {}
    return {"climate_target_offset": math.copysign(0.1, delta)}


def _water_step(event: PreferenceEvent) -> dict[str, float]:
    """Tank-target overrides learn per weekday: 1 C per event in the
    override's direction, so e.g. repeatedly boosting on Fridays warms
    Fridays only."""
    try:
        delta = float(event.manual_value) - float(event.planner_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}
    if abs(delta) <= VALUE_TOLERANCE:
        return {}
    return {f"water_weekday_{event.when.weekday()}": math.copysign(1.0, delta)}


def _ilp_step(event: PreferenceEvent) -> dict[str, float]:
    planner = str(event.planner_value)
    manual = str(event.manual_value)
    room = event.context.get("room_temp")
    steps: dict[str, float] = {}
    if manual == "cool" and planner in (*_ILP_OFF_MODES, "dry"):
        steps["ilp_cool_room_above"] = -0.1
    elif manual in _ILP_OFF_MODES and planner == "cool":
        steps["ilp_cool_room_above"] = +0.1
    elif manual == "dry" and planner in _ILP_OFF_MODES:
        steps["ilp_dry_humidity_above"] = -0.5
        # forcing dry in a room the floor would block: floor is too high
        if isinstance(room, (int, float)) and room < 23.0:
            steps["ilp_dry_room_floor"] = -0.1
    elif manual in _ILP_OFF_MODES and planner == "dry":
        steps["ilp_dry_humidity_above"] = +0.5
        # stopping dry in an already-coolish room: raise the floor
        if isinstance(room, (int, float)) and room < 23.5:
            steps["ilp_dry_room_floor"] = +0.1
    elif manual == "heat" and planner in (*_ILP_OFF_MODES, "dry"):
        steps["ilp_heat_room_below"] = +0.1  # wants assist to start earlier
    elif manual in _ILP_OFF_MODES and planner == "heat":
        steps["ilp_heat_room_below"] = -0.1
    return steps


def derive_adjustments(
    events: list[PreferenceEvent], now: datetime
) -> dict[str, float]:
    """Fold the event log into bounded parameter offsets (see module
    docstring for the mechanics). Keys are always all of CAPS, zeros
    included, so the published `learned` attribute has a stable shape."""
    sums = {key: 0.0 for key in CAPS}
    for event in events:
        if event.module == MODULE_CLIMATE:
            steps = _climate_step(event)
        elif event.module == MODULE_ILP:
            steps = _ilp_step(event)
        elif event.module == MODULE_WATER_HEATER:
            steps = _water_step(event)
        else:
            continue  # log-only modules (climate_hvac)
        weight = _weight(event, now)
        for key, step in steps.items():
            sums[key] += step * weight
    return {
        key: round(max(-cap, min(cap, sums[key])), 2) for key, cap in CAPS.items()
    }
