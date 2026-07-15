"""Preference learning from manual overrides.

Every respected manual override is also evidence about the household's
real comfort preferences. This module keeps a persistent event log
(each override with the context it happened in) and folds it into
small, bounded threshold adjustments:

- each event contributes one fixed step in the direction it points
  (magnitude of the user's change is deliberately ignored — robust to
  one-off extremes);
- opposite events cancel;
- events are weighted by *resemblance to now* (weather-state kernel:
  outdoor temp, 7-day trailing mean, daylight hours), so a cold-snap
  override resurfaces in the next cold snap regardless of which winter
  it happened in, and a 15 C day in July never borrows November's
  preferences. Calendar features are deliberately absent — an unusually
  warm November should behave like October, not like last November;
- evidence primarily ages by being *contradicted*, not by the clock:
  each event is discounted by the similarity-weighted count of newer
  opposite-direction events on the same parameter (two similar-context
  contradictions halve it). Consistent winters therefore accumulate
  undiminished, a changed preference flips after a few counter-events
  instead of one-per-stale-event, and a summer contradiction cannot
  retire winter evidence. Silence in similar conditions is consent:
  the planner applies the offset and the user isn't fighting it;
- a gentle 2-year calendar half-life remains as the backstop for
  change that never generates counter-events (household turnover);
  events recorded before feature capture existed keep the legacy
  30-day half-life (age is all we know about them);
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
from datetime import date, datetime, timedelta

# multi-season memory: at the household override rate (~1-2/week) this
# holds several years of events; the store is a small JSON blob
RING_CAP = 1000
DEDUPE_HOURS = 6.0
HALF_LIFE_DAYS = 730.0  # slow backstop; contradiction does the retiring
LEGACY_HALF_LIFE_DAYS = 30.0  # events without weather features
# similar-context contradictions needed to halve an old event's weight
CONTRADICTION_HALF_COUNT = 2.0
VALUE_TOLERANCE = 0.11

# weather-state kernel: per-feature Gaussian bandwidths. Only these three
# steer the weighting today; the context deliberately records MORE
# (30-day mean, trends, condition, solar) because history cannot be
# retrofitted — widen this dict when the data earns a new feature.
FEATURE_BANDWIDTHS = {
    "outdoor_temp": 4.0,
    "outdoor_mean_7d": 4.0,
    "daylight_hours": 2.5,
}


def daylight_hours(latitude_deg: float, day: date) -> float:
    """Approximate daylight length in hours (standard declination formula).

    Good to ~15 min at Finnish latitudes; clamped for polar day/night.
    """
    n = day.timetuple().tm_yday
    decl = math.radians(23.45) * math.sin(2.0 * math.pi * (284 + n) / 365.0)
    lat = math.radians(latitude_deg)
    cos_h = -math.tan(lat) * math.tan(decl)
    if cos_h <= -1.0:
        return 24.0
    if cos_h >= 1.0:
        return 0.0
    return 2.0 * math.degrees(math.acos(cos_h)) / 15.0


def similarity(event_context: dict, now_context: dict) -> float | None:
    """Product of per-feature Gaussian kernels; None when the event has
    no usable features (recorded before capture existed)."""
    weight = 1.0
    used = False
    for key, bandwidth in FEATURE_BANDWIDTHS.items():
        a = event_context.get(key)
        b = now_context.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            weight *= math.exp(-(((float(a) - float(b)) / bandwidth) ** 2))
            used = True
    return weight if used else None

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


def _weight(
    event: PreferenceEvent, now: datetime, now_context: dict | None
) -> float:
    age_days = max(0.0, (now - event.when).total_seconds() / 86400.0)
    sim = similarity(event.context, now_context) if now_context else None
    if sim is None:
        # no features on the event (or no live context): age-only with the
        # legacy fast fade, so pre-capture events cannot leak across seasons
        return math.pow(0.5, age_days / LEGACY_HALF_LIFE_DAYS)
    return math.pow(0.5, age_days / HALF_LIFE_DAYS) * sim


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


def _steps_for(event: PreferenceEvent) -> dict[str, float]:
    if event.module == MODULE_CLIMATE:
        return _climate_step(event)
    if event.module == MODULE_ILP:
        return _ilp_step(event)
    if event.module == MODULE_WATER_HEATER:
        return _water_step(event)
    return {}  # log-only modules (climate_hvac)


def _contradiction_counts(
    prepared: list[tuple[PreferenceEvent, dict[str, float]]],
) -> list[float]:
    """Similarity-weighted count of NEWER opposite-direction events per
    event (``prepared`` is chronological — the log appends in time order).
    Same-direction events never discount (confirmation preserves);
    a dissimilar-context contradiction barely counts (summer cannot
    retire winter). Featureless pairs count fully — pre-capture events
    are suspect and should yield quickly. Pairwise per shared parameter:
    fine at the household event rate (dedupe caps density; even a
    hundred events on one key is ~10k kernel evaluations)."""
    counts = [0.0] * len(prepared)
    for i, (older, steps_old) in enumerate(prepared):
        for j in range(i + 1, len(prepared)):
            newer, steps_new = prepared[j]
            opposed = any(
                key in steps_new and steps_new[key] * step < 0
                for key, step in steps_old.items()
            )
            if not opposed:
                continue
            sim = similarity(older.context, newer.context)
            counts[i] += 1.0 if sim is None else sim
    return counts


def derive_adjustments(
    events: list[PreferenceEvent],
    now: datetime,
    now_context: dict | None = None,
) -> dict[str, float]:
    """Fold the event log into bounded parameter offsets (see module
    docstring for the mechanics). ``now_context`` carries the current
    weather-state features; without it, folding is age-only. Keys are
    always all of CAPS, zeros included, so the published `learned`
    attribute has a stable shape."""
    prepared = [
        (event, steps) for event in events if (steps := _steps_for(event))
    ]
    contradictions = _contradiction_counts(prepared)
    sums = {key: 0.0 for key in CAPS}
    for (event, steps), contradicted in zip(prepared, contradictions):
        weight = _weight(event, now, now_context) * math.pow(
            0.5, contradicted / CONTRADICTION_HALF_COUNT
        )
        for key, step in steps.items():
            sums[key] += step * weight
    return {
        key: round(max(-cap, min(cap, sums[key])), 2) for key, cap in CAPS.items()
    }
