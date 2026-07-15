"""Pure ILP (air-to-air heat pump) comfort-band recommendation.

Hold the living-room comfort band from both sides when energy is
effectively free or cheap — measured grid export first,
cheap-half-of-horizon second: cool when warm (pre-cool ahead of hot
afternoons on surplus), heat-assist when the room sags under the band
(the slab does the bulk; this trims). Full winter heating dispatch joins
the joint optimizer once the COP curve is identified (see research doc);
this module is also the actuator for those identification experiments.

Comfort thresholds sit just above the climate module's warm-correction
band (warm correction starts at 24.4 C) so the two modules pull the
same direction instead of fighting.

No Home Assistant imports; unit-testable standalone.
"""

from __future__ import annotations

from dataclasses import dataclass

ACTION_COOL = "cool"
ACTION_DRY = "dry"
ACTION_HEAT = "heat"
ACTION_OFF = "off"


@dataclass(frozen=True)
class IlpConfig:
    cool_room_above: float = 24.5  # cool when warm AND energy cheap/free
    cool_room_stop: float = 23.5  # keep cooling until back here
    hard_room_max: float = 25.5  # cool at any price above this
    precool_room_above: float = 23.8  # pre-cool floor (never below stop band)
    cool_target_temp: float = 22.0
    surplus_export_w: float = 500.0
    hot_day_outdoor_max: float = 25.0  # forecast max that marks a hot day
    price_window_quarters: int = 96
    # "cheap" = inside a ranked cheapest window, not merely the cheaper
    # half of the day (the old median split marked ~12 h/day as cheap)
    cheap_quarters: int = 16  # ~4 h opportunistic budget per day
    cheap_min_run_quarters: int = 2  # ILP responds fast; no long min-run
    cheap_margin_cents: float = 1.0
    # humidity comfort (living-room Aqara): cooling already dehumidifies,
    # so dry only runs when temperature does not call for cooling.
    # Band fitted to the household's revealed preference (2026-07-06,
    # 14 days of manual dry usage): maintained means 44-48 %, action
    # taken around 50, post-run minima 42-45 — not the generic 30-60 %
    # comfort guidance.
    dry_humidity_above: float = 50.0  # dry when humid AND energy cheap/free
    dry_humidity_hard: float = 58.0  # dry at any price above this
    dry_humidity_stop: float = 44.0  # keep drying until back here
    # dry mode also cools: never dry a room already at/below the bottom of
    # the 23-23.5 comfort band, not even past the hard humidity limit
    dry_room_floor: float = 23.0
    # heating assist: trim the bottom of the comfort band when energy is
    # cheap/free; the slab does the bulk. Thresholds anchored to the
    # owner band (23-23.5) and protect_below_room lineage (22.8).
    heat_room_min: float = 22.0  # heat at any price below this (comfort net)
    heat_room_below: float = 22.8  # heat when cool AND energy cheap/free
    heat_room_stop: float = 23.3  # keep heating until back here
    heat_target_temp: float = 23.5


@dataclass(frozen=True)
class IlpInputs:
    room_temp: float | None
    room_humidity: float | None
    grid_export_w: float
    future_all_in: list[float]
    outdoor_forecast_max_24h: float | None
    currently_cooling: bool
    currently_drying: bool
    currently_heating: bool = False
    # slab (Versati) COOL regime active: the slow actuator does the bulk,
    # so the ILP only trims peaks (threshold bumped)
    slab_cooling: bool = False


@dataclass(frozen=True)
class IlpResult:
    action: str
    target_temp: float
    reason: str
    actual_surplus: bool
    price_delta: float  # current - median of the coming day


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def compute_ilp_action(
    inputs: IlpInputs, config: IlpConfig | None = None
) -> IlpResult:
    config = config or IlpConfig()

    from .price_windows import plan_cheap_windows

    window = inputs.future_all_in[: config.price_window_quarters]
    median = _median(window) if window else 0.0
    current = window[0] if window else 0.0
    delta = current - median
    surplus = inputs.grid_export_w >= config.surplus_export_w
    windows = (
        plan_cheap_windows(
            window,
            min_run_quarters=config.cheap_min_run_quarters,
            budget_quarters=config.cheap_quarters,
            margin_cents=config.cheap_margin_cents,
        )
        if len(window) >= 8
        else []
    )
    cheap = any(start == 0 for start, _end in windows)
    hot_day = (
        inputs.outdoor_forecast_max_24h is not None
        and inputs.outdoor_forecast_max_24h >= config.hot_day_outdoor_max
    )

    room = inputs.room_temp
    humidity = inputs.room_humidity
    cool_above = config.cool_room_above + (0.5 if inputs.slab_cooling else 0.0)
    action = ACTION_OFF
    reason = "room unknown" if room is None else "comfortable"
    if room is not None:
        if room >= config.hard_room_max:
            action, reason = ACTION_COOL, "room above hard max"
        elif room >= cool_above and (surplus or cheap):
            action, reason = (
                ACTION_COOL,
                "warm room on surplus" if surplus else "warm room in cheap half",
            )
        elif room >= config.precool_room_above and surplus and hot_day:
            action, reason = ACTION_COOL, "pre-cool for hot day on surplus"
        elif inputs.currently_cooling and room > config.cool_room_stop:
            action, reason = ACTION_COOL, "finishing cooling run"

    # heat assist: only when the slab is not in its cooling regime (the
    # two must never fight), and before dry — a cold room heats, not dries
    if action == ACTION_OFF and room is not None and not inputs.slab_cooling:
        if room <= config.heat_room_min:
            action, reason = ACTION_HEAT, "room below hard min"
        elif room <= config.heat_room_below and (surplus or cheap):
            action, reason = (
                ACTION_HEAT,
                "cool room on surplus" if surplus else "cool room in cheap half",
            )
        elif inputs.currently_heating and room < config.heat_room_stop:
            action, reason = ACTION_HEAT, "finishing heat run"

    if action == ACTION_OFF and humidity is not None:
        dry_allowed = room is not None and room >= config.dry_room_floor
        if not dry_allowed:
            if room is not None and humidity >= config.dry_humidity_above:
                reason = "humid but room below dry floor"
        elif humidity >= config.dry_humidity_hard:
            action, reason = ACTION_DRY, "humidity above hard max"
        elif humidity >= config.dry_humidity_above and (surplus or cheap):
            action, reason = (
                ACTION_DRY,
                "humid room on surplus" if surplus else "humid room in cheap half",
            )
        elif inputs.currently_drying and humidity > config.dry_humidity_stop:
            action, reason = ACTION_DRY, "finishing dry run"

    return IlpResult(
        action=action,
        target_temp=config.heat_target_temp
        if action == ACTION_HEAT
        else config.cool_target_temp,
        reason=reason,
        actual_surplus=surplus,
        price_delta=round(delta, 3),
    )
