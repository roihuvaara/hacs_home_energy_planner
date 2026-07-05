"""Pure ILP (air-to-air heat pump) cooling recommendation.

Summer v1 of the ILP asset: hold the living-room comfort band by
cooling when the room is warm and energy is effectively free or cheap —
measured grid export first, cheap-half-of-horizon second — and pre-cool
ahead of hot afternoons on surplus. Winter heating assistance joins the
joint optimizer once the COP curve is identified (see research doc);
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
    # humidity comfort (living-room Aqara): cooling already dehumidifies,
    # so dry only runs when temperature does not call for cooling
    dry_humidity_above: float = 62.0  # dry when humid AND energy cheap/free
    dry_humidity_hard: float = 70.0  # dry at any price above this
    dry_humidity_stop: float = 55.0  # keep drying until back here


@dataclass(frozen=True)
class IlpInputs:
    room_temp: float | None
    room_humidity: float | None
    grid_export_w: float
    future_all_in: list[float]
    outdoor_forecast_max_24h: float | None
    currently_cooling: bool
    currently_drying: bool


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

    window = inputs.future_all_in[: config.price_window_quarters]
    median = _median(window) if window else 0.0
    current = window[0] if window else 0.0
    delta = current - median
    surplus = inputs.grid_export_w >= config.surplus_export_w
    cheap = bool(window) and delta <= 0.0
    hot_day = (
        inputs.outdoor_forecast_max_24h is not None
        and inputs.outdoor_forecast_max_24h >= config.hot_day_outdoor_max
    )

    room = inputs.room_temp
    humidity = inputs.room_humidity
    action = ACTION_OFF
    reason = "room unknown" if room is None else "comfortable"
    if room is not None:
        if room >= config.hard_room_max:
            action, reason = ACTION_COOL, "room above hard max"
        elif room >= config.cool_room_above and (surplus or cheap):
            action, reason = (
                ACTION_COOL,
                "warm room on surplus" if surplus else "warm room in cheap half",
            )
        elif room >= config.precool_room_above and surplus and hot_day:
            action, reason = ACTION_COOL, "pre-cool for hot day on surplus"
        elif inputs.currently_cooling and room > config.cool_room_stop:
            action, reason = ACTION_COOL, "finishing cooling run"

    if action == ACTION_OFF and humidity is not None:
        if humidity >= config.dry_humidity_hard:
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
        target_temp=config.cool_target_temp,
        reason=reason,
        actual_surplus=surplus,
        price_delta=round(delta, 3),
    )
