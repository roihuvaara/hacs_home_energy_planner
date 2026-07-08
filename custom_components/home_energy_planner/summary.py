"""Human-facing summary: turn planner internals into what a person reads.

The module sensors expose optimizer state (modes, targets, price deltas,
override windows). A person living in the house cares about five things:
is now a good time to use power, will my stuff be ready, what is the
system doing, is it actually in control, and how did the day go. This
pure module maps the internals to those answers; the sensor assembles the
inputs and the dashboard renders them. No Home Assistant imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

# stance thresholds: where the current price sits within the horizon
CHEAP_RANK = 0.30
EXPENSIVE_RANK = 0.70
# a day whose peak-to-trough spread is under this is "flat": timing buys
# nothing, so neither cheap nor expensive regardless of the rank
FLAT_SPREAD_CENTS = 1.0
WINDOW_QUARTERS = 4  # 1 h — a meaningful "cheap window to run something"


def percentile_rank(values: list[float], current: float) -> float:
    """Fraction of the horizon at or below the current price (0 = cheapest)."""
    if not values:
        return 0.0
    return sum(1 for v in values if v <= current) / len(values)


def price_stance(horizon: list[float]) -> dict:
    if not horizon:
        return {"stance": "unknown", "rank_pct": None, "cheap_now": False, "price_now": None}
    current = horizon[0]
    rank = percentile_rank(horizon, current)
    if max(horizon) - min(horizon) < FLAT_SPREAD_CENTS:
        stance = "normal"  # flat day: timing buys nothing
    elif rank <= CHEAP_RANK:
        stance = "cheap"
    elif rank >= EXPENSIVE_RANK:
        stance = "expensive"
    else:
        stance = "normal"
    return {
        "stance": stance,
        "rank_pct": round(rank * 100),
        "cheap_now": stance == "cheap",
        "price_now": round(current, 2),
    }


def extreme_window(horizon: list[float], run: int, *, cheapest: bool) -> dict | None:
    """(start,end index, mean) of the cheapest/most-expensive run in the horizon."""
    if len(horizon) < run:
        return None
    means = [(sum(horizon[i : i + run]) / run, i) for i in range(len(horizon) - run + 1)]
    mean, start = (min if cheapest else max)(means)
    return {"start_index": start, "end_index": start + run, "mean": round(mean, 2)}


def _hhmm(start: datetime, period_minutes: int, index: int) -> str:
    return (start + timedelta(minutes=period_minutes * index)).strftime("%H:%M")


def _window_times(win: dict | None, start: datetime, period_minutes: int) -> dict | None:
    if win is None:
        return None
    return {
        "start": _hhmm(start, period_minutes, win["start_index"]),
        "end": _hhmm(start, period_minutes, win["end_index"]),
        "mean": win["mean"],
    }


def battery_text(soc: float | None, charging_now: bool, discharging_now: bool) -> str:
    if charging_now:
        verb = "charging"
    elif discharging_now:
        verb = "covering the load"
    else:
        verb = "holding"
    return f"{verb} · {round(soc)}%" if soc is not None else verb


def water_text(mode: str | None, tank_temp: float | None, override_hhmm: str | None) -> str:
    base = {
        "solar_boost": "heating on solar surplus",
        "cheap_boost": "heating — cheap window",
        "hold": "resting — price high",
        "normal": "ready" if (tank_temp is None or tank_temp >= 50) else "reheating",
    }.get(mode or "", mode or "unknown")
    temp = f" · {round(tank_temp)}°C" if tank_temp is not None else ""
    override = f" · manual until {override_hhmm}" if override_hhmm else ""
    return f"{base}{temp}{override}"


def climate_text(regime: str | None, target: float | None, room_temp: float | None) -> str:
    base = {
        "heat": f"heating — setpoint {round(target, 1)}°C" if target is not None else "heating",
        "cool": "cooling",
        "neutral": "idle — house comfortable",
    }.get(regime or "", "idle")
    room = f" · room {round(room_temp, 1)}°C" if room_temp is not None else ""
    return f"{base}{room}"


def ilp_text(action: str | None, reason: str | None) -> str:
    verb = {"off": "off", "cool": "cooling", "dry": "drying", "heat": "heating"}.get(
        action or "", action or "off"
    )
    return f"{verb} — {reason}" if reason else verb


@dataclass
class SummaryInputs:
    horizon: list[float]
    horizon_start: datetime | None
    period_minutes: int = 15
    soc_pct: float | None = None
    battery_charging_now: bool = False
    battery_discharging_now: bool = False
    water_mode: str | None = None
    tank_temp: float | None = None
    water_override_until: datetime | None = None
    climate_regime: str | None = None
    climate_target: float | None = None
    room_temp: float | None = None
    ilp_action: str | None = None
    ilp_reason: str | None = None
    issues: list[str] = field(default_factory=list)  # attention-level, formatted
    info: list[str] = field(default_factory=list)  # info-level, formatted


def build_summary(inp: SummaryInputs) -> dict:
    price = price_stance(inp.horizon)
    start = inp.horizon_start
    cheap = None
    peak = None
    if start is not None:
        cheap = _window_times(
            extreme_window(inp.horizon, WINDOW_QUARTERS, cheapest=True),
            start,
            inp.period_minutes,
        )
        peak = _window_times(
            extreme_window(inp.horizon, WINDOW_QUARTERS, cheapest=False),
            start,
            inp.period_minutes,
        )

    override_hhmm = (
        inp.water_override_until.strftime("%H:%M") if inp.water_override_until else None
    )
    assets = [
        {"asset": "Battery", "text": battery_text(
            inp.soc_pct, inp.battery_charging_now, inp.battery_discharging_now)},
        {"asset": "Hot water", "text": water_text(
            inp.water_mode, inp.tank_temp, override_hhmm)},
        {"asset": "Heat pump", "text": climate_text(
            inp.climate_regime, inp.climate_target, inp.room_temp)},
        {"asset": "Air-air (ILP)", "text": ilp_text(inp.ilp_action, inp.ilp_reason)},
    ]

    coming: list[str] = []
    if price["cheap_now"]:
        coming.append("cheap power now")
    elif cheap is not None:
        coming.append(f"cheapest {cheap['start']}–{cheap['end']}")
    if peak is not None and price["stance"] != "expensive":
        coming.append(f"peak to avoid {peak['start']}–{peak['end']}")

    status = "attention" if inp.issues else "ok"
    return {
        "status": status,
        "headline": "Attention" if inp.issues else "In control",
        "price": price,
        "next_cheap_window": cheap,
        "peak_window": peak,
        "assets": assets,
        "coming_up": " · ".join(coming) if coming else "steady",
        "issues": list(inp.issues),
        "info": list(inp.info),
    }
