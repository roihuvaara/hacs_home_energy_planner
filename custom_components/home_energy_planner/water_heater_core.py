"""Pure hot-water control mode computation.

Rewritten for the integration (ADR 0009) rather than ported: the tank is
a thermal battery scheduled against the actual price horizon instead of
the legacy absolute-threshold rules.

- solar_boost on **measured** grid export (the house is provably
  exporting, so tank heating consumes otherwise-cheap surplus) — the
  legacy stack had to trust the Forecast.Solar estimate
- buffer preserve (ADR 0007's insight, kept): a strong upcoming solar
  forecast suppresses early grid boosting so the tank has headroom when
  the surplus actually arrives
- cheap_boost is scheduled as contiguous windows, never single quarters:
  the compressor should run for a stretch, so boosting follows the
  cheapest minimum-length windows of the coming day (by window average,
  meaningfully below the day median), not scattered cheap quarters.
  Hold triggers when the current price is meaningfully above the median;
  flat days just run normal.

Mode targets (device mapping, unchanged): 66/60/55/51 C.
No Home Assistant imports; unit-testable standalone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MODE_SOLAR_BOOST = "solar_boost"
MODE_CHEAP_BOOST = "cheap_boost"
MODE_NORMAL = "normal"
MODE_HOLD = "hold"


@dataclass(frozen=True)
class WaterHeaterConfig:
    targets: dict[str, int] = field(
        default_factory=lambda: {
            MODE_SOLAR_BOOST: 66,
            MODE_CHEAP_BOOST: 60,
            MODE_NORMAL: 55,
            MODE_HOLD: 51,
        }
    )
    # measured export above this means real surplus is flowing out
    surplus_export_w: float = 500.0
    # preserve tank headroom when this much solar is forecast soon
    preserve_upcoming_solar_kwh: float = 6.0
    # schedule grid boosting into the cheapest contiguous windows of the
    # coming day. Measured on the Versati (2026-07-05 run): ~19 min
    # startup transient with zero tank gain, then ~0.23 C/min, so a
    # 5 C cheap boost physically takes ~41 min — windows shorter than
    # that mostly buy the transient. The exact short-run tradeoff is a
    # per-start cost in the planned MILP formulation, not a rule here.
    price_window_quarters: int = 96  # 24 h
    cheap_quarters: int = 8  # ~2 h boosting budget per day
    min_run_quarters: int = 3  # 45 min >= measured 41 min 5 C boost run
    cheap_margin_cents: float = 2.0  # window mean this far below the median
    hold_margin_cents: float = 4.0  # current this far above the median


@dataclass(frozen=True)
class WaterHeaterInputs:
    future_all_in: list[float]
    grid_export_w: float
    upcoming_solar_kwh: float  # forecast over the next ~6 h


@dataclass(frozen=True)
class WaterHeaterResult:
    mode: str
    target_temp: int
    actual_surplus: bool
    buffer_preserve: bool
    cheap_windows: list[tuple[int, int]]  # [start, end) quarter indices
    price_median: float
    price_delta: float  # current - median


def _median(ordered: list[float]) -> float:
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def plan_cheap_windows(
    prices: list[float], config: WaterHeaterConfig
) -> list[tuple[int, int]]:
    """Cheapest non-overlapping min-run windows within the boosting budget.

    Greedy by window average price; a window qualifies only when its
    average sits `cheap_margin_cents` below the day median, so a flat
    day yields no windows at all.
    """
    run = config.min_run_quarters
    if len(prices) < run:
        return []
    median = _median(sorted(prices))
    means = [
        (sum(prices[i : i + run]) / run, i) for i in range(len(prices) - run + 1)
    ]
    chosen: list[tuple[int, int]] = []
    budget = config.cheap_quarters
    for mean, start in sorted(means):
        if budget < run or mean > median - config.cheap_margin_cents:
            break
        end = start + run
        if any(start < c_end and c_start < end for c_start, c_end in chosen):
            continue
        chosen.append((start, end))
        budget -= run
    # merge adjacent picks into longer runs
    chosen.sort()
    merged: list[tuple[int, int]] = []
    for start, end in chosen:
        if merged and start == merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def compute_water_heater_mode(
    inputs: WaterHeaterInputs, config: WaterHeaterConfig | None = None
) -> WaterHeaterResult:
    config = config or WaterHeaterConfig()

    surplus = inputs.grid_export_w >= config.surplus_export_w
    preserve = (
        inputs.upcoming_solar_kwh >= config.preserve_upcoming_solar_kwh
        and not surplus
    )

    window = inputs.future_all_in[: config.price_window_quarters]
    median = _median(sorted(window)) if window else 0.0
    current = window[0] if window else 0.0
    delta = current - median
    usable = len(window) >= 8
    windows = plan_cheap_windows(window, config) if usable else []
    cheap_now = any(start == 0 for start, _end in windows)
    expensive_now = usable and delta >= config.hold_margin_cents

    if surplus:
        mode = MODE_SOLAR_BOOST
    elif preserve:
        mode = MODE_NORMAL
    elif cheap_now:
        mode = MODE_CHEAP_BOOST
    elif expensive_now:
        mode = MODE_HOLD
    else:
        mode = MODE_NORMAL

    return WaterHeaterResult(
        mode=mode,
        target_temp=config.targets[mode],
        actual_surplus=surplus,
        buffer_preserve=preserve,
        cheap_windows=windows,
        price_median=round(median, 3),
        price_delta=round(delta, 3),
    )
