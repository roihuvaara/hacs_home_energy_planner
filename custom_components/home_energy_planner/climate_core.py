"""Pure climate target computation.

The room-feedback layers (positive-only comfort correction, warm and
sun corrections, lead boost, cold-dip boost, weather base) carry over
from the ADR 0004/0006 lineage — they encode household physics: the
fireplace-biased living-room sensor, no controllable covers, observed
overshoot. The price layer is rewritten for the integration (ADR 0009):
instead of the legacy reactive threshold bands with absolute VAT gates,
the offset comes from where the current price sits in the distribution
of the coming horizon, with a spread deadband so uniformly cheap or
expensive days are not price-shaped at all.

No Home Assistant imports; unit-testable standalone.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ForecastHour:
    temperature: float
    wind_gust_speed: float = 0.0
    condition: str = ""
    cloud_coverage: float | None = None


@dataclass(frozen=True)
class ClimateConfig:
    # (average 12 h forecast temperature threshold, base target) descending
    base_bands: tuple[tuple[float, float], ...] = (
        (14.0, 20.0),
        (10.0, 22.0),
        (6.0, 23.0),
        (2.0, 25.0),
        (-2.0, 26.0),
        (-6.0, 28.0),
    )
    base_coldest: float = 30.0
    min_target: float = 20.0
    max_target: float = 35.0
    # wind bump: gust thresholds (km/h) and hour counts over the next 24 h
    windy_gust: float = 35.0
    very_windy_gust: float = 50.0
    extreme_gust: float = 60.0
    windy_hours: int = 6
    very_windy_hours: int = 6
    extreme_hours: int = 4
    # horizon price shaping: offset from the current price's position in
    # the next `price_window_quarters` distribution, +max when far below
    # the median, -max when far above; no shaping below the spread deadband
    price_window_quarters: int = 48  # 12 h
    price_min_spread_cents: float = 3.0  # p90 - p10 deadband
    price_max_offset: float = 4.0
    # comfort floor used to suppress negative price offsets
    protect_below_room: float = 22.8
    room_fallback: float = 23.0
    # positive-only comfort correction bands
    comfort_2_below: float = 23.3
    comfort_1_below: float = 24.1
    # always-on warm correction bands
    warm_2_above: float = 25.0
    warm_1_above: float = 24.4
    # near-term sun correction
    sun_conditions: tuple[str, ...] = ("sunny", "clear-night", "partlycloudy")
    sun_cloud_below: float = 60.0
    sun_2_room_above: float = 23.6
    sun_2_energy_kwh: float = 1.0
    sun_1_room_above: float = 22.79
    sun_1_energy_kwh: float = 0.2
    # short-horizon cold-dip boost (next 6 h minimum)
    cold_dip_2_at_or_below: float = -4.0
    cold_dip_1_at_or_below: float = -1.0
    # lead boost: falling room temperature below the comfort floor
    lead_room_below: float = 23.1
    lead_hold_minutes: int = 90
    # hydronic (slab) cooling: conservative chilled-water targets, always
    # floored by room dew point + margin so the floor surface and the
    # exposed pannuhuone piping stay dry
    cool_water_target: float = 18.0
    cool_water_target_hot: float = 16.0  # allowed on genuinely hot days
    cool_hot_day_outdoor: float = 27.0
    dew_point_margin: float = 2.0
    test_day_outdoor_max: float = 25.0  # notify: good day to trial cooling
    # thermal regime (input routing: slow actuator, slow inputs —
    # direction never reads instantaneous room temperature). Transitions
    # are free; only exit hysteresis keeps a hovering forecast from
    # flapping the regime hourly.
    regime_cool_mean_72h: float = 17.0  # COOL needs a genuinely warm stretch...
    regime_cool_max_24h: float = 25.0  # ...with a hot day ahead
    regime_room_mean_heat_below: float = 23.2  # backup net under the 23-23.5 band
    regime_exit_forecast_margin: float = 1.0  # widen own entry band while inside
    regime_exit_room_margin: float = 0.2
    # within COOL, when cooling actually runs: ranked cheapest quarters
    # of the price horizon (effective price = all-in + COP penalty for
    # warm outdoor air) or measured solar surplus. Budget is a fraction
    # of the horizon — 0.5 keeps the duty the shipped 21-09 night block
    # had; placement is what the ranking optimizes.
    cool_budget_fraction: float = 0.5
    cool_min_run_quarters: int = 4  # slab compressor: no sub-hour cycling
    # cents/kWh added per °C of forecast outdoor temperature: rejecting
    # heat into cool air buys more cooling per kWh, so warm quarters must
    # be cheaper on the meter to rank equal. Only relative differences
    # matter to the ranking. 0.3 c/°C makes the observed ~10 °C night-day
    # swing weigh about one p10-p90 price spread (~3 c, the shaping
    # deadband above) — night wins by default, but a genuinely cheap
    # afternoon (windy/negative prices) can still take the slot. To be
    # recalibrated from plan-vs-reality data once cooling COP is
    # identified.
    cool_cop_penalty_c_per_deg: float = 0.3
    # fallback block when the price horizon is missing: degrade to the
    # shipped behaviour (cheap, high-COP night air), never to no cooling
    cool_night_start_hour: int = 21
    cool_night_end_hour: int = 9
    cool_surplus_export_w: float = 500.0


@dataclass(frozen=True)
class ClimateInputs:
    hourly_forecast: list[ForecastHour]
    fallback_temp: float
    room_temp: float | None
    future_all_in: list[float]
    solar_current_hour_kwh: float
    solar_next_hour_kwh: float
    lead_hold_active: bool


@dataclass(frozen=True)
class PriceShape:
    offset: float
    current: float
    p10: float
    median: float
    p90: float
    spread: float
    position: float  # (current - median) / spread; negative = cheap side


@dataclass(frozen=True)
class ClimateResult:
    target: float
    weather_base: float
    wind_bump: float
    price: PriceShape
    protected_price_offset: float
    cold_dip_boost: float
    comfort_correction: float
    lead_boost: float
    warm_correction: float
    sun_correction: float


def weather_base(
    hours: list[ForecastHour], fallback_temp: float, config: ClimateConfig
) -> tuple[float, float]:
    """Steady-state base from the 12 h average, wind bump from 24 h gusts."""
    hours24 = hours[:24]
    hours12 = hours24[:12]
    avg = (
        sum(h.temperature for h in hours12) / len(hours12)
        if hours12
        else fallback_temp
    )
    base = config.base_coldest
    for threshold, value in config.base_bands:
        if avg >= threshold:
            base = value
            break
    windy = sum(1 for h in hours24 if h.wind_gust_speed >= config.windy_gust)
    very = sum(1 for h in hours24 if h.wind_gust_speed >= config.very_windy_gust)
    extreme = sum(1 for h in hours24 if h.wind_gust_speed >= config.extreme_gust)
    bump = (
        3.0
        if extreme >= config.extreme_hours
        else 2.0
        if very >= config.very_windy_hours
        else 1.0
        if windy >= config.windy_hours
        else 0.0
    )
    return min(config.max_target, base + bump), bump


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = fraction * (len(sorted_values) - 1)
    low = int(index)
    high = min(low + 1, len(sorted_values) - 1)
    weight = index - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def price_shape(future_all_in: list[float], config: ClimateConfig) -> PriceShape:
    """Offset from the current price's position in the coming distribution.

    Cheap relative to the window -> bank heat into the house mass (+),
    expensive -> coast on it (-). A narrow p10-p90 spread means there is
    nothing worth shifting, so the offset stays 0 however cheap or
    expensive the absolute level is.
    """
    window = future_all_in[: config.price_window_quarters]
    if len(window) < 8:
        return PriceShape(0.0, window[0] if window else 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    current = window[0]
    ordered = sorted(window)
    p10 = percentile(ordered, 0.10)
    median = percentile(ordered, 0.50)
    p90 = percentile(ordered, 0.90)
    spread = p90 - p10
    if spread < config.price_min_spread_cents:
        return PriceShape(0.0, current, p10, median, p90, spread, 0.0)
    position = (current - median) / spread
    offset = max(
        -config.price_max_offset,
        min(config.price_max_offset, -round(position * 2 * config.price_max_offset)),
    )
    return PriceShape(float(offset), current, p10, median, p90, spread, round(position, 3))


def comfort_correction(room_temp: float | None, config: ClimateConfig) -> float:
    if room_temp is None:
        return 0.0
    if room_temp < config.comfort_2_below:
        return 2.0
    if room_temp < config.comfort_1_below:
        return 1.0
    return 0.0


def warm_correction(room_temp: float | None, config: ClimateConfig) -> float:
    if room_temp is None:
        return 0.0
    if room_temp > config.warm_2_above:
        return 2.0
    if room_temp > config.warm_1_above:
        return 1.0
    return 0.0


def sun_correction(
    room_temp: float | None,
    next_two_hours: list[ForecastHour],
    energy_current_hour_kwh: float,
    energy_next_hour_kwh: float,
    config: ClimateConfig,
) -> float:
    if room_temp is None:
        return 0.0
    eligible = sum(
        1
        for hour in next_two_hours[:2]
        if hour.condition in config.sun_conditions
        and (hour.cloud_coverage is None or hour.cloud_coverage < config.sun_cloud_below)
    )
    if eligible == 0:
        return 0.0
    if room_temp > config.sun_2_room_above and (
        energy_current_hour_kwh > config.sun_2_energy_kwh
        or energy_next_hour_kwh > config.sun_2_energy_kwh
    ):
        return 2.0
    if room_temp > config.sun_1_room_above and (
        energy_current_hour_kwh > config.sun_1_energy_kwh
        or energy_next_hour_kwh > config.sun_1_energy_kwh
    ):
        return 1.0
    return 0.0


def cold_dip_boost(
    hours: list[ForecastHour], fallback_temp: float, config: ClimateConfig
) -> float:
    hours6 = hours[:6]
    minimum = min((h.temperature for h in hours6), default=fallback_temp)
    if minimum <= config.cold_dip_2_at_or_below:
        return 2.0
    if minimum <= config.cold_dip_1_at_or_below:
        return 1.0
    return 0.0


def lead_boost(
    room_temp: float | None, hold_active: bool, config: ClimateConfig
) -> float:
    if room_temp is None or not hold_active:
        return 0.0
    return 1.0 if room_temp < config.lead_room_below else 0.0


REGIME_HEAT = "heat"
REGIME_NEUTRAL = "neutral"
REGIME_COOL = "cool"


def decide_regime(
    current: str | None,
    forecast_avg_12h: float | None,
    forecast_mean_72h: float | None,
    forecast_max_24h: float | None,
    room_mean_24h: float | None,
    config: ClimateConfig | None = None,
) -> tuple[str, str]:
    """Thermal regime for the Versati space circuit.

    Deliberately takes no instantaneous room temperature: the slab is a
    slow actuator, so direction is decided by forecasts (predictive) and
    at most the 24 h room *mean* as a backup net. Conditions rule
    directly — any transition, any time — so a cold snap can never be
    parked behind a dwell. The only memory is exit hysteresis: the
    regime we are in widens its own entry band by a margin, so a
    forecast hovering at a threshold cannot flap the regime hourly.
    """
    config = config or ClimateConfig()

    # predictive heat demand: the weather-base table asks for more than
    # its mildest band exactly when the 12 h forecast avg drops below
    # the first band threshold — the already-tuned heating signal
    in_heat = current == REGIME_HEAT
    heat_needed = (
        forecast_avg_12h is not None
        and forecast_avg_12h
        < config.base_bands[0][0]
        + (config.regime_exit_forecast_margin if in_heat else 0.0)
    ) or (
        room_mean_24h is not None
        and room_mean_24h
        < config.regime_room_mean_heat_below
        + (config.regime_exit_room_margin if in_heat else 0.0)
    )
    cool_margin = config.regime_exit_forecast_margin if current == REGIME_COOL else 0.0
    cool_wanted = (
        not heat_needed
        and forecast_mean_72h is not None
        and forecast_mean_72h >= config.regime_cool_mean_72h - cool_margin
        and forecast_max_24h is not None
        and forecast_max_24h >= config.regime_cool_max_24h - cool_margin
    )
    target = (
        REGIME_HEAT if heat_needed else REGIME_COOL if cool_wanted else REGIME_NEUTRAL
    )

    if current is None:
        return target, f"initial: {target}"
    if target == current:
        return current, f"stay {current}"
    return target, f"{current}->{target}"


def plan_cool_windows(
    future_all_in: list[float],
    hourly_forecast: list[ForecastHour],
    config: ClimateConfig | None = None,
) -> list[tuple[int, int]]:
    """Ranked cheapest [start, end) quarter windows for slab cooling.

    Quarters are ranked by effective price — all-in cents plus the COP
    penalty for the forecast outdoor temperature of that quarter's hour —
    and the cheapest non-overlapping min-run windows are picked until the
    budget fraction of the horizon is placed. No qualification gate: in
    the COOL regime the slab must get its hours; only placement moves.
    """
    config = config or ClimateConfig()
    quarters = len(future_all_in)
    if quarters < config.cool_min_run_quarters:
        return []
    effective = list(future_all_in)
    if hourly_forecast:
        last = len(hourly_forecast) - 1
        effective = [
            price
            + config.cool_cop_penalty_c_per_deg
            * hourly_forecast[min(i // 4, last)].temperature
            for i, price in enumerate(future_all_in)
        ]
    from .price_windows import plan_budget_windows

    return plan_budget_windows(
        effective,
        min_run_quarters=config.cool_min_run_quarters,
        budget_quarters=int(quarters * config.cool_budget_fraction),
    )


def cool_active_now(
    local_hour: int,
    grid_export_w: float,
    config: ClimateConfig | None = None,
    *,
    cool_windows: list[tuple[int, int]] | None = None,
) -> bool:
    """Within COOL regime: in a ranked cheap window or on measured surplus.

    cool_windows=None means no usable price horizon — fall back to the
    fixed night block so the planner degrades to the pre-price behaviour
    rather than to no cooling at all.
    """
    config = config or ClimateConfig()
    if grid_export_w >= config.cool_surplus_export_w:
        return True
    if cool_windows is not None:
        return any(start == 0 for start, _end in cool_windows)
    start, end = config.cool_night_start_hour, config.cool_night_end_hour
    return (local_hour >= start or local_hour < end) if start > end else (
        start <= local_hour < end
    )


def dew_point_c(temp_c: float, rh_pct: float) -> float:
    """Magnus approximation; the condensation limit for chilled surfaces."""
    import math

    gamma = math.log(max(1.0, min(100.0, rh_pct)) / 100.0) + (
        17.62 * temp_c / (243.12 + temp_c)
    )
    return 243.12 * gamma / (17.62 - gamma)


def cool_water_target(
    room_temp: float | None,
    room_humidity: float | None,
    outdoor_forecast_max: float | None,
    config: ClimateConfig | None = None,
) -> tuple[float, float | None]:
    """Chilled-water target and the dew point it was floored against.

    Hot days may use the lower target, but the dew-point floor always
    wins: drier air permits colder water, humid air forbids it.
    """
    config = config or ClimateConfig()
    target = (
        config.cool_water_target_hot
        if outdoor_forecast_max is not None
        and outdoor_forecast_max >= config.cool_hot_day_outdoor
        else config.cool_water_target
    )
    dew = None
    if room_temp is not None and room_humidity is not None:
        dew = round(dew_point_c(room_temp, room_humidity), 1)
        target = max(target, dew + config.dew_point_margin)
    return round(target, 1), dew


def project_targets(
    inputs: ClimateInputs,
    config: ClimateConfig | None = None,
    extra_offset: float = 0.0,
) -> list[float]:
    """Projected target per future quarter over the price horizon.

    The forecast-driven layers (weather base, wind bump, cold dip) and
    the price shape are recomputed at each future quarter from the same
    inputs the live computation uses, just shifted forward. The
    room-feedback layers (comfort, lead, warm, sun) react to a room
    temperature that cannot be forecast, so they are held at their
    current values: the projection matches the live target at quarter 0
    and converges to the pure weather+price target as those corrections
    age out of relevance further along the horizon.
    """
    config = config or ClimateConfig()
    room = inputs.room_temp if inputs.room_temp is not None else config.room_fallback
    comfort = comfort_correction(inputs.room_temp, config)
    lead = lead_boost(inputs.room_temp, inputs.lead_hold_active, config)
    warm = warm_correction(inputs.room_temp, config)
    sun = sun_correction(
        inputs.room_temp,
        inputs.hourly_forecast[:2],
        inputs.solar_current_hour_kwh,
        inputs.solar_next_hour_kwh,
        config,
    )
    held = comfort + lead - warm - sun + extra_offset
    targets: list[float] = []
    for quarter in range(len(inputs.future_all_in)):
        hours = inputs.hourly_forecast[quarter // 4 :]
        base, _bump = weather_base(hours, inputs.fallback_temp, config)
        shape = price_shape(inputs.future_all_in[quarter:], config)
        offset = shape.offset
        protected = 0.0 if room < config.protect_below_room and offset < 0 else offset
        cold_dip = cold_dip_boost(hours, inputs.fallback_temp, config)
        target = min(
            config.max_target,
            max(config.min_target, base + protected + cold_dip + held),
        )
        targets.append(round(target, 1))
    return targets


def compute_climate_target(
    inputs: ClimateInputs, config: ClimateConfig | None = None
) -> ClimateResult:
    config = config or ClimateConfig()
    base, bump = weather_base(inputs.hourly_forecast, inputs.fallback_temp, config)
    shape = price_shape(inputs.future_all_in, config)
    offset = shape.offset
    room = inputs.room_temp if inputs.room_temp is not None else config.room_fallback
    protected = 0.0 if room < config.protect_below_room and offset < 0 else offset
    cold_dip = cold_dip_boost(inputs.hourly_forecast, inputs.fallback_temp, config)
    comfort = comfort_correction(inputs.room_temp, config)
    lead = lead_boost(inputs.room_temp, inputs.lead_hold_active, config)
    warm = warm_correction(inputs.room_temp, config)
    sun = sun_correction(
        inputs.room_temp,
        inputs.hourly_forecast[:2],
        inputs.solar_current_hour_kwh,
        inputs.solar_next_hour_kwh,
        config,
    )
    target = min(
        config.max_target,
        max(config.min_target, base + protected + cold_dip + comfort + lead - warm - sun),
    )
    return ClimateResult(
        target=round(target, 1),
        weather_base=base,
        wind_bump=bump,
        price=shape,
        protected_price_offset=protected,
        cold_dip_boost=cold_dip,
        comfort_correction=comfort,
        lead_boost=lead,
        warm_correction=warm,
        sun_correction=sun,
    )
