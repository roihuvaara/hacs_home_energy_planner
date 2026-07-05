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
    cool_room_above: float = 24.5  # switch to cooling when room is warm
    cool_room_stop: float = 23.5  # back to heating mode below this
    test_day_outdoor_max: float = 25.0  # notify: good day to trial cooling


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
