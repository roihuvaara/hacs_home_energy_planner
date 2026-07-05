"""Pure climate target computation (ADR 0004/0006 lineage).

Port of the proxy_climate_* automation pipeline. Constants are the live
automation values as of 2026-07-05 (which supersede the older ADR
tables): retuned weather-base bands, always-on warm correction, sun
correction gated on bright forecast hours, and the falling-temperature
lead boost. No Home Assistant imports; unit-testable standalone.
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
    # price mode offsets (c/kWh thresholds are in price_mode())
    offsets: dict[str, float] = field(
        default_factory=lambda: {"setback": -4.0, "normal": 0.0, "preheat": 2.0, "boost": 4.0}
    )
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


@dataclass(frozen=True)
class ClimateInputs:
    hourly_forecast: list[ForecastHour]
    fallback_temp: float
    room_temp: float | None
    current_vat: float
    current_all_in: float
    future_all_in: list[float]
    solar_current_hour_kwh: float
    solar_next_hour_kwh: float
    lead_hold_active: bool


@dataclass(frozen=True)
class ClimateResult:
    target: float
    weather_base: float
    wind_bump: float
    price_mode: str
    price_offset: float
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


def price_mode(
    current_vat: float, current_all_in: float, future_all_in: list[float]
) -> str:
    """Mode from the VAT price gate plus all-in horizon ranking."""
    future = future_all_in[:192]
    if not future:
        return "normal"
    average = sum(future) / len(future)
    ahead_8 = future[1:9]
    ahead_12 = future[1:13]
    min_ahead_8 = min(ahead_8) if ahead_8 else current_all_in
    max_ahead_12 = max(ahead_12) if ahead_12 else current_all_in
    if current_vat < 4:
        return "normal"
    if (
        current_vat <= 4.5
        and average > 0
        and current_all_in <= average * 0.90
        and max_ahead_12 >= current_all_in + 2.0
    ):
        return "boost"
    if (
        average > 0
        and current_vat < 8
        and current_all_in <= average * 0.95
        and max_ahead_12 >= current_all_in + 1.0
    ):
        return "preheat"
    if (
        average > 0
        and current_vat >= 8
        and current_all_in >= average * 1.10
        and min_ahead_8 <= current_all_in - 1.5
    ):
        return "setback"
    return "normal"


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


def compute_climate_target(
    inputs: ClimateInputs, config: ClimateConfig | None = None
) -> ClimateResult:
    config = config or ClimateConfig()
    base, bump = weather_base(inputs.hourly_forecast, inputs.fallback_temp, config)
    mode = price_mode(inputs.current_vat, inputs.current_all_in, inputs.future_all_in)
    offset = config.offsets.get(mode, 0.0)
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
        price_mode=mode,
        price_offset=offset,
        protected_price_offset=protected,
        cold_dip_boost=cold_dip,
        comfort_correction=comfort,
        lead_boost=lead,
        warm_correction=warm,
        sun_correction=sun,
    )
