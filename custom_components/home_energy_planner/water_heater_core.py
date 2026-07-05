"""Pure hot-water control mode computation (ADR 0007 lineage).

Port of proxy_water_heater_control_mode with the live 2026-07-05
constants: live-surplus solar boost, sunny-day buffer preserve, cheap
boost, normal, hold — mapped to Versati tank targets. "Live surplus" is
the Forecast.Solar production estimate, as in the legacy automation.
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
    surplus_solar_w: float = 1800.0
    surplus_solar_full_battery_w: float = 1200.0
    surplus_battery_soc: float = 95.0
    strong_day_remaining_kwh: float = 12.0
    strong_day_next_hour_w: float = 3500.0
    preserve_vat_below: float = 8.0
    cheap_vat_below: float = 4.0
    cheap_vs_average: float = 0.88
    cheap_vs_min_margin: float = 0.75
    normal_vs_average: float = 1.05
    normal_vat_below: float = 8.0


@dataclass(frozen=True)
class WaterHeaterInputs:
    current_vat: float
    current_all_in: float
    future_all_in: list[float]
    solar_now_w: float
    solar_next_hour_w: float
    solar_remaining_today_kwh: float
    battery_soc_pct: float


@dataclass(frozen=True)
class WaterHeaterResult:
    mode: str
    target_temp: int
    actual_surplus: bool
    strong_solar_day: bool
    buffer_preserve: bool


def compute_water_heater_mode(
    inputs: WaterHeaterInputs, config: WaterHeaterConfig | None = None
) -> WaterHeaterResult:
    config = config or WaterHeaterConfig()
    surplus = inputs.solar_now_w >= config.surplus_solar_w or (
        inputs.solar_now_w >= config.surplus_solar_full_battery_w
        and inputs.battery_soc_pct >= config.surplus_battery_soc
    )
    strong_day = (
        inputs.solar_remaining_today_kwh >= config.strong_day_remaining_kwh
        or inputs.solar_next_hour_w >= config.strong_day_next_hour_w
    )
    preserve = (
        strong_day and not surplus and inputs.current_vat < config.preserve_vat_below
    )

    future = inputs.future_all_in[:192]
    count = len(future)
    average = (sum(future) / count) if count else 0.0
    min_future = min(future) if future else inputs.current_all_in

    if surplus:
        mode = MODE_SOLAR_BOOST
    elif preserve:
        mode = MODE_NORMAL
    elif (
        inputs.current_vat < config.cheap_vat_below
        or (
            count > 0
            and average > 0
            and inputs.current_all_in <= average * config.cheap_vs_average
        )
        or (
            count > 0
            and inputs.current_all_in <= min_future + config.cheap_vs_min_margin
        )
    ):
        mode = MODE_CHEAP_BOOST
    elif count > 0 and (
        inputs.current_all_in <= average * config.normal_vs_average
        or inputs.current_vat < config.normal_vat_below
    ):
        mode = MODE_NORMAL
    else:
        mode = MODE_HOLD

    return WaterHeaterResult(
        mode=mode,
        target_temp=config.targets[mode],
        actual_surplus=surplus,
        strong_solar_day=strong_day,
        buffer_preserve=preserve,
    )
