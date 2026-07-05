"""Pure battery dispatch planning: problem model, solver, slot compilation.

The solver is an exact dynamic program over quantized battery energy
states. For this convex problem size (~140 quarter-hour periods, ~50
states) it returns the same dispatch an LP would; it exists behind
``solve()`` so an LP/MILP engine can replace it without touching callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from .solis_slots import SlotSpec, find_cross_side_overlaps

PERIOD_MINUTES = 15
PERIOD_HOURS = PERIOD_MINUTES / 60.0
STATE_STEP_KWH = 0.1
BATTERY_NOMINAL_VOLTAGE = 50.0
ROUND_TRIP_EFFICIENCY = 0.9
CHARGE_EFF = ROUND_TRIP_EFFICIENCY**0.5
DISCHARGE_EFF = ROUND_TRIP_EFFICIENCY**0.5
CYCLE_COST_CENTS_PER_KWH = 4.0

Action = Literal["charge", "hold", "self_use"]


@dataclass(frozen=True)
class Period:
    start: datetime
    price_cents_per_kwh: float
    load_kwh: float
    solar_kwh: float


@dataclass(frozen=True)
class BatteryParams:
    capacity_kwh: float
    state_of_health_pct: float
    soc_pct: float
    reserve_soc_pct: float
    max_charge_current: int
    max_discharge_current: int
    planned_charge_current: int = 12

    @property
    def effective_capacity_kwh(self) -> float:
        return self.capacity_kwh * self.state_of_health_pct / 100.0

    @property
    def usable_above_reserve_kwh(self) -> float:
        return max(
            0.0,
            self.effective_capacity_kwh * (100.0 - self.reserve_soc_pct) / 100.0,
        )

    def buffer_kwh_from_soc(self, soc_pct: float) -> float:
        return max(
            0.0,
            self.effective_capacity_kwh
            * (soc_pct - self.reserve_soc_pct)
            / 100.0,
        )

    def soc_from_buffer_kwh(self, buffer_kwh: float) -> float:
        if self.effective_capacity_kwh <= 0:
            return self.reserve_soc_pct
        return min(
            100.0,
            self.reserve_soc_pct + buffer_kwh / self.effective_capacity_kwh * 100.0,
        )


@dataclass(frozen=True)
class PeriodPlan:
    start: datetime
    action: Action
    buffer_start_kwh: float
    buffer_end_kwh: float
    grid_charge_kwh: float
    discharge_to_load_kwh: float
    grid_import_kwh: float
    price_cents_per_kwh: float


@dataclass(frozen=True)
class DispatchPlan:
    periods: list[PeriodPlan]
    total_cost_cents: float
    baseline_cost_cents: float
    end_soc_pct: float


def current_to_period_kwh(current_a: int) -> float:
    return max(0.0, current_a * BATTERY_NOMINAL_VOLTAGE * PERIOD_HOURS / 1000.0)


def solve(periods: list[Period], battery: BatteryParams) -> DispatchPlan:
    """Minimize import cost + cycle cost over the horizon."""

    capacity = battery.usable_above_reserve_kwh
    max_units = max(1, int(round(capacity / STATE_STEP_KWH)))
    start_units = min(
        max_units,
        int(round(battery.buffer_kwh_from_soc(battery.soc_pct) / STATE_STEP_KWH)),
    )
    charge_step = current_to_period_kwh(
        min(battery.planned_charge_current, battery.max_charge_current)
    )
    discharge_step = current_to_period_kwh(battery.max_discharge_current)
    charge_units = int(charge_step / STATE_STEP_KWH)
    discharge_units = int(discharge_step / STATE_STEP_KWH)

    n = len(periods)
    inf = float("inf")
    # Terminal value: leftover buffer avoids future imports at least at the
    # horizon's cheapest price. Without it the solver dumps the battery at
    # any price above cycle cost near the horizon end; anything above the
    # minimum price would instead reward charging purely to bank credit.
    min_price = min((p.price_cents_per_kwh for p in periods), default=0.0)
    terminal_value = max(0.0, DISCHARGE_EFF * min_price)
    # dp[t][s]: minimal future cost from period t at buffer state s
    dp = [[inf] * (max_units + 1) for _ in range(n)] + [
        [-s * STATE_STEP_KWH * terminal_value for s in range(max_units + 1)]
    ]
    choice = [[("self_use", 0)] * (max_units + 1) for _ in range(n)]

    for t in range(n - 1, -1, -1):
        period = periods[t]
        net_load = max(0.0, period.load_kwh - period.solar_kwh)
        surplus = max(0.0, period.solar_kwh - period.load_kwh)
        surplus_units = int(round(surplus * CHARGE_EFF / STATE_STEP_KWH))
        for s in range(max_units + 1):
            best = inf
            best_choice = ("self_use", s)

            # hold: no battery contribution, passive solar surplus capture
            s_hold = min(max_units, s + surplus_units)
            cost_hold = net_load * period.price_cents_per_kwh + dp[t + 1][s_hold]
            if cost_hold < best:
                best, best_choice = cost_hold, ("hold", s_hold)

            # self_use: discharge toward net load
            usable = min(s, discharge_units)
            deliver = min(usable * STATE_STEP_KWH * DISCHARGE_EFF, net_load)
            drain_units = int(round(deliver / DISCHARGE_EFF / STATE_STEP_KWH))
            s_use = min(max_units, s - drain_units + surplus_units)
            cost_use = (
                (net_load - deliver) * period.price_cents_per_kwh
                + deliver * CYCLE_COST_CENTS_PER_KWH
                + dp[t + 1][s_use]
            )
            if drain_units > 0 and cost_use < best - 1e-9:
                best, best_choice = cost_use, ("self_use", s_use)

            # charge: grid-charge up to the per-period limit
            for add in range(1, charge_units + 1):
                s_charge = s + surplus_units + add
                if s_charge > max_units:
                    break
                grid_kwh = add * STATE_STEP_KWH / CHARGE_EFF
                cost_charge = (
                    (net_load + grid_kwh) * period.price_cents_per_kwh
                    + dp[t + 1][s_charge]
                )
                if cost_charge < best - 1e-9:
                    best, best_choice = cost_charge, ("charge", s_charge)

            dp[t][s] = best
            choice[t][s] = best_choice

    # forward pass
    plans: list[PeriodPlan] = []
    baseline = 0.0
    total = 0.0
    s = start_units
    for t, period in enumerate(periods):
        net_load = max(0.0, period.load_kwh - period.solar_kwh)
        baseline += net_load * period.price_cents_per_kwh
        action, s_next = choice[t][s]
        surplus = max(0.0, period.solar_kwh - period.load_kwh)
        surplus_units = int(round(surplus * CHARGE_EFF / STATE_STEP_KWH))
        grid_charge = 0.0
        deliver = 0.0
        if action == "charge":
            add = s_next - s - surplus_units
            grid_charge = max(0, add) * STATE_STEP_KWH / CHARGE_EFF
        elif action == "self_use":
            drain = s + surplus_units - s_next
            deliver = max(0, drain) * STATE_STEP_KWH * DISCHARGE_EFF
        grid_import = max(0.0, net_load - deliver) + grid_charge
        total += (
            grid_import * period.price_cents_per_kwh
            + deliver * CYCLE_COST_CENTS_PER_KWH
        )
        plans.append(
            PeriodPlan(
                start=period.start,
                action=action,  # type: ignore[arg-type]
                buffer_start_kwh=round(s * STATE_STEP_KWH, 3),
                buffer_end_kwh=round(s_next * STATE_STEP_KWH, 3),
                grid_charge_kwh=round(grid_charge, 3),
                discharge_to_load_kwh=round(deliver, 3),
                grid_import_kwh=round(grid_import, 3),
                price_cents_per_kwh=period.price_cents_per_kwh,
            )
        )
        s = s_next

    return DispatchPlan(
        periods=plans,
        total_cost_cents=round(total, 2),
        baseline_cost_cents=round(baseline, 2),
        end_soc_pct=round(battery.soc_from_buffer_kwh(s * STATE_STEP_KWH), 1),
    )


def _windows(plans: list[PeriodPlan], action: Action) -> list[list[PeriodPlan]]:
    windows: list[list[PeriodPlan]] = []
    run: list[PeriodPlan] = []
    for plan in plans:
        if plan.action == action and (
            action != "charge" or plan.grid_charge_kwh > 0
        ):
            if run and plan.start - run[-1].start != timedelta(minutes=PERIOD_MINUTES):
                windows.append(run)
                run = []
            run.append(plan)
        elif run:
            windows.append(run)
            run = []
    if run:
        windows.append(run)
    return windows


def _window_value(window: list[PeriodPlan], action: Action) -> float:
    if action == "charge":
        return sum(p.grid_charge_kwh for p in window)
    return sum(p.price_cents_per_kwh for p in window) * len(window)


def compile_slots(
    plans: list[PeriodPlan],
    battery: BatteryParams,
    max_slots: int = 6,
) -> tuple[list[SlotSpec], list[SlotSpec]]:
    """Compile the period plan into non-overlapping Solis slot tables.

    Hold windows become enabled 0 A discharge slots; charge windows get the
    minimum current that still fits the planned energy. Windows are ranked
    by value and dropped (never truncated) when they exceed 6 per side or
    would collide wall-clock with a higher-value window on the other side —
    Solis slots recur daily, so cross-side overlap is a real conflict.
    """

    def to_slot(window: list[PeriodPlan], action: Action) -> SlotSpec:
        start = window[0].start
        end = window[-1].start + timedelta(minutes=PERIOD_MINUTES)
        time = f"{start:%H:%M}-{end:%H:%M}"
        if action == "charge":
            energy = sum(p.grid_charge_kwh for p in window)
            hours = len(window) * PERIOD_HOURS
            amps = max(
                1,
                min(
                    battery.max_charge_current,
                    int(energy / CHARGE_EFF / hours / BATTERY_NOMINAL_VOLTAGE * 1000 + 0.999),
                ),
            )
            soc = int(min(100, round(battery.soc_from_buffer_kwh(window[-1].buffer_end_kwh))))
            return SlotSpec(time=time, enabled=True, current=amps, soc=soc)
        soc = int(min(100, round(battery.soc_from_buffer_kwh(window[0].buffer_start_kwh))))
        return SlotSpec(time=time, enabled=True, current=0, soc=soc)

    candidates: list[tuple[float, str, SlotSpec]] = []
    for action in ("charge", "hold"):
        for window in _windows(plans, action):  # type: ignore[arg-type]
            candidates.append(
                (
                    _window_value(window, action),  # type: ignore[arg-type]
                    action,
                    to_slot(window, action),  # type: ignore[arg-type]
                )
            )

    charge: list[SlotSpec] = []
    discharge: list[SlotSpec] = []
    for _value, action, slot in sorted(candidates, key=lambda c: -c[0]):
        target, other = (charge, discharge) if action == "charge" else (discharge, charge)
        if len(target) >= max_slots:
            continue
        cross = (
            find_cross_side_overlaps([slot], other)
            if action == "charge"
            else find_cross_side_overlaps(other, [slot])
        )
        same_side = find_cross_side_overlaps(target, [slot])
        if cross or same_side:
            continue
        target.append(slot)

    charge.sort(key=lambda s: s.time)
    discharge.sort(key=lambda s: s.time)
    while len(charge) < max_slots:
        charge.append(SlotSpec())
    while len(discharge) < max_slots:
        discharge.append(SlotSpec())
    return charge, discharge
