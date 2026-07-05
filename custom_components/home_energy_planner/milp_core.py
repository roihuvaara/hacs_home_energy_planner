"""LP battery dispatch via HiGHS (ADR 0009 phase 1b optimizer step 1).

Same problem as the DP in battery_core (import cost + cycle cost over
the horizon, terminal value on leftover buffer) formulated as a
continuous LP — no 0.1 kWh quantization, milliseconds to solve, and the
foundation the joint battery+tank MILP extends with binaries. The DP
stays as the cross-check engine (tests assert cost parity) and as the
runtime fallback when highspy is unavailable.
"""

from __future__ import annotations

import logging

from .battery_core import (
    CHARGE_EFF,
    CYCLE_COST_CENTS_PER_KWH,
    DISCHARGE_EFF,
    BatteryParams,
    DispatchPlan,
    Period,
    PeriodPlan,
    current_to_period_kwh,
)

_LOGGER = logging.getLogger(__name__)

_EPS = 1e-6


def solve_best(
    periods: list[Period], battery: BatteryParams, engine: str = "lp"
) -> tuple[DispatchPlan, str]:
    """Solve with the requested engine, falling back LP -> DP."""
    if engine == "lp":
        try:
            return solve_lp(periods, battery), "lp"
        except Exception as err:  # noqa: BLE001 - DP fallback by design
            _LOGGER.warning("LP engine unavailable (%s); falling back to DP", err)
    from .battery_core import solve

    return solve(periods, battery), "dp"


def solve_lp(periods: list[Period], battery: BatteryParams) -> DispatchPlan:
    """Minimize import + cycle cost; raises ImportError without highspy."""
    import highspy

    n = len(periods)
    if n == 0:
        return DispatchPlan(
            periods=[],
            total_cost_cents=0.0,
            baseline_cost_cents=0.0,
            end_soc_pct=battery.soc_pct,
        )
    capacity = battery.usable_above_reserve_kwh
    start_buffer = min(capacity, battery.buffer_kwh_from_soc(battery.soc_pct))
    charge_step = current_to_period_kwh(
        min(battery.planned_charge_current, battery.max_charge_current)
    )
    discharge_step = current_to_period_kwh(battery.max_discharge_current)

    net_load = [max(0.0, p.load_kwh - p.solar_kwh) for p in periods]
    surplus = [max(0.0, p.solar_kwh - p.load_kwh) for p in periods]
    price = [p.price_cents_per_kwh for p in periods]
    min_price = min(price, default=0.0)
    terminal_value = max(0.0, DISCHARGE_EFF * min_price)

    solver = highspy.Highs()
    solver.silent()
    inf = highspy.kHighsInf

    # variable layout per period: g (grid->buffer), a (solar->buffer),
    # d (buffer drain); plus s (buffer state) for periods 1..n
    def gi(t: int) -> int:
        return 3 * t

    def ai(t: int) -> int:
        return 3 * t + 1

    def di(t: int) -> int:
        return 3 * t + 2

    def si(t: int) -> int:  # state AFTER period t
        return 3 * n + t

    num_vars = 4 * n
    lower = [0.0] * num_vars
    upper = [0.0] * num_vars
    cost = [0.0] * num_vars
    for t in range(n):
        upper[gi(t)] = charge_step
        upper[ai(t)] = surplus[t] * CHARGE_EFF
        upper[di(t)] = min(discharge_step, net_load[t] / DISCHARGE_EFF)
        upper[si(t)] = capacity
        # import cost: grid charge buys g/eff at price; discharge saves
        # d*eff of load at price but pays cycle cost on delivered energy
        cost[gi(t)] = price[t] / CHARGE_EFF
        cost[di(t)] = DISCHARGE_EFF * (CYCLE_COST_CENTS_PER_KWH - price[t])
        # tiny reward for absorbing free surplus so ties resolve like the
        # DP (which always captures solar) instead of leaving it
        cost[ai(t)] = -1e-6
    cost[si(n - 1)] -= terminal_value

    solver.addVars(num_vars, lower, upper)
    solver.changeColsCost(num_vars, list(range(num_vars)), cost)

    # state balance: s_t - s_{t-1} - g_t - a_t + d_t = 0
    for t in range(n):
        idx = [si(t), gi(t), ai(t), di(t)]
        coef = [1.0, -1.0, -1.0, 1.0]
        rhs = start_buffer if t == 0 else 0.0
        if t > 0:
            idx.append(si(t - 1))
            coef.append(-1.0)
        solver.addRow(rhs, rhs, len(idx), idx, coef)

    solver.run()
    status = solver.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"HiGHS returned {status}")
    values = list(solver.getSolution().col_value)

    plans: list[PeriodPlan] = []
    total = 0.0
    baseline = 0.0
    buffer = start_buffer
    for t, period in enumerate(periods):
        g = max(0.0, values[gi(t)])
        a = max(0.0, values[ai(t)])
        d = max(0.0, values[di(t)])
        buffer_end = max(0.0, values[si(t)])
        deliver = d * DISCHARGE_EFF
        grid_charge = g / CHARGE_EFF
        grid_import = max(0.0, net_load[t] - deliver) + grid_charge
        baseline += net_load[t] * price[t]
        total += grid_import * price[t] + deliver * CYCLE_COST_CENTS_PER_KWH
        if g > _EPS:
            action = "charge"
        elif d > _EPS:
            action = "self_use"
        else:
            action = "hold"
        plans.append(
            PeriodPlan(
                start=period.start,
                action=action,  # type: ignore[arg-type]
                buffer_start_kwh=round(buffer, 3),
                buffer_end_kwh=round(buffer_end, 3),
                grid_charge_kwh=round(grid_charge, 3),
                discharge_to_load_kwh=round(deliver, 3),
                grid_import_kwh=round(grid_import, 3),
                price_cents_per_kwh=price[t],
            )
        )
        buffer = buffer_end

    return DispatchPlan(
        periods=plans,
        total_cost_cents=round(total, 2),
        baseline_cost_cents=round(baseline, 2),
        end_soc_pct=round(battery.soc_from_buffer_kwh(buffer), 1),
    )
