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
    """Minimize import + cycle cost − export revenue; needs highspy.

    Export economics: absorbing surplus pays its forgone export value;
    unabsorbed surplus earns ``export_cents_per_kwh``. Discharge-to-export
    is deliberately NOT modeled (``d`` stays capped at net load): with
    ~0.9 round-trip and 4 c/kWh cycle cost, selling at spot never clears
    the FI all-in import spread, and the export switch itself is still
    unprobed (todo 002).
    """
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
    export = [p.export_cents_per_kwh for p in periods]
    min_price = min(price, default=0.0)
    # net of cycle cost (spending the stored energy later pays it too);
    # epsilon prefers holding on exact ties, matching the DP
    terminal_value = (
        max(0.0, DISCHARGE_EFF * (min_price - CYCLE_COST_CENTS_PER_KWH)) + 1e-3
    )

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
        # absorbing surplus forgoes exporting it (a_t is battery-side, so
        # a_t/eff of exportable kWh is consumed); the tiny epsilon keeps
        # zero-export-price ties resolving like the DP (capture solar).
        # Negative spot flips the sign: absorbing is then paid.
        cost[ai(t)] = export[t] / CHARGE_EFF - 1e-6
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
    export_revenue = 0.0
    buffer = start_buffer
    for t, period in enumerate(periods):
        g = max(0.0, values[gi(t)])
        a = max(0.0, values[ai(t)])
        d = max(0.0, values[di(t)])
        buffer_end = max(0.0, values[si(t)])
        deliver = d * DISCHARGE_EFF
        grid_charge = g / CHARGE_EFF
        grid_import = max(0.0, net_load[t] - deliver) + grid_charge
        export_kwh = max(0.0, surplus[t] - a / CHARGE_EFF)
        # baseline = no battery: all load imported, all surplus exported
        baseline += net_load[t] * price[t] - surplus[t] * export[t]
        total += (
            grid_import * price[t]
            + deliver * CYCLE_COST_CENTS_PER_KWH
            - export_kwh * export[t]
        )
        export_revenue += export_kwh * export[t]
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
                export_kwh=round(export_kwh, 3),
            )
        )
        buffer = buffer_end

    return DispatchPlan(
        periods=plans,
        total_cost_cents=round(total, 2),
        baseline_cost_cents=round(baseline, 2),
        end_soc_pct=round(battery.soc_from_buffer_kwh(buffer), 1),
        export_revenue_cents=round(export_revenue, 2),
    )


class TankParams:
    """Measured Versati DHW characteristics (2026-07-05 run)."""

    def __init__(
        self,
        power_kw: float = 3.3,
        per_start_kwh: float = 1.0,
        min_run_quarters: int = 3,
        daily_need_kwh: float = 3.1,
        fuse_kw: float = 17.0,
    ) -> None:
        self.power_kw = power_kw
        self.per_start_kwh = per_start_kwh
        self.min_run_quarters = min_run_quarters
        self.daily_need_kwh = daily_need_kwh
        self.fuse_kw = fuse_kw


def solve_joint(
    periods: list[Period],
    battery: BatteryParams,
    tank: TankParams | None = None,
) -> tuple[DispatchPlan, list[tuple[int, int]]]:
    """Co-optimize battery dispatch and tank heat runs over the horizon.

    First true multi-asset solve (research doc step 1): tank runs are
    binaries with min-up and a per-start transient cost, the tank's
    electric draw shares the fuse cap with battery charging, and both
    trade against the same price series. Returns the battery dispatch
    plan plus the scheduled tank run windows as [start, end) period
    indices. MIP with ~2n binaries; HiGHS solves it in milliseconds.
    """
    import highspy

    tank = tank or TankParams()
    n = len(periods)
    if n == 0:
        return (
            DispatchPlan([], 0.0, 0.0, battery.soc_pct),
            [],
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
    terminal_value = (
        max(0.0, DISCHARGE_EFF * (min(price) - CYCLE_COST_CENTS_PER_KWH)) + 1e-3
    )

    delta_h = 0.25
    tank_kwh_per_quarter = tank.power_kw * delta_h
    need_kwh = tank.daily_need_kwh * (n * delta_h / 24.0)
    run_quarters = tank.min_run_quarters

    solver = highspy.Highs()
    solver.silent()

    def gi(t):
        return 3 * t

    def ai(t):
        return 3 * t + 1

    def di(t):
        return 3 * t + 2

    def si(t):
        return 3 * n + t

    def ui(t):
        return 4 * n + t

    def ri(t):  # run-start binary
        return 5 * n + t

    num_vars = 6 * n
    lower = [0.0] * num_vars
    upper = [0.0] * num_vars
    cost = [0.0] * num_vars
    for t in range(n):
        upper[gi(t)] = charge_step
        upper[ai(t)] = surplus[t] * CHARGE_EFF
        upper[di(t)] = min(discharge_step, net_load[t] / DISCHARGE_EFF)
        upper[si(t)] = capacity
        upper[ui(t)] = 1.0
        upper[ri(t)] = 1.0
        cost[gi(t)] = price[t] / CHARGE_EFF
        cost[di(t)] = DISCHARGE_EFF * (CYCLE_COST_CENTS_PER_KWH - price[t])
        # forgone export on absorbed surplus, mirroring solve_lp
        cost[ai(t)] = periods[t].export_cents_per_kwh / CHARGE_EFF - 1e-6
        cost[ui(t)] = tank_kwh_per_quarter * price[t]
        cost[ri(t)] = tank.per_start_kwh * price[t]
    cost[si(n - 1)] -= terminal_value

    solver.addVars(num_vars, lower, upper)
    solver.changeColsCost(num_vars, list(range(num_vars)), cost)
    integer = highspy.HighsVarType.kInteger
    binaries = [ui(t) for t in range(n)] + [ri(t) for t in range(n)]
    solver.changeColsIntegrality(len(binaries), binaries, [integer] * len(binaries))

    inf = highspy.kHighsInf
    for t in range(n):
        # battery state balance
        idx = [si(t), gi(t), ai(t), di(t)]
        coef = [1.0, -1.0, -1.0, 1.0]
        rhs = start_buffer if t == 0 else 0.0
        if t > 0:
            idx.append(si(t - 1))
            coef.append(-1.0)
        solver.addRow(rhs, rhs, len(idx), idx, coef)
        # run start linking: r_t >= u_t - u_{t-1}
        if t == 0:
            solver.addRow(0.0, inf, 2, [ri(0), ui(0)], [1.0, -1.0])
        else:
            solver.addRow(0.0, inf, 3, [ri(t), ui(t), ui(t - 1)], [1.0, -1.0, 1.0])
        # min-up: a start at t forces the next L quarters on
        horizon_left = min(run_quarters, n - t)
        idx_up = [ui(t + k) for k in range(horizon_left)] + [ri(t)]
        coef_up = [1.0] * horizon_left + [-float(horizon_left)]
        solver.addRow(0.0, inf, len(idx_up), idx_up, coef_up)
        # fuse cap: house net load + battery grid charge + tank draw
        solver.addRow(
            -inf,
            tank.fuse_kw * delta_h - net_load[t],
            2,
            [gi(t), ui(t)],
            [1.0 / CHARGE_EFF, tank_kwh_per_quarter],
        )
    # tank energy need over the horizon
    solver.addRow(
        need_kwh,
        inf,
        n,
        [ui(t) for t in range(n)],
        [tank_kwh_per_quarter] * n,
    )

    solver.run()
    if solver.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"HiGHS joint solve: {solver.getModelStatus()}")
    values = list(solver.getSolution().col_value)

    plans: list[PeriodPlan] = []
    total = 0.0
    baseline = 0.0
    buffer = start_buffer
    for t, period in enumerate(periods):
        g = max(0.0, values[gi(t)])
        d = max(0.0, values[di(t)])
        buffer_end = max(0.0, values[si(t)])
        deliver = d * DISCHARGE_EFF
        grid_charge = g / CHARGE_EFF
        grid_import = max(0.0, net_load[t] - deliver) + grid_charge
        baseline += net_load[t] * price[t]
        total += grid_import * price[t] + deliver * CYCLE_COST_CENTS_PER_KWH
        action = "charge" if g > _EPS else "self_use" if d > _EPS else "hold"
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

    windows: list[tuple[int, int]] = []
    run_start = None
    for t in range(n):
        on = values[ui(t)] > 0.5
        if on and run_start is None:
            run_start = t
        elif not on and run_start is not None:
            windows.append((run_start, t))
            run_start = None
    if run_start is not None:
        windows.append((run_start, n))

    return (
        DispatchPlan(
            periods=plans,
            total_cost_cents=round(total, 2),
            baseline_cost_cents=round(baseline, 2),
            end_soc_pct=round(battery.soc_from_buffer_kwh(buffer), 1),
        ),
        windows,
    )
