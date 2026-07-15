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
from dataclasses import dataclass
from datetime import tzinfo

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
    """Measured Versati DHW tank physics (thermal-battery model).

    Sources (owner rule: constants from data, never textbook):
    - power_kw / heat_c_per_min: 2026-07-05 measured run (3.3 kW electric,
      0.23 C/min — the electric-equivalent kWh/C below therefore already
      folds in whatever COP the DHW mode achieves).
    - start_shortfall_c: ~19 min zero-gain startup transient (the old
      per_start_kwh = 1.0 kWh expressed as missing temperature rise).
    - loss_per_hour: exponential fit of the 2026-07-07 17:00 -> 07-08
      09:00 coast (58.0 -> 51.1 C, pannuhuone ~21 C, no draws): k = 0.013.
      NOTE: the earlier "53->30 in 18 h" figure included a large shower
      draw and overstates loss ~5x.
    - daily_draw_kwh: electric-equivalent of actual DHW draws only —
      standing loss is modeled explicitly now, and loss alone accounts
      for most of the tank's measured ~3 kWh/day.
    - min_c/max_c: owner comfort floor 50 C (2026-07-15), dump ceiling 66
      (device max 80 verified).
    """

    def __init__(
        self,
        power_kw: float = 3.3,
        heat_c_per_min: float = 0.23,
        start_shortfall_c: float = 4.4,
        loss_per_hour: float = 0.013,
        ambient_c: float = 21.0,
        min_c: float = 50.0,
        max_c: float = 66.0,
        daily_draw_kwh: float = 1.0,
        min_run_quarters: int = 3,
        fuse_kw: float = 17.0,
    ) -> None:
        self.power_kw = power_kw
        self.heat_c_per_min = heat_c_per_min
        self.start_shortfall_c = start_shortfall_c
        self.loss_per_hour = loss_per_hour
        self.ambient_c = ambient_c
        self.min_c = min_c
        self.max_c = max_c
        self.daily_draw_kwh = daily_draw_kwh
        self.min_run_quarters = min_run_quarters
        self.fuse_kw = fuse_kw

    @property
    def kwh_per_quarter(self) -> float:
        return self.power_kw * 0.25

    @property
    def gain_c_per_quarter(self) -> float:
        return self.heat_c_per_min * 15.0

    @property
    def kwh_per_c(self) -> float:
        """Electric-equivalent kWh to raise the tank 1 C (COP folded in)."""
        return self.power_kw / (self.heat_c_per_min * 60.0)


@dataclass(frozen=True)
class TankPlan:
    """Planned tank trajectory: run schedule + predicted temperature."""

    on: list[bool]
    temp_c: list[float]
    windows: list[tuple[int, int]]  # [start, end) period indices
    surplus_kwh: list[float]  # surplus fed to the tank per period
    electric_cost_cents: float
    floor_slack_c: float  # >0 means the comfort floor was unreachable


def solve_joint(
    periods: list[Period],
    battery: BatteryParams,
    tank: TankParams | None = None,
    initial_temp_c: float | None = None,
    local_tz: tzinfo | None = None,
) -> tuple[DispatchPlan, TankPlan]:
    """Co-optimize battery dispatch and the DHW tank as a thermal battery.

    The tank carries a temperature state: runs raise it 3.45 C/quarter
    (minus a per-start shortfall), standing loss decays it toward the
    pannuhuone ambient, forecast draws debit it, and the comfort floor
    must hold at all times (a slack keeps a cold start solvable — heat
    first, don't go infeasible). Surplus PV is shared three ways per
    quarter — battery absorb, tank feed, export — priced by the export
    contract, so "dump surplus into hot water vs sell it" is decided by
    the objective, not a rule. MIP with 2n binaries; HiGHS stays in the
    milliseconds.

    ``local_tz`` places the draw profile (07:00-23:00 local); omit when
    period starts are already local. Returns the battery dispatch plan
    plus a TankPlan (run windows, predicted temperature trajectory).
    """
    import highspy

    tank = tank or TankParams()
    n = len(periods)
    if n == 0:
        return (
            DispatchPlan([], 0.0, 0.0, battery.soc_pct),
            TankPlan([], [], [], [], 0.0, 0.0),
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
    terminal_value = (
        max(0.0, DISCHARGE_EFF * (min(price) - CYCLE_COST_CENTS_PER_KWH)) + 1e-3
    )

    tank_kwh_q = tank.kwh_per_quarter
    gain_q = tank.gain_c_per_quarter
    k_q = tank.loss_per_hour / 4.0
    run_quarters = tank.min_run_quarters
    temp0 = tank.min_c if initial_temp_c is None else float(initial_temp_c)
    # forecast draws as a temperature debit, spread over waking hours;
    # the MPC re-reads the live tank temp every tick so errors self-heal
    draw_hours = range(7, 23)
    draw_quarters = [
        t
        for t in range(n)
        if (
            periods[t].start.astimezone(local_tz)
            if local_tz is not None
            else periods[t].start
        ).hour
        in draw_hours
    ]
    draw_c = [0.0] * n
    if draw_quarters:
        per_quarter_c = (
            tank.daily_draw_kwh * (n * 0.25 / 24.0) / len(draw_quarters)
        ) / tank.kwh_per_c
        for t in draw_quarters:
            draw_c[t] = per_quarter_c

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

    def Ti(t):  # tank temperature AFTER period t
        return 6 * n + t

    def wi(t):  # surplus kWh fed to the tank
        return 7 * n + t

    def zi(t):  # per-quarter comfort-floor slack (cold starts stay solvable;
        # a shared slack would relax EVERY quarter once paid, letting the
        # solver camp below the floor forever)
        return 8 * n + t

    num_vars = 9 * n
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
        upper[Ti(t)] = tank.max_c
        upper[wi(t)] = min(surplus[t], tank_kwh_q)
        cost[gi(t)] = price[t] / CHARGE_EFF
        cost[di(t)] = DISCHARGE_EFF * (CYCLE_COST_CENTS_PER_KWH - price[t])
        # forgone export on absorbed surplus, mirroring solve_lp
        cost[ai(t)] = export[t] / CHARGE_EFF - 1e-6
        # a run buys grid energy unless fed from surplus: w swaps grid
        # cost for forgone export (negative saving when price > export)
        cost[ui(t)] = tank_kwh_q * price[t]
        cost[wi(t)] = export[t] - price[t]
        upper[zi(t)] = tank.min_c
        # well above any credible energy price so violations only absorb
        # the genuinely unreachable deficit of a below-floor start
        cost[zi(t)] = 50.0
    cost[si(n - 1)] -= terminal_value
    # leftover tank heat avoids future runs at least at the horizon's
    # cheapest price; without it every solve coasts to the floor edge
    cost[Ti(n - 1)] -= tank.kwh_per_c * max(0.0, min(price))

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
            tank.fuse_kw * 0.25 - net_load[t],
            2,
            [gi(t), ui(t)],
            [1.0 / CHARGE_EFF, tank_kwh_q],
        )
        # tank temperature dynamics:
        # T_t = (1-k)*T_{t-1} + k*ambient + gain*u_t - shortfall*r_t - draw
        # (the start quarter nets 3.45-4.4 < 0 — a ~1 C artifact dip since
        # the real transient is "no gain", not negative; it mildly penalizes
        # starting exactly at the floor, which is conservative)
        rhs_t = k_q * tank.ambient_c - draw_c[t]
        idx_temp = [Ti(t), ui(t), ri(t)]
        coef_temp = [1.0, -gain_q, tank.start_shortfall_c]
        if t == 0:
            rhs_t += (1.0 - k_q) * temp0
        else:
            idx_temp.append(Ti(t - 1))
            coef_temp.append(-(1.0 - k_q))
        solver.addRow(rhs_t, rhs_t, len(idx_temp), idx_temp, coef_temp)
        # comfort floor with slack: T_t + z_t >= min_c
        solver.addRow(tank.min_c, inf, 2, [Ti(t), zi(t)], [1.0, 1.0])
        # surplus feed only while running
        solver.addRow(-inf, 0.0, 2, [wi(t), ui(t)], [1.0, -tank_kwh_q])
        # surplus is shared: battery absorb + tank feed <= surplus
        solver.addRow(
            -inf, surplus[t], 2, [ai(t), wi(t)], [1.0 / CHARGE_EFF, 1.0]
        )

    solver.run()
    if solver.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"HiGHS joint solve: {solver.getModelStatus()}")
    values = list(solver.getSolution().col_value)

    plans: list[PeriodPlan] = []
    total = 0.0
    baseline = 0.0
    export_revenue = 0.0
    tank_cost = 0.0
    buffer = start_buffer
    for t, period in enumerate(periods):
        g = max(0.0, values[gi(t)])
        a = max(0.0, values[ai(t)])
        d = max(0.0, values[di(t)])
        w = max(0.0, values[wi(t)])
        buffer_end = max(0.0, values[si(t)])
        deliver = d * DISCHARGE_EFF
        grid_charge = g / CHARGE_EFF
        grid_import = max(0.0, net_load[t] - deliver) + grid_charge
        export_kwh = max(0.0, surplus[t] - a / CHARGE_EFF - w)
        baseline += net_load[t] * price[t] - surplus[t] * export[t]
        total += (
            grid_import * price[t]
            + deliver * CYCLE_COST_CENTS_PER_KWH
            - export_kwh * export[t]
        )
        export_revenue += export_kwh * export[t]
        on = values[ui(t)] > 0.5
        tank_cost += (tank_kwh_q * (1.0 if on else 0.0) - w) * price[t] + w * export[t]
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
                export_kwh=round(export_kwh, 3),
            )
        )
        buffer = buffer_end

    on_series = [values[ui(t)] > 0.5 for t in range(n)]
    windows: list[tuple[int, int]] = []
    run_start = None
    for t, on in enumerate(on_series):
        if on and run_start is None:
            run_start = t
        elif not on and run_start is not None:
            windows.append((run_start, t))
            run_start = None
    if run_start is not None:
        windows.append((run_start, n))

    tank_plan = TankPlan(
        on=on_series,
        temp_c=[round(values[Ti(t)], 2) for t in range(n)],
        windows=windows,
        surplus_kwh=[round(max(0.0, values[wi(t)]), 3) for t in range(n)],
        electric_cost_cents=round(tank_cost, 2),
        floor_slack_c=round(max(0.0, max(values[zi(t)] for t in range(n))), 2),
    )
    return (
        DispatchPlan(
            periods=plans,
            total_cost_cents=round(total, 2),
            baseline_cost_cents=round(baseline, 2),
            end_soc_pct=round(battery.soc_from_buffer_kwh(buffer), 1),
            export_revenue_cents=round(export_revenue, 2),
        ),
        tank_plan,
    )
