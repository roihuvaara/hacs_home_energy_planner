"""Planner self-monitoring: notify when silently degraded.

The planner acts loudly (slot writes, trials, regime notifications) but
its failure modes are quiet: a stale price horizon, a dead input sensor,
persistent inverter apply failures, or the LP engine falling back to DP.
This watchdog checks twice an hour, notifies once per issue per day, and
sends a monthly savings report on the 1st.

`evaluate_issues` is pure and unit-tested; the coordinator glue only
builds the snapshot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

_LOGGER = logging.getLogger(__name__)

PRICES_STALE_HOURS = 2.0
HORIZON_MIN_HOURS = 8.0
INPUT_STALE_HOURS = 3.0
APPLY_FAILING_HOURS = 2.0
ENGINE_FALLBACK_HOURS = 1.0
RENOTIFY_HOURS = 24.0

@dataclass(frozen=True)
class InputRule:
    """Staleness rule for one critical input.

    stale_hours: silence tolerated before flagging.
    """

    entity_id: str
    stale_hours: float = INPUT_STALE_HOURS


# sensor.ilp_ulkolampotila is deliberately NOT monitored: MELCloud reports
# outdoor temp as "unknown" for hours-long stretches even while the unit
# runs (observed 2026-07-14..16), and its only consumer is a gap-tolerant
# 7-day recorder mean — a warning is never actionable.
CRITICAL_INPUTS = [
    InputRule("sensor.olohuone_climate_lampotila"),
    InputRule("sensor.olohuone_climate_kosteus"),
    # SolisCloud re-reports SOC only on change: a battery idling at the
    # reserve floor is legitimately silent all night (observed 2026-07-14,
    # flat 18 % from 23:07). Datalogger death still alarms within 3 h via
    # the grid power sensor below.
    InputRule("sensor.solis_remaining_battery_capacity", stale_hours=12.0),
    InputRule("sensor.solis_total_consumption_power"),
    InputRule("sensor.solis_power_grid_total_power"),
]


def stale_critical_inputs(get_state, now: datetime) -> list[str]:
    """Entity ids of critical inputs that are stale per their InputRule."""
    stale: list[str] = []
    for rule in CRITICAL_INPUTS:
        if input_is_stale(get_state(rule.entity_id), now, rule.stale_hours):
            stale.append(rule.entity_id)
    return stale


def input_is_stale(
    state: Any, now: datetime, stale_hours: float = INPUT_STALE_HOURS
) -> bool:
    """Missing/unavailable, or nothing reported for stale_hours.

    Staleness reads last_reported, not last_updated: a polled sensor
    re-reporting an unchanged value (flat outdoor temp for hours) only
    advances last_reported, and it is alive.
    """
    if state is None or state.state in ("unavailable", "unknown"):
        return True
    reported = getattr(state, "last_reported", None) or state.last_updated
    return (now - reported).total_seconds() > stale_hours * 3600


@dataclass(frozen=True)
class WatchdogSnapshot:
    prices_age_hours: float | None  # None = no price data at all
    horizon_hours: float
    stale_inputs: list[str] = field(default_factory=list)
    battery_apply_failing_hours: float = 0.0
    engine_fallback_hours: float = 0.0


def evaluate_issues(snapshot: WatchdogSnapshot) -> list[tuple[str, str]]:
    """(key, message) per active issue; keys are stable for debouncing."""
    issues: list[tuple[str, str]] = []
    if snapshot.prices_age_hours is None:
        issues.append(("prices_missing", "No price horizon at all — planner is blind."))
    elif snapshot.prices_age_hours > PRICES_STALE_HOURS:
        issues.append(
            (
                "prices_stale",
                f"Price horizon is {snapshot.prices_age_hours:.1f} h stale "
                "(Nord Pool fetches failing?).",
            )
        )
    elif snapshot.horizon_hours < HORIZON_MIN_HOURS:
        issues.append(
            (
                "horizon_short",
                f"Price horizon covers only {snapshot.horizon_hours:.1f} h.",
            )
        )
    for entity_id in snapshot.stale_inputs:
        issues.append(
            (
                f"input:{entity_id}",
                f"{entity_id} unavailable or silent past its staleness budget — "
                "dependent logic is degraded (check sensor/battery).",
            )
        )
    if snapshot.battery_apply_failing_hours >= APPLY_FAILING_HOURS:
        issues.append(
            (
                "battery_apply",
                f"Inverter slot applies failing for "
                f"{snapshot.battery_apply_failing_hours:.1f} h.",
            )
        )
    if snapshot.engine_fallback_hours >= ENGINE_FALLBACK_HOURS:
        issues.append(
            (
                "engine_fallback",
                f"LP engine unavailable for {snapshot.engine_fallback_hours:.1f} h "
                "(running on the DP fallback — check highspy install).",
            )
        )
    return issues


class Debouncer:
    """At most one notification per issue key per RENOTIFY_HOURS."""

    def __init__(self) -> None:
        self._last: dict[str, datetime] = {}

    def due(self, key: str, now: datetime) -> bool:
        last = self._last.get(key)
        if last is not None and now - last < timedelta(hours=RENOTIFY_HOURS):
            return False
        self._last[key] = now
        return True


class PlannerWatchdog:
    STARTUP_GRACE = timedelta(minutes=15)

    def __init__(self, hass, entry_id: str) -> None:
        self.hass = hass
        self._entry_id = entry_id
        self._debounce = Debouncer()
        self._apply_failing_since: datetime | None = None
        self._engine_fallback_since: datetime | None = None
        self._last_report_month: tuple[int, int] | None = None
        from homeassistant.util import dt as dt_util

        self._started = dt_util.now()

    def _bundle(self) -> dict[str, Any] | None:
        from .const import DOMAIN

        return self.hass.data.get(DOMAIN, {}).get(self._entry_id)

    def build_snapshot(self, now: datetime) -> WatchdogSnapshot:
        bundle = self._bundle() or {}
        pricing = bundle.get("pricing")
        prices_age = None
        horizon_hours = 0.0
        data = getattr(pricing, "data", None)
        if data is not None and data.periods:
            prices_age = max(
                0.0, (now - data.horizon_start).total_seconds() / 3600.0
            )
            horizon_hours = len(data.periods) * 0.25

        stale: list[str] = []
        # restored states keep pre-restart last_updated until integrations
        # poll, so skip input staleness during the startup grace window
        if now - self._started >= self.STARTUP_GRACE:
            stale = stale_critical_inputs(self.hass.states.get, now)

        battery = bundle.get("battery")
        battery_data = getattr(battery, "data", None)
        failing = (
            battery_data is not None
            and battery_data.mode == "control"
            and battery_data.applied is not None
            and not battery_data.applied.get("success")
        )
        if failing and self._apply_failing_since is None:
            self._apply_failing_since = now
        if not failing:
            self._apply_failing_since = None
        fallback = False
        if battery is not None and battery_data is not None:
            fallback = battery_data.engine != "lp" and (
                str(battery._option("battery_engine")) == "lp"  # noqa: SLF001
            )
        if fallback and self._engine_fallback_since is None:
            self._engine_fallback_since = now
        if not fallback:
            self._engine_fallback_since = None

        def hours_since(since: datetime | None) -> float:
            return (now - since).total_seconds() / 3600.0 if since else 0.0

        return WatchdogSnapshot(
            prices_age_hours=prices_age,
            horizon_hours=horizon_hours,
            stale_inputs=stale,
            battery_apply_failing_hours=hours_since(self._apply_failing_since),
            engine_fallback_hours=hours_since(self._engine_fallback_since),
        )

    async def _notify(self, title: str, message: str, notification_id: str) -> None:
        await self.hass.services.async_call(
            "persistent_notification",
            "create",
            {"title": title, "message": message, "notification_id": notification_id},
            blocking=False,
        )
        try:
            await self.hass.services.async_call(
                "notify", "notify", {"title": title, "message": message}, blocking=False
            )
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("Mobile notify unavailable: %s", err)

    async def async_check(self, now: datetime) -> dict[str, Any]:
        snapshot = self.build_snapshot(now)
        issues = evaluate_issues(snapshot)
        for key, message in issues:
            _LOGGER.warning("Watchdog: %s", message)
            if self._debounce.due(key, now):
                await self._notify(
                    "Energy planner degraded", message, f"hep_watchdog_{key}"
                )
        await self._maybe_monthly_report(now)
        return {
            "issues": [{"key": k, "message": m} for k, m in issues],
            "snapshot": {
                "prices_age_hours": snapshot.prices_age_hours,
                "horizon_hours": snapshot.horizon_hours,
                "stale_inputs": snapshot.stale_inputs,
                "battery_apply_failing_hours": snapshot.battery_apply_failing_hours,
                "engine_fallback_hours": snapshot.engine_fallback_hours,
            },
        }

    async def _maybe_monthly_report(self, now: datetime) -> None:
        if now.day != 1 or not 8 <= now.hour < 10:
            return
        month_key = (now.year, now.month)
        if self._last_report_month == month_key:
            return
        self._last_report_month = month_key
        bundle = self._bundle() or {}
        try:
            from .backtest import async_backtest

            report = await async_backtest(
                bundle["pricing"], bundle["battery"], {"days": 30}
            )
            totals = report["totals"]
            preference_line = ""
            preferences = bundle.get("preferences")
            if preferences is not None:
                summary = preferences.summary()
                active = {
                    key: value
                    for key, value in summary["adjustments"].items()
                    if value
                }
                preference_line = (
                    f" Overrides: {summary['event_counts'] or 'none'};"
                    f" learned: {active or 'no drift'}."
                )
            await self._notify(
                "Energy planner monthly report",
                f"Last 30 days: baseline {totals['baseline_cents'] / 100:.2f} e, "
                f"planned {totals['planned_cents'] / 100:.2f} e "
                f"(savings ceiling {totals['planned_savings_cents'] / 100:.2f} e, "
                f"{totals['planned_savings_pct']}%), actual "
                f"{(totals['actual_cents'] or 0) / 100:.2f} e over "
                f"{totals['days_evaluated']} days." + preference_line,
                "hep_monthly_report",
            )
        except Exception as err:  # noqa: BLE001 - report is best-effort
            _LOGGER.warning("Monthly report failed: %s", err)
