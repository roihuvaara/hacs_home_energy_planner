"""Manual-override tracking with expiry.

A device value that differs from what the planner last wrote means a
human changed it. The planner respects that for a hold window instead of
stomping it on the next tick — but only for the window, so a stale
manual setting can never override the automation forever. Each detection
is counted so recurring interventions surface as data for future
preference learning.

Unknown provenance (planner restarted, never wrote yet) gets the same
grace window when the device disagrees with the plan.

No Home Assistant imports; unit-testable standalone.
"""

from __future__ import annotations

from datetime import datetime, timedelta


class ManualOverrideTracker:
    def __init__(self, tolerance: float = 0.11) -> None:
        self._tolerance = tolerance
        self.last_written: object | None = None
        self.until: datetime | None = None
        self.count = 0
        self._manual_value: object | None = None
        self._expired_manual: object | None = None

    def _same(self, a: object, b: object) -> bool:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) <= self._tolerance
        return a == b

    def record_write(self, value: object) -> None:
        self.last_written = value
        self.until = None
        self._manual_value = None
        self._expired_manual = None

    def to_dict(self) -> dict[str, object | None]:
        """Serialize for persistence across restarts (JSON-safe primitives)."""
        return {
            "last_written": self.last_written,
            "until": self.until.isoformat() if self.until else None,
            "count": self.count,
            "manual_value": self._manual_value,
            "expired_manual": self._expired_manual,
        }

    def load_dict(self, data: dict | None) -> None:
        """Restore state saved by ``to_dict``; a missing/corrupt blob is a no-op.

        Restoring provenance is the whole point: without it, a restart makes
        the planner disown its own last write and treat the device's current
        (planner-authored) state as a fresh manual override, blocking
        reconciliation for a full hold window.
        """
        if not data:
            return
        self.last_written = data.get("last_written")
        until = data.get("until")
        self.until = datetime.fromisoformat(until) if isinstance(until, str) else None
        try:
            self.count = int(data.get("count") or 0)
        except (TypeError, ValueError):
            self.count = 0
        self._manual_value = data.get("manual_value")
        self._expired_manual = data.get("expired_manual")

    def suppressed(
        self, device_value: object | None, desired: object, now: datetime, hold: timedelta
    ) -> bool:
        """True when the planner must not write this tick."""
        if self.until is not None:
            if now < self.until:
                # track the latest manual state so it is consumed at expiry
                if device_value is not None:
                    self._manual_value = device_value
                return True
            # window over: the manual value seen during it is consumed —
            # the planner is free game again (the user's stale-override rule)
            self.until = None
            self._expired_manual = self._manual_value
            self._manual_value = None
        if device_value is None or self._same(device_value, desired):
            return False
        if self.last_written is not None and self._same(
            device_value, self.last_written
        ):
            return False  # device still holds our value; normal write
        if self._expired_manual is not None and self._same(
            device_value, self._expired_manual
        ):
            return False  # same manual value whose window already expired
        if hold <= timedelta(0):
            return False
        # device disagrees with the plan, our last write, and any consumed
        # override: a fresh manual change
        self._manual_value = device_value
        self.until = now + hold
        self.count += 1
        return True
