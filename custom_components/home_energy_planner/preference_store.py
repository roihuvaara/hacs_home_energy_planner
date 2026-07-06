"""Store-backed preference log shared by the module coordinators.

Thin HA glue over the pure `preference` module: loads/saves the event
log, recomputes the learned adjustments on every accepted event, and
hands both to the coordinators and sensors.
"""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .preference import PreferenceEvent, PreferenceLog, derive_adjustments

_LOGGER = logging.getLogger(__name__)


class PreferenceStore:
    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        self._store: Store = Store(hass, 1, f"{DOMAIN}.preferences_{entry_id}")
        self.log = PreferenceLog()
        self._adjustments: dict[str, float] = derive_adjustments([], dt_util.now())

    async def async_load(self) -> None:
        try:
            data = await self._store.async_load()
        except Exception as err:  # noqa: BLE001 - corrupt store: start fresh
            _LOGGER.warning("Preference restore failed: %s", err)
            return
        if data:
            self.log = PreferenceLog.from_dict(data)
        self._recompute()

    def record(
        self,
        module: str,
        planner_value: object,
        manual_value: object,
        context: dict[str, Any],
    ) -> bool:
        event = PreferenceEvent(
            when=dt_util.now(),
            module=module,
            planner_value=planner_value,
            manual_value=manual_value,
            context=context,
        )
        stored = self.log.append(event)
        if stored:
            _LOGGER.info(
                "Preference event: %s planner=%s manual=%s (%s)",
                module,
                planner_value,
                manual_value,
                context,
            )
            self._recompute()
            self._store.async_delay_save(self.log.as_dict, 10)
        return stored

    def _recompute(self) -> None:
        self._adjustments = derive_adjustments(self.log.events, dt_util.now())

    @property
    def adjustments(self) -> dict[str, float]:
        return self._adjustments

    def adjustment(self, key: str) -> float:
        return float(self._adjustments.get(key, 0.0))

    def summary(self) -> dict[str, Any]:
        """Compact shape for sensor attributes / the monthly report."""
        return {
            "event_counts": self.log.counts_by_module(),
            "adjustments": dict(self._adjustments),
        }
