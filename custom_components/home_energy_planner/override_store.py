"""Persist manual-override trackers across restarts.

The pure ``ManualOverrideTracker`` stays HA-import-free; this thin wrapper
gives a coordinator's set of named trackers a single Store so their
provenance survives a restart. Without it, every restart disowns the
planner's own last write and turns the device's current (planner-authored)
state into a phantom manual override for a full hold window.
"""

from __future__ import annotations

import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DOMAIN
from .manual_override import ManualOverrideTracker

_LOGGER = logging.getLogger(__name__)


class OverridePersistence:
    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        key: str,
        trackers: dict[str, ManualOverrideTracker],
    ) -> None:
        self._store: Store = Store(hass, 1, f"{DOMAIN}.{key}_overrides_{entry_id}")
        self._trackers = trackers

    async def async_restore(self) -> None:
        try:
            data = await self._store.async_load()
        except Exception as err:  # noqa: BLE001 - corrupt store: start fresh
            _LOGGER.warning("Override restore failed: %s", err)
            return
        if not data:
            return
        for name, tracker in self._trackers.items():
            tracker.load_dict(data.get(name))

    def _data(self) -> dict[str, dict]:
        return {name: t.to_dict() for name, t in self._trackers.items()}

    def save(self) -> None:
        self._store.async_delay_save(self._data, 10)
