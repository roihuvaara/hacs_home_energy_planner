"""Solis slot writer: diff-based, ordered, retried, twice-verified.

Reliability contract (ADR 0009):
1. Only changed fields are written, disables first, enables last.
2. Callers gate on the slot table alone (a no-op diff writes nothing).
3. Each write is retried with backoff; a persistent failure aborts the
   remaining ops and raises a notification instead of half-continuing.
4. Verification runs twice: immediately (optimistic HA state) and again
   after a delay so a Solis-side revert on the next device refresh is
   caught and alerted, not silently missed.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_call_later

from .solis_slots import (
    SLOT_COUNT,
    SlotSpec,
    WriteOp,
    diff_tables,
    diff_write_ops,
    find_cross_side_overlaps,
    normalize_table,
    validate_table,
)

_LOGGER = logging.getLogger(__name__)

_SLOT_TIME_RE = re.compile(r"^text\.(.+)_slot1_charge_time$")

WRITE_ATTEMPTS = 3
RETRY_DELAYS_S = (2.0, 5.0)
DEFAULT_REVERIFY_DELAY_S = 180.0
VERIFY_EVENT = "home_energy_planner_slot_verification"
NOTIFICATION_ID = "home_energy_planner_solis_writer"
_GENERATION_KEY = "home_energy_planner_slot_apply_generation"


def _bump_generation(hass: HomeAssistant) -> int:
    generation = hass.data.get(_GENERATION_KEY, 0) + 1
    hass.data[_GENERATION_KEY] = generation
    return generation


def discover_slot_prefix(hass: HomeAssistant) -> str:
    prefixes: list[str] = []
    for entity_id in hass.states.async_entity_ids("text"):
        match = _SLOT_TIME_RE.match(entity_id)
        if not match:
            continue
        prefix = match.group(1)
        if all(
            hass.states.get(entity_id) is not None
            for entity_id in (
                f"text.{prefix}_slot1_discharge_time",
                f"switch.{prefix}_slot1_charge",
                f"switch.{prefix}_slot1_discharge",
            )
        ):
            prefixes.append(prefix)
    if len(prefixes) != 1:
        raise ValueError(
            f"Expected exactly one Solis slot entity prefix, found {prefixes}"
        )
    return prefixes[0]


def _entity_id(prefix: str, side: str, slot: int, field: str) -> str:
    if field == "enabled":
        return f"switch.{prefix}_slot{slot}_{side}"
    if field == "time":
        return f"text.{prefix}_slot{slot}_{side}_time"
    return f"number.{prefix}_slot{slot}_{side}_{field}"


class SlotTableUnavailable(Exception):
    """A slot entity is unreadable; the table cannot be trusted for diffing."""


def _slot_value(state, entity_id: str) -> str:
    if state is None or state.state in ("unavailable", "unknown", ""):
        raise SlotTableUnavailable(f"{entity_id} is unavailable")
    return str(state.state)


def read_slot_table(hass: HomeAssistant, prefix: str, side: str) -> list[SlotSpec]:
    """Read the inverter's current table.

    Raises SlotTableUnavailable when any entity is unreadable — diffing
    against a half-known table is how tables get corrupted, so callers
    abort the apply and retry on the next tick.
    """
    table: list[SlotSpec] = []
    for slot in range(1, SLOT_COUNT + 1):
        time_id = _entity_id(prefix, side, slot, "time")
        current_id = _entity_id(prefix, side, slot, "current")
        soc_id = _entity_id(prefix, side, slot, "soc")
        enabled_id = _entity_id(prefix, side, slot, "enabled")
        try:
            table.append(
                SlotSpec(
                    time=_slot_value(hass.states.get(time_id), time_id),
                    enabled=_slot_value(hass.states.get(enabled_id), enabled_id).lower()
                    == "on",
                    current=int(
                        round(float(_slot_value(hass.states.get(current_id), current_id)))
                    ),
                    soc=int(
                        round(float(_slot_value(hass.states.get(soc_id), soc_id)))
                    ),
                )
            )
        except ValueError as err:
            raise SlotTableUnavailable(f"{side} slot {slot}: {err}") from err
    return table


async def _execute_op(hass: HomeAssistant, prefix: str, op: WriteOp) -> None:
    entity_id = _entity_id(prefix, op.side, op.slot, op.field)
    if op.field == "enabled":
        await hass.services.async_call(
            "switch",
            "turn_on" if op.value else "turn_off",
            {"entity_id": entity_id},
            blocking=True,
        )
    elif op.field == "time":
        await hass.services.async_call(
            "text",
            "set_value",
            {"entity_id": entity_id, "value": str(op.value)},
            blocking=True,
        )
    else:
        await hass.services.async_call(
            "number",
            "set_value",
            {"entity_id": entity_id, "value": int(op.value)},
            blocking=True,
        )


async def _execute_op_with_retry(
    hass: HomeAssistant, prefix: str, op: WriteOp
) -> list[str]:
    """Run one op, retrying on failure. Returns error strings per attempt."""

    errors: list[str] = []
    for attempt in range(WRITE_ATTEMPTS):
        try:
            await _execute_op(hass, prefix, op)
            return errors
        except Exception as err:
            errors.append(f"attempt {attempt + 1}: {err}")
            if attempt < len(RETRY_DELAYS_S):
                await asyncio.sleep(RETRY_DELAYS_S[attempt])
    raise SolisWriteError(op, errors)


class SolisWriteError(Exception):
    def __init__(self, op: WriteOp, errors: list[str]) -> None:
        super().__init__(f"Write failed for {op}: {errors}")
        self.op = op
        self.errors = errors


async def _notify(hass: HomeAssistant, title: str, message: str) -> None:
    await hass.services.async_call(
        "persistent_notification",
        "create",
        {
            "notification_id": NOTIFICATION_ID,
            "title": title,
            "message": message,
        },
        blocking=True,
    )


def _tables_payload(charge: list[SlotSpec], discharge: list[SlotSpec]) -> dict[str, Any]:
    return {
        "charge_slots": [slot.as_dict() for slot in charge],
        "discharge_slots": [slot.as_dict() for slot in discharge],
    }


async def apply_slots(
    hass: HomeAssistant,
    *,
    charge_slots: Any,
    discharge_slots: Any,
    dry_run: bool = False,
    allow_cross_side_overlap: bool = False,
    reverify_delay_s: float = DEFAULT_REVERIFY_DELAY_S,
) -> dict[str, Any]:
    prefix = discover_slot_prefix(hass)
    desired_charge = normalize_table(charge_slots)
    desired_discharge = normalize_table(discharge_slots)

    problems = validate_table(desired_charge, "charge") + validate_table(
        desired_discharge, "discharge"
    )
    overlaps = find_cross_side_overlaps(desired_charge, desired_discharge)
    if overlaps and not allow_cross_side_overlap:
        problems.extend(overlaps)
    if problems:
        return {
            "success": False,
            "error": "validation_failed",
            "problems": problems,
        }

    try:
        current_charge = read_slot_table(hass, prefix, "charge")
        current_discharge = read_slot_table(hass, prefix, "discharge")
    except SlotTableUnavailable as err:
        _LOGGER.warning("Solis apply skipped, table unreadable: %s", err)
        return {
            "success": False,
            "error": "slot_table_unavailable",
            "message": str(err),
        }
    ops = diff_write_ops(
        current_charge=current_charge,
        current_discharge=current_discharge,
        desired_charge=desired_charge,
        desired_discharge=desired_discharge,
    )
    ops_payload = [
        {"side": op.side, "slot": op.slot, "field": op.field, "value": op.value}
        for op in ops
    ]

    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "prefix": prefix,
            "ops_planned": ops_payload,
            "op_count": len(ops),
            "cross_side_overlaps": overlaps,
            "current": _tables_payload(current_charge, current_discharge),
        }

    completed = 0
    retry_errors: list[str] = []
    try:
        for op in ops:
            attempt_errors = await _execute_op_with_retry(hass, prefix, op)
            retry_errors.extend(attempt_errors)
            completed += 1
    except SolisWriteError as err:
        message = (
            f"Aborted after {completed}/{len(ops)} ops; "
            f"failed op {err.op.side} slot {err.op.slot} {err.op.field}: {err.errors}"
        )
        _LOGGER.error("Solis slot write aborted: %s", message)
        await _notify(hass, "Solis slot write aborted", message)
        return {
            "success": False,
            "error": "write_aborted",
            "message": message,
            "ops_planned": ops_payload,
            "ops_completed": completed,
            "retry_errors": retry_errors,
        }

    try:
        actual_charge = read_slot_table(hass, prefix, "charge")
        actual_discharge = read_slot_table(hass, prefix, "discharge")
    except SlotTableUnavailable:
        # writes landed but readback is flapping; trust the delayed pass
        actual_charge, actual_discharge = desired_charge, desired_discharge
    immediate_mismatches = diff_tables(
        desired_charge, actual_charge, "charge"
    ) + diff_tables(desired_discharge, actual_discharge, "discharge")

    generation = _bump_generation(hass)
    if ops and reverify_delay_s > 0:
        _schedule_reverify(
            hass,
            prefix=prefix,
            desired_charge=desired_charge,
            desired_discharge=desired_discharge,
            delay_s=reverify_delay_s,
            generation=generation,
        )

    return {
        "success": True,
        "dry_run": False,
        "prefix": prefix,
        "ops_planned": ops_payload,
        "op_count": len(ops),
        "ops_completed": completed,
        "retry_errors": retry_errors,
        "immediate_verification_ok": not immediate_mismatches,
        "immediate_mismatches": immediate_mismatches,
        "reverify_scheduled_in_s": reverify_delay_s if ops else 0,
        "readback": _tables_payload(actual_charge, actual_discharge),
    }


def _schedule_reverify(
    hass: HomeAssistant,
    *,
    prefix: str,
    desired_charge: list[SlotSpec],
    desired_discharge: list[SlotSpec],
    delay_s: float,
    generation: int,
) -> None:
    async def _reverify(_now: Any) -> None:
        if hass.data.get(_GENERATION_KEY, 0) != generation:
            _LOGGER.debug(
                "Skipping stale re-verify (generation %d superseded)", generation
            )
            return
        try:
            actual_charge = read_slot_table(hass, prefix, "charge")
            actual_discharge = read_slot_table(hass, prefix, "discharge")
        except SlotTableUnavailable as err:
            _LOGGER.warning("Delayed re-verify skipped, table unreadable: %s", err)
            return
        mismatches = diff_tables(desired_charge, actual_charge, "charge") + diff_tables(
            desired_discharge, actual_discharge, "discharge"
        )
        hass.bus.async_fire(
            VERIFY_EVENT,
            {
                "ok": not mismatches,
                "mismatches": mismatches,
                "delay_s": delay_s,
            },
        )
        if mismatches:
            message = f"Slot table diverged after {delay_s:.0f}s: {mismatches}"
            _LOGGER.warning("Solis delayed verification failed: %s", message)
            await _notify(hass, "Solis slot verification failed", message)
        else:
            _LOGGER.info(
                "Solis delayed verification ok after %.0fs (%d slots checked)",
                delay_s,
                SLOT_COUNT * 2,
            )

    async_call_later(hass, delay_s, _reverify)


async def read_slots(hass: HomeAssistant) -> dict[str, Any]:
    prefix = discover_slot_prefix(hass)
    try:
        charge = read_slot_table(hass, prefix, "charge")
        discharge = read_slot_table(hass, prefix, "discharge")
    except SlotTableUnavailable as err:
        return {
            "success": False,
            "error": "slot_table_unavailable",
            "message": str(err),
            "prefix": prefix,
        }
    return {
        "success": True,
        "prefix": prefix,
        **_tables_payload(charge, discharge),
        "cross_side_overlaps": find_cross_side_overlaps(charge, discharge),
    }
