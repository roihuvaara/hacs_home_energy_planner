# Home Energy Planner

Home Assistant custom integration that owns the household energy optimization
pipeline: electricity pricing, battery (Solis) slot planning, climate target
computation, and hot-water control — replacing a stack of template automations
and helper entities.

Architecture and the migration plan are documented in ADR 0009 of the
maintainer's separate (non-public) Home Assistant docs repository.

## Status

Phase 0: **pricing module** only.

- Fetches the Nord Pool quarter-hour spot series from the `nordpool`
  integration (today, plus tomorrow once published).
- Computes VAT, seller margin, and day/night transfer fees into an all-in
  price horizon (all constants configurable via options).
- Publishes three sensors — spot, with-VAT, and all-in — whose state is the
  current quarter-hour price and whose `horizon` attribute carries the full
  forward series. No more chunked `input_text` helpers.

Planned modules (see ADR 0009): battery (Solis slot planner with a hardened
writer), climate, hot water, EV charging. Each module gets an
`off | observe | control` mode so it can run in shadow mode against the legacy
automations before taking over.

## Installation

1. Add this repository to HACS as a custom repository (type: integration).
2. Install "Home Energy Planner" and restart Home Assistant.
3. Add the integration; the config flow asks for pricing constants and
   auto-discovers the Nord Pool config entry.

## Development

Pure logic lives in modules with no Home Assistant imports
(`pricing.py`) and is covered by unit tests that run without an HA install:

```
python -m pytest tests
```
