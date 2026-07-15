"""Shared pure price-window helpers for rule-based modules.

Ranked cheapest non-overlapping windows within a budget — used by the
water heater rule fallback and the ILP module, so "cheap" means the
genuinely cheapest ranked stretches of the coming day, not merely the
cheaper half (a median split marks ~12 h/day as cheap).
"""

from __future__ import annotations


def median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def plan_cheap_windows(
    prices: list[float],
    *,
    min_run_quarters: int,
    budget_quarters: int,
    margin_cents: float,
) -> list[tuple[int, int]]:
    """Cheapest non-overlapping min-run windows within the budget.

    Greedy by window average price; a window qualifies only when its
    average sits ``margin_cents`` below the day median, so a flat day
    yields no windows at all. Adjacent picks merge into longer runs.
    """
    run = min_run_quarters
    if len(prices) < run:
        return []
    mid = median(prices)
    means = [
        (sum(prices[i : i + run]) / run, i) for i in range(len(prices) - run + 1)
    ]
    chosen: list[tuple[int, int]] = []
    budget = budget_quarters
    for mean, start in sorted(means):
        if budget < run or mean > mid - margin_cents:
            break
        end = start + run
        if any(start < c_end and c_start < end for c_start, c_end in chosen):
            continue
        chosen.append((start, end))
        budget -= run
    chosen.sort()
    merged: list[tuple[int, int]] = []
    for start, end in chosen:
        if merged and start == merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged
