"""Pure weekday-aware load baseline blending.

The 7-day flat time-of-day average collapsed weekdays, so a Saturday
laundry rhythm polluted Wednesday's forecast and vice versa. This
blends a per-(weekday, quarter) mean with the all-days quarter mean,
weighted by how many samples the weekday bucket actually has — thin
buckets lean on the all-days shape, rich ones follow their weekday.
No Home Assistant imports; unit-testable standalone.
"""

from __future__ import annotations

from collections.abc import Iterable

PERIOD_MINUTES = 15

# k is the pseudo-count of the all-days prior. One weekday-day of 5-min
# recorder rows per quarter bucket is ~3 samples x 4 weeks = 12, so with
# k=12 a 4-week history splits weekday/all-days weight roughly 50/50.
DEFAULT_PRIOR_WEIGHT = 12.0

WEEKDAYS = range(7)


def quarter_bucket(hour: int, minute: int) -> int:
    return hour * 60 + (minute // PERIOD_MINUTES) * PERIOD_MINUTES


def blend_weekday_baseline(
    samples: Iterable[tuple[int, int, float]],
    k: float = DEFAULT_PRIOR_WEIGHT,
) -> dict[tuple[int, int], float]:
    """Blend (weekday, bucket, value) samples into a keyed baseline.

    Returns a value for every (weekday, bucket) combination for each
    bucket seen at all: buckets with no samples for a given weekday fall
    back to the all-days mean; sampled buckets get
    (n * weekday_mean + k * alldays_mean) / (n + k).
    """
    sums: dict[tuple[int, int], float] = {}
    counts: dict[tuple[int, int], int] = {}
    all_sums: dict[int, float] = {}
    all_counts: dict[int, int] = {}
    for weekday, bucket, value in samples:
        key = (weekday, bucket)
        sums[key] = sums.get(key, 0.0) + value
        counts[key] = counts.get(key, 0) + 1
        all_sums[bucket] = all_sums.get(bucket, 0.0) + value
        all_counts[bucket] = all_counts.get(bucket, 0) + 1

    blended: dict[tuple[int, int], float] = {}
    for bucket, total in all_sums.items():
        all_mean = total / all_counts[bucket]
        for weekday in WEEKDAYS:
            key = (weekday, bucket)
            n = counts.get(key, 0)
            if n == 0:
                blended[key] = all_mean
            else:
                weekday_mean = sums[key] / n
                blended[key] = (n * weekday_mean + k * all_mean) / (n + k)
    return blended
