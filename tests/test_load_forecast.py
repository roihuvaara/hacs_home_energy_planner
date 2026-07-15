from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "custom_components"))

from home_energy_planner.load_forecast import (  # noqa: E402
    blend_weekday_baseline,
    quarter_bucket,
)


def samples_for(weekday, bucket, value, n):
    return [(weekday, bucket, value)] * n


def test_quarter_bucket():
    assert quarter_bucket(0, 0) == 0
    assert quarter_bucket(7, 14) == 7 * 60
    assert quarter_bucket(7, 15) == 7 * 60 + 15
    assert quarter_bucket(23, 59) == 23 * 60 + 45


def test_thin_weekday_bucket_leans_on_all_days_mean():
    # one Saturday sample vs a rich weekday history: the Saturday value
    # barely moves the blend away from the all-days mean
    samples = samples_for(2, 480, 300.0, 24) + samples_for(5, 480, 3000.0, 1)
    blended = blend_weekday_baseline(samples, k=12.0)
    all_mean = (300.0 * 24 + 3000.0) / 25
    assert blended[(5, 480)] < 1000.0
    assert blended[(5, 480)] > all_mean  # nudged toward its own sample
    # the unsampled weekdays get the plain all-days mean
    assert blended[(0, 480)] == pytest.approx(all_mean)


def test_rich_weekday_bucket_follows_its_own_mean():
    samples = samples_for(5, 480, 3000.0, 100) + samples_for(2, 480, 300.0, 100)
    blended = blend_weekday_baseline(samples, k=12.0)
    assert blended[(5, 480)] > 2500.0
    assert blended[(2, 480)] < 800.0


def test_weekend_heavy_history_separates_saturday_from_wednesday():
    samples = []
    for weekday in range(7):
        value = 2000.0 if weekday >= 5 else 400.0
        samples += samples_for(weekday, 600, value, 12)  # 4 weeks of 5-min rows
    blended = blend_weekday_baseline(samples)
    assert blended[(5, 600)] > blended[(2, 600)] + 500.0


def test_empty_history_yields_empty_baseline():
    assert blend_weekday_baseline([]) == {}
