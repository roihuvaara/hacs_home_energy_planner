from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "custom_components"))

from home_energy_planner.summary import (  # noqa: E402
    SummaryInputs,
    battery_text,
    build_summary,
    climate_text,
    extreme_window,
    ilp_text,
    price_stance,
    water_text,
)

TZ = ZoneInfo("Europe/Helsinki")
START = datetime(2026, 7, 8, 0, 0, tzinfo=TZ)


def test_price_stance_cheap_normal_expensive():
    rising = list(range(1, 21))  # current is the cheapest of the horizon
    assert price_stance(rising)["stance"] == "cheap"
    assert price_stance(rising)["cheap_now"] is True
    falling = list(range(20, 0, -1))  # current is the most expensive
    assert price_stance(falling)["stance"] == "expensive"
    flat = [10.0] * 10  # flat day: timing buys nothing, so neither extreme
    assert price_stance(flat)["stance"] == "normal"
    assert price_stance([])["stance"] == "unknown"


def test_extreme_window_finds_cheapest_and_peak():
    horizon = [10, 10, 2, 2, 2, 2, 10, 20, 20, 20, 20, 10]
    cheap = extreme_window(horizon, 4, cheapest=True)
    assert cheap["start_index"] == 2 and cheap["end_index"] == 6 and cheap["mean"] == 2
    peak = extreme_window(horizon, 4, cheapest=False)
    assert peak["start_index"] == 7 and peak["mean"] == 20
    assert extreme_window([1, 2], 4, cheapest=True) is None


def test_asset_text_reads_like_a_human():
    assert battery_text(33, False, True) == "covering the load · 33%"
    assert battery_text(None, True, False) == "charging"
    assert "manual until 17:40" in water_text("normal", 55, "17:40")
    assert water_text("cheap_boost", 40, None) == "heating — cheap window · 40°C"
    assert climate_text("neutral", None, 23.0) == "idle — house comfortable · room 23.0°C"
    assert ilp_text("dry", "humid room in cheap half") == "drying — humid room in cheap half"
    assert ilp_text("off", "comfortable") == "off — comfortable"


def test_build_summary_headline_and_windows():
    horizon = [2, 5, 5, 5, 10, 10, 20, 20, 20, 20, 10, 10]
    healthy = build_summary(
        SummaryInputs(horizon=horizon, horizon_start=START, soc_pct=50)
    )
    assert healthy["headline"] == "In control"
    assert healthy["status"] == "ok"
    assert healthy["price"]["cheap_now"] is True
    assert healthy["coming_up"].startswith("cheap power now")
    # peak window is reported so the person knows when to hold off
    assert healthy["peak_window"]["start"] == "01:30"

    degraded = build_summary(
        SummaryInputs(
            horizon=horizon,
            horizon_start=START,
            issues=["sensor.foo unavailable > 3 h"],
        )
    )
    assert degraded["headline"] == "Attention"
    assert degraded["status"] == "attention"
    assert degraded["issues"] == ["sensor.foo unavailable > 3 h"]
