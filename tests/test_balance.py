"""Cell-balance charge: deadline-with-slack policy and the LP constraint."""

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "custom_components"))

from home_energy_planner.balance import (  # noqa: E402
    STAGE_FORCED,
    STAGE_IDLE,
    STAGE_OPPORTUNISTIC,
    BalanceConfig,
    BalanceState,
    allowance_cents,
    days_since,
    decide,
    reference_charge_cost_cents,
    stage_for,
)

TZ = ZoneInfo("Europe/Helsinki")
NOW = datetime(2026, 8, 15, 23, 0, tzinfo=TZ)
CFG = BalanceConfig()


def test_config_rejects_inverted_deadlines():
    with pytest.raises(ValueError):
        BalanceConfig(soft_days=21.0, hard_days=14.0)


def test_stages_split_at_the_two_deadlines():
    assert stage_for(0.0, CFG) == STAGE_IDLE
    assert stage_for(13.9, CFG) == STAGE_IDLE
    assert stage_for(14.0, CFG) == STAGE_OPPORTUNISTIC
    assert stage_for(20.9, CFG) == STAGE_OPPORTUNISTIC
    assert stage_for(21.0, CFG) == STAGE_FORCED
    assert stage_for(400.0, CFG) == STAGE_FORCED


def test_allowance_ramps_from_zero_to_the_full_fraction():
    # at the soft deadline we pay nothing over the optimum: only a charge
    # the solver was going to make anyway (or better) can arm
    assert allowance_cents(14.0, CFG, 30.0) == 0.0
    assert allowance_cents(17.5, CFG, 30.0) == pytest.approx(7.5)  # half way
    assert allowance_cents(21.0, CFG, 30.0) == pytest.approx(15.0)
    # never exceeds the cap, however overdue
    assert allowance_cents(90.0, CFG, 30.0) == pytest.approx(15.0)


def test_idle_never_arms_even_when_the_charge_would_be_free():
    decision = decide(3.0, CFG, 30.0, premium_cents=-5.0)
    assert not decision.armed
    assert decision.stage == STAGE_IDLE
    assert not decision.needs_constrained_solve


def test_opportunistic_arms_only_within_the_allowance():
    # 17.5 d -> 7.5 c allowance on a 30 c reference charge
    assert decide(17.5, CFG, 30.0, premium_cents=6.0).armed
    assert not decide(17.5, CFG, 30.0, premium_cents=9.0).armed


def test_free_or_profitable_balance_arms_on_the_first_opportunity():
    # the case the user cares about: charging through an expensive night
    # ahead of an even more expensive day nets out cheap or negative, so
    # it arms at the very start of the window where the allowance is 0
    decision = decide(14.0, CFG, 30.0, premium_cents=-2.5)
    assert decision.armed
    assert decision.stage == STAGE_OPPORTUNISTIC


def test_expensive_winter_horizon_waits_instead_of_charging():
    # a week of bad prices: premium stays over the allowance every day
    # until the hard deadline, then it charges regardless
    for age in (14.0, 16.0, 18.0, 20.0):
        assert not decide(age, CFG, 30.0, premium_cents=40.0).armed
    forced = decide(21.0, CFG, 30.0, premium_cents=40.0)
    assert forced.armed and forced.stage == STAGE_FORCED


def test_missing_premium_holds_off_but_not_past_the_hard_deadline():
    assert not decide(17.0, CFG, 30.0, premium_cents=None).armed
    assert decide(22.0, CFG, 30.0, premium_cents=None).armed


def test_reference_cost_scales_with_the_horizon_price():
    cheap = reference_charge_cost_cents(4.07, 6.42, 0.9487)
    dear = reference_charge_cost_cents(4.07, 40.0, 0.9487)
    assert cheap == pytest.approx(27.5, abs=0.5)
    assert dear > cheap * 6  # the allowance follows the season by itself


def test_negative_prices_do_not_make_the_reference_negative():
    assert reference_charge_cost_cents(4.07, -3.0, 0.9487) == 0.0


def test_state_resets_the_clock_on_any_full_charge():
    state = BalanceState(last_balanced=NOW - timedelta(days=19))
    # a sunny afternoon reaching 99 % counts — the pack does not care
    # whether the planner or the weather put it there
    assert state.observe_soc(99.0, NOW, CFG)
    assert days_since(state.last_balanced, NOW) == 0.0


def test_a_full_pack_does_not_rewrite_the_store_every_tick():
    state = BalanceState(last_balanced=NOW - timedelta(days=19))
    assert state.observe_soc(99.0, NOW, CFG)  # first mark: persist
    later = NOW + timedelta(minutes=15)
    assert not state.observe_soc(99.0, later, CFG)  # same hour: in memory only
    assert state.last_balanced == later  # ...but still kept fresh
    assert state.observe_soc(99.0, NOW + timedelta(hours=2), CFG)


def test_state_ignores_a_near_miss():
    state = BalanceState(last_balanced=NOW - timedelta(days=19))
    assert not state.observe_soc(96.0, NOW, CFG)
    assert days_since(state.last_balanced, NOW) == pytest.approx(19.0)


def test_fresh_install_is_not_treated_as_overdue():
    state = BalanceState()
    assert days_since(state.last_balanced, NOW) == 0.0
    state.seed(NOW)
    assert stage_for(days_since(state.last_balanced, NOW), CFG) == STAGE_IDLE


def test_seed_does_not_overwrite_a_restored_timestamp():
    earlier = NOW - timedelta(days=19)
    state = BalanceState(last_balanced=earlier)
    state.seed(NOW)
    assert state.last_balanced == earlier


def test_state_round_trips_through_the_store():
    state = BalanceState(last_balanced=NOW - timedelta(days=5))
    restored = BalanceState()
    restored.load_dict(state.to_dict())
    assert restored.last_balanced == state.last_balanced


def test_corrupt_store_payload_starts_fresh():
    restored = BalanceState()
    restored.load_dict({"last_balanced": "not-a-timestamp"})
    assert restored.last_balanced is None
    restored.load_dict(None)
    assert restored.last_balanced is None


def test_deadline_is_the_hard_one():
    state = BalanceState(last_balanced=NOW)
    assert state.next_deadline(CFG) == NOW + timedelta(days=21)
