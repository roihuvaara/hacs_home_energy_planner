"""Periodic cell-balance charge policy (pure, no Home Assistant imports).

A pack the cost optimizer never drives to full loses two things the cost
model cannot see: the BMS balances cells only in the top of the charge
curve, and the SoC estimate needs a full charge to recalibrate against
something other than the flat OCV plateau. Left alone the optimizer parks
the pack at the reserve floor for weeks — right on cents, wrong on pack
health. Observed live 2026-07-16..2026-08-09: 13 consecutive days peaking
under 50 %, only 4 days above 95 % in 30.

The policy is deliberately NOT "charge every N days". It is a deadline
with slack:

- before ``soft_days`` nothing happens at all — a sunny day or an ordinary
  arbitrage charge that reaches ``detect_soc_pct`` resets the clock for
  free, which is what already happens most of the summer;
- between soft and hard the charge is armed only when it is cheap enough,
  and the price we will pay ramps from zero at ``soft_days`` to the full
  allowance at ``hard_days`` — pickiest at the start of the window,
  resigned by the end of it;
- at ``hard_days`` it happens at whatever the horizon costs.

"Cheap enough" is the **premium**: the objective difference between the
optimizer's plan with the balance constraint and without it. Both solves
may spend the stored energy again inside the same horizon, so the premium
is already net of the recovery discharge — charging through an expensive
night ahead of an even more expensive day shows up as a small or negative
premium and arms on its own, with no rule needed to describe that case.

The allowance is scaled to the horizon's own cheapest price rather than
being a fixed cent figure, so it means the same thing on a 6 c/kWh summer
night as on a 40 c/kWh winter one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

STAGE_IDLE = "idle"
STAGE_OPPORTUNISTIC = "opportunistic"
STAGE_FORCED = "forced"


@dataclass(frozen=True)
class BalanceConfig:
    # aim for a genuine full charge: balancing happens in the CV taper.
    # The LP's linear charge model overshoots that taper, but the slot the
    # writer emits carries an SoC target, so the inverter owns the last
    # few percent.
    target_soc_pct: float = 100.0
    # what counts as "balanced" on the way back in. The pack does report a
    # flat 100 on good solar days, but never require the last percent to
    # land exactly — a 99 that stalls would restart the whole clock.
    detect_soc_pct: float = 97.0
    soft_days: float = 14.0
    hard_days: float = 21.0
    # willingness to pay at the hard deadline, as a fraction of what the
    # same charge would cost at the horizon's cheapest price
    max_premium_frac: float = 0.5

    def __post_init__(self) -> None:
        if self.hard_days <= self.soft_days:
            raise ValueError("hard_days must exceed soft_days")
        if self.detect_soc_pct > self.target_soc_pct:
            raise ValueError("detect_soc_pct cannot exceed target_soc_pct")


@dataclass(frozen=True)
class BalanceDecision:
    stage: str
    armed: bool
    days_since: float
    allowance_cents: float
    premium_cents: float | None
    reason: str

    @property
    def needs_constrained_solve(self) -> bool:
        """Whether the caller should pay for the second solve this tick."""
        return self.stage != STAGE_IDLE

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "armed": self.armed,
            "days_since": round(self.days_since, 2),
            "allowance_cents": round(self.allowance_cents, 2),
            "premium_cents": (
                None if self.premium_cents is None else round(self.premium_cents, 2)
            ),
            "reason": self.reason,
        }


def days_since(last_balanced: datetime | None, now: datetime) -> float:
    """Age of the last full charge in days; 0.0 when never recorded.

    A missing timestamp means a fresh install, not an overdue pack — the
    caller seeds it to ``now`` so a new setup does not force a charge on
    its first tick.
    """
    if last_balanced is None:
        return 0.0
    return max(0.0, (now - last_balanced).total_seconds() / 86400.0)


def stage_for(age_days: float, config: BalanceConfig) -> str:
    if age_days >= config.hard_days:
        return STAGE_FORCED
    if age_days >= config.soft_days:
        return STAGE_OPPORTUNISTIC
    return STAGE_IDLE


def reference_charge_cost_cents(
    balance_energy_kwh: float, min_price_cents: float, charge_eff: float
) -> float:
    """What a reserve-to-target charge costs at the horizon's best price.

    The full span is used rather than the charge still outstanding from
    the current SoC: the allowance should express how much a balance
    charge is worth to the pack, which does not shrink just because today
    happens to start half full.
    """
    return max(0.0, balance_energy_kwh) / charge_eff * max(0.0, min_price_cents)


def allowance_cents(
    age_days: float, config: BalanceConfig, reference_cost_cents: float
) -> float:
    """Premium we will pay today: zero at soft_days, full at hard_days."""
    if age_days < config.soft_days:
        return 0.0
    span = config.hard_days - config.soft_days
    ramp = min(1.0, (age_days - config.soft_days) / span)
    return config.max_premium_frac * max(0.0, reference_cost_cents) * ramp


def decide(
    age_days: float,
    config: BalanceConfig,
    reference_cost_cents: float,
    premium_cents: float | None,
) -> BalanceDecision:
    """Arm or hold off, given the premium the constrained solve reported.

    ``premium_cents=None`` means the constrained solve was not run or did
    not return — the balance charge is then skipped unless the pack is
    past the hard deadline, where refusing to act is the worse failure.
    """
    stage = stage_for(age_days, config)
    allowance = allowance_cents(age_days, config, reference_cost_cents)

    if stage == STAGE_IDLE:
        return BalanceDecision(
            stage,
            False,
            age_days,
            allowance,
            premium_cents,
            f"{age_days:.1f}d since full charge, below the {config.soft_days:.0f}d window",
        )

    if stage == STAGE_FORCED:
        return BalanceDecision(
            stage,
            True,
            age_days,
            allowance,
            premium_cents,
            f"{age_days:.1f}d since full charge, past the {config.hard_days:.0f}d "
            "deadline — charging at whatever the horizon costs",
        )

    if premium_cents is None:
        return BalanceDecision(
            stage,
            False,
            age_days,
            allowance,
            None,
            "no constrained solve available; waiting for a later tick",
        )

    armed = premium_cents <= allowance
    verb = "within" if armed else "over"
    return BalanceDecision(
        stage,
        armed,
        age_days,
        allowance,
        premium_cents,
        f"{age_days:.1f}d since full charge; balance premium "
        f"{premium_cents:.1f} c is {verb} today's {allowance:.1f} c allowance",
    )


@dataclass
class BalanceState:
    """Mutable last-balanced timestamp; persisted across restarts."""

    last_balanced: datetime | None = None

    def observe_soc(self, soc_pct: float, now: datetime, config: BalanceConfig) -> bool:
        """Reset the clock when the pack is (however it got there) full.

        Returns whether the change is worth persisting. A pack sitting at
        100 % all afternoon would otherwise rewrite the store every tick;
        an hour of lag is nothing against a 14-day deadline.
        """
        if soc_pct < config.detect_soc_pct:
            return False
        previous = self.last_balanced
        self.last_balanced = now
        return previous is None or (now - previous) > timedelta(hours=1)

    def seed(self, now: datetime) -> None:
        if self.last_balanced is None:
            self.last_balanced = now

    def next_deadline(self, config: BalanceConfig) -> datetime | None:
        if self.last_balanced is None:
            return None
        return self.last_balanced + timedelta(days=config.hard_days)

    def to_dict(self) -> dict:
        return {
            "last_balanced": (
                self.last_balanced.isoformat() if self.last_balanced else None
            )
        }

    def load_dict(self, data: dict | None) -> None:
        if not data:
            return
        raw = data.get("last_balanced")
        if not raw:
            return
        try:
            self.last_balanced = datetime.fromisoformat(raw)
        except (TypeError, ValueError):
            self.last_balanced = None
