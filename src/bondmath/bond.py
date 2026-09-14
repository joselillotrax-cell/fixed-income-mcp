"""The instrument, its cash flows, and accrued interest."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .daycount import DayCount, accrual_fraction
from .schedule import bracketing_period, coupon_schedule

__all__ = ["Bond", "CashFlow", "Accrual", "accrued_interest", "cash_flows"]


@dataclass(frozen=True)
class Bond:
    """A fixed-coupon bullet bond.

    Attributes:
        face: Redemption amount, also the base for the coupon.
        coupon_rate: Annual coupon as a decimal (0.04 for a 4% coupon).
        frequency: Coupon payments per year.
        issue: Issue (dated) date.
        maturity: Redemption date.
        basis: Day-count convention used for accrual.
    """

    face: float
    coupon_rate: float
    frequency: int
    issue: date
    maturity: date
    basis: DayCount = DayCount.ACT_ACT_ICMA

    def __post_init__(self) -> None:
        if self.face <= 0:
            raise ValueError(f"face must be positive, got {self.face}")
        if self.coupon_rate < 0:
            raise ValueError(f"coupon_rate must not be negative, got {self.coupon_rate}")
        if self.frequency <= 0 or 12 % self.frequency != 0:
            raise ValueError(
                f"frequency must be a positive divisor of 12, got {self.frequency}"
            )
        if self.maturity <= self.issue:
            raise ValueError(
                f"maturity {self.maturity} must be after issue {self.issue}"
            )

    @property
    def coupon(self) -> float:
        """Amount of a single coupon payment."""
        return self.face * self.coupon_rate / self.frequency

    @property
    def annual_coupon(self) -> float:
        """Total coupon paid per year."""
        return self.face * self.coupon_rate

    def schedule(self) -> list[date]:
        """Coupon dates from issue to maturity."""
        return coupon_schedule(self.issue, self.maturity, self.frequency)


@dataclass(frozen=True)
class CashFlow:
    """One remaining payment, positioned relative to settlement.

    Attributes:
        date: Payment date.
        amount: Payment amount, including principal on the final flow.
        periods: Distance from settlement in coupon periods. The first flow
            sits at a fraction of a period, not a whole one.
        years: The same distance expressed in years.
    """

    date: date
    amount: float
    periods: float
    years: float


@dataclass(frozen=True)
class Accrual:
    """Accrued interest and the period it was measured in.

    Attributes:
        amount: Accrued interest in currency units.
        fraction: Fraction of the coupon period elapsed.
        period_start: Last coupon date on or before settlement.
        period_end: Next coupon date.
        days_elapsed: Days from period_start to settlement, under the basis.
        days_in_period: Length of the period, under the basis.
    """

    amount: float
    fraction: float
    period_start: date
    period_end: date
    days_elapsed: int
    days_in_period: float


def accrued_interest(bond: Bond, settlement: date) -> Accrual:
    """Interest accrued between the last coupon and settlement.

    Raises:
        ValueError: If settlement is outside [issue, maturity).
    """
    from .daycount import day_count_between  # local import keeps the API flat

    schedule = bond.schedule()
    start, end = bracketing_period(schedule, settlement)

    fraction = accrual_fraction(start, settlement, end, bond.basis, bond.frequency)
    elapsed = day_count_between(start, settlement, bond.basis)

    if bond.basis is DayCount.ACT_365:
        in_period: float = 365.0 / bond.frequency
    elif bond.basis is DayCount.ACT_360:
        in_period = 360.0 / bond.frequency
    else:
        in_period = float(day_count_between(start, end, bond.basis))

    return Accrual(
        amount=bond.coupon * fraction,
        fraction=fraction,
        period_start=start,
        period_end=end,
        days_elapsed=elapsed,
        days_in_period=in_period,
    )


def cash_flows(bond: Bond, settlement: date) -> list[CashFlow]:
    """Remaining payments after settlement, in date order.

    The first flow sits `1 - accrued_fraction` periods away rather than a full
    period. Getting that exponent right is what separates a correct yield from
    one that is quietly off by a few basis points.

    Raises:
        ValueError: If no flows remain after settlement.
    """
    schedule = bond.schedule()
    accrual = accrued_interest(bond, settlement)
    first_gap = 1.0 - accrual.fraction

    flows: list[CashFlow] = []
    for pay_date in schedule[1:]:
        if pay_date <= settlement:
            continue
        index = len(flows)
        amount = bond.coupon + (bond.face if pay_date == bond.maturity else 0.0)
        periods = first_gap + index
        flows.append(
            CashFlow(
                date=pay_date,
                amount=amount,
                periods=periods,
                years=periods / bond.frequency,
            )
        )

    if not flows:
        raise ValueError(f"no cash flows remain after settlement {settlement}")

    return flows
