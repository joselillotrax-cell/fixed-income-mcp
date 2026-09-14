"""Day-count conventions.

A day-count convention answers one question: what fraction of a coupon period
has elapsed between two dates? Every downstream quantity — accrued interest,
the exponent on the first discount factor, and therefore the yield itself —
depends on that answer, so the convention has to be an explicit input rather
than an assumption.

Four conventions are supported. They differ in whether they count real days or
idealised 30-day months, and in what they take the length of a full period to
be.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

__all__ = ["DayCount", "accrual_fraction", "day_count_between"]


class DayCount(str, Enum):
    """Supported conventions.

    ACT_ACT_ICMA
        Actual days elapsed over actual days in the coupon period. The ICMA
        standard for most government and corporate bonds, and the convention
        under which a period always accrues to exactly one full coupon.
    THIRTY_360
        US bond basis. Months are treated as 30 days and years as 360.
    ACT_365
        Actual days elapsed, with a period assumed to be 365/frequency days.
    ACT_360
        Actual days elapsed, with a period assumed to be 360/frequency days.
    """

    ACT_ACT_ICMA = "ACT/ACT ICMA"
    THIRTY_360 = "30/360"
    ACT_365 = "ACT/365"
    ACT_360 = "ACT/360"


def _actual_days(start: date, end: date) -> int:
    return (end - start).days


def _thirty_360_days(start: date, end: date) -> int:
    """US 30/360 (bond basis).

    Both day numbers are capped at 30, with the standard adjustment that a
    31st only becomes a 30th in the end date when the start date is already
    on a 30th or 31st.
    """
    d1 = min(start.day, 30)
    d2 = end.day
    if d1 == 30 and d2 == 31:
        d2 = 30
    return (end.year - start.year) * 360 + (end.month - start.month) * 30 + (d2 - d1)


def day_count_between(start: date, end: date, basis: DayCount) -> int:
    """Days between two dates under `basis`.

    Returns a signed count: negative when `end` precedes `start`.
    """
    if basis is DayCount.THIRTY_360:
        return _thirty_360_days(start, end)
    return _actual_days(start, end)


def accrual_fraction(
    period_start: date,
    settlement: date,
    period_end: date,
    basis: DayCount,
    frequency: int,
) -> float:
    """Fraction of the current coupon period that has already accrued.

    Args:
        period_start: Date of the last coupon paid on or before settlement.
        settlement: Valuation date.
        period_end: Date of the next coupon.
        basis: Day-count convention.
        frequency: Coupon payments per year.

    Returns:
        A fraction in [0, 1] under ACT/ACT ICMA and 30/360. Under ACT/365 and
        ACT/360 the denominator is a nominal period length rather than the
        real one, so the result can exceed 1 in a long period — that is the
        convention behaving as defined, not an error.

    Raises:
        ValueError: If the dates are not ordered, or the period has zero
            length under the chosen basis.
    """
    if not period_start <= settlement <= period_end:
        raise ValueError(
            f"settlement {settlement} must fall within the coupon period "
            f"{period_start} to {period_end}"
        )
    if frequency <= 0:
        raise ValueError(f"frequency must be positive, got {frequency}")

    elapsed = day_count_between(period_start, settlement, basis)

    if basis is DayCount.ACT_365:
        period = 365.0 / frequency
    elif basis is DayCount.ACT_360:
        period = 360.0 / frequency
    else:
        period = float(day_count_between(period_start, period_end, basis))

    if period == 0:
        raise ValueError(
            f"coupon period {period_start} to {period_end} has zero length "
            f"under {basis.value}"
        )

    return elapsed / period
