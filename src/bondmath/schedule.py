"""Coupon schedule generation.

Schedules are generated backward from maturity rather than forward from issue.
That is the market convention and it matters: it anchors every coupon date to
the maturity day-of-month, so a bond issued off-cycle gets a short or long
first period rather than a drifting schedule.
"""

from __future__ import annotations

import calendar
from datetime import date

__all__ = ["add_months", "coupon_schedule", "bracketing_period"]

_MAX_PERIODS = 4000  # ~333 years at monthly frequency; guards against bad input


def add_months(start: date, months: int) -> date:
    """Shift a date by whole months, clamping to the end of the target month.

    Clamping matters for maturities late in the month: 31 January shifted back
    one month lands on 28 (or 29) February, not on an invalid date.
    """
    total = start.month - 1 + months
    year = start.year + total // 12
    month = total % 12 + 1
    day = min(start.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def coupon_schedule(issue: date, maturity: date, frequency: int) -> list[date]:
    """Coupon dates from issue to maturity, inclusive of both ends.

    Generated backward from maturity. The first entry is the issue date, which
    may sit closer to the second entry than a full period — an irregular first
    period, handled correctly downstream because accrual is always measured
    against the real period the settlement date falls in.

    Args:
        issue: Issue (or dated) date.
        maturity: Redemption date.
        frequency: Coupon payments per year. Must divide 12 evenly.

    Returns:
        Ascending list of dates, starting at `issue` and ending at `maturity`.

    Raises:
        ValueError: On non-positive frequency, a frequency that does not
            divide 12, or a maturity that does not follow issue.
    """
    if frequency <= 0:
        raise ValueError(f"frequency must be positive, got {frequency}")
    if 12 % frequency != 0:
        raise ValueError(
            f"frequency must divide 12 evenly, got {frequency}; "
            "use 1, 2, 3, 4, 6 or 12"
        )
    if maturity <= issue:
        raise ValueError(f"maturity {maturity} must be after issue {issue}")

    step = 12 // frequency
    dates = [maturity]
    cursor = maturity

    for _ in range(_MAX_PERIODS):
        cursor = add_months(cursor, -step)
        if cursor <= issue:
            dates.insert(0, issue)
            return dates
        dates.insert(0, cursor)

    raise ValueError(
        f"schedule from {issue} to {maturity} at frequency {frequency} "
        f"exceeds {_MAX_PERIODS} periods"
    )


def bracketing_period(schedule: list[date], settlement: date) -> tuple[date, date]:
    """The coupon period containing `settlement`.

    Returns the pair (period_start, period_end) such that
    period_start <= settlement < period_end.

    Raises:
        ValueError: If settlement falls outside the schedule. Settlement on
            the maturity date is rejected: there are no remaining flows to
            value.
    """
    for start, end in zip(schedule, schedule[1:]):
        if start <= settlement < end:
            return start, end

    raise ValueError(
        f"settlement {settlement} is outside the schedule "
        f"{schedule[0]} to {schedule[-1]}"
    )
