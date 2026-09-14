"""bondmath — fixed-income analytics that check their own work.

Prices fixed-coupon bonds and measures their interest-rate sensitivity, with
two things a plain calculator leaves out: every analytic duration is validated
against a convention-free numerical one, and every valuation is tested against
the bounds a correct answer has to satisfy.

Pure standard library. No dependencies.

    >>> from datetime import date
    >>> from bondmath import Bond, DayCount, yield_from_price, analytics
    >>> bond = Bond(
    ...     face=1000,
    ...     coupon_rate=0.04,
    ...     frequency=2,
    ...     issue=date(2024, 3, 15),
    ...     maturity=date(2029, 3, 15),
    ...     basis=DayCount.ACT_ACT_ICMA,
    ... )
    >>> settle = date(2025, 9, 11)
    >>> quote = yield_from_price(bond, settle, 978.00)
    >>> round(quote.ytm * 100, 4)
    4.6867
    >>> round(analytics(bond, settle, quote.ytm).modified, 4)
    3.1686
"""

from .analytics import Analytics, Scenario, analytics, scenario
from .bond import Accrual, Bond, CashFlow, accrued_interest, cash_flows
from .checks import Check, ConsistencyReport, check_consistency
from .daycount import DayCount, accrual_fraction, day_count_between
from .pricing import Quote, present_value, price_from_yield, yield_from_price
from .schedule import add_months, bracketing_period, coupon_schedule

__version__ = "0.1.0"

__all__ = [
    "Accrual",
    "Analytics",
    "Bond",
    "CashFlow",
    "Check",
    "ConsistencyReport",
    "DayCount",
    "Quote",
    "Scenario",
    "accrual_fraction",
    "accrued_interest",
    "add_months",
    "analytics",
    "bracketing_period",
    "cash_flows",
    "check_consistency",
    "coupon_schedule",
    "day_count_between",
    "present_value",
    "price_from_yield",
    "scenario",
    "yield_from_price",
]
