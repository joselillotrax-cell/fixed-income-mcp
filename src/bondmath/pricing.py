"""Pricing and yield solving.

The yield is solved by bisection rather than Newton-Raphson. Newton converges
faster but can diverge or settle on a value that satisfies the iteration
without solving the pricing equation — which is precisely the failure this
library exists to rule out. Bisection cannot do that: given a bracket where
the function changes sign, it always converges to the root.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .bond import Bond, CashFlow, accrued_interest, cash_flows

__all__ = ["Quote", "present_value", "price_from_yield", "yield_from_price"]

_YIELD_LOW = -0.95
_YIELD_HIGH = 10.0
_BISECTION_STEPS = 200


@dataclass(frozen=True)
class Quote:
    """A bond valuation at a point in time.

    Attributes:
        clean: Price excluding accrued interest, in currency units.
        dirty: Price including accrued interest — what actually settles.
        accrued: Accrued interest.
        clean_pct: Clean price as a percentage of face.
        dirty_pct: Dirty price as a percentage of face.
        ytm: Nominal annual yield, compounded at the coupon frequency.
        effective_annual_yield: The same yield expressed as an annual
            effective rate, which is what makes bonds of different coupon
            frequencies comparable.
    """

    clean: float
    dirty: float
    accrued: float
    clean_pct: float
    dirty_pct: float
    ytm: float
    effective_annual_yield: float


def present_value(flows: list[CashFlow], ytm: float, frequency: int) -> float:
    """Discount cash flows at a nominal yield compounded `frequency` times a year."""
    periodic = ytm / frequency
    if periodic <= -1:
        raise ValueError(
            f"periodic yield {periodic} implies a non-positive discount factor"
        )
    return sum(f.amount / (1 + periodic) ** f.periods for f in flows)


def _quote(bond: Bond, settlement: date, dirty: float, ytm: float) -> Quote:
    accrual = accrued_interest(bond, settlement)
    clean = dirty - accrual.amount
    periodic = ytm / bond.frequency
    return Quote(
        clean=clean,
        dirty=dirty,
        accrued=accrual.amount,
        clean_pct=clean / bond.face * 100,
        dirty_pct=dirty / bond.face * 100,
        ytm=ytm,
        effective_annual_yield=(1 + periodic) ** bond.frequency - 1,
    )


def price_from_yield(bond: Bond, settlement: date, ytm: float) -> Quote:
    """Value the bond at a given yield to maturity.

    Args:
        bond: The instrument.
        settlement: Valuation date.
        ytm: Nominal annual yield as a decimal, compounded at the bond's
            coupon frequency.
    """
    flows = cash_flows(bond, settlement)
    dirty = present_value(flows, ytm, bond.frequency)
    return _quote(bond, settlement, dirty, ytm)


def yield_from_price(bond: Bond, settlement: date, clean_price: float) -> Quote:
    """Solve for the yield that reproduces a given clean price.

    Args:
        bond: The instrument.
        settlement: Valuation date.
        clean_price: Clean price in currency units — not a percentage. Pass
            978.0 for a bond quoted at 97.80% of a 1,000 face.

    Raises:
        ValueError: If no yield above the search floor can produce that price,
            which happens when the price exceeds the undiscounted sum of the
            remaining flows.
    """
    flows = cash_flows(bond, settlement)
    accrual = accrued_interest(bond, settlement)
    target = clean_price + accrual.amount

    if target <= 0:
        raise ValueError(f"dirty price must be positive, got {target}")

    low, high = _YIELD_LOW, _YIELD_HIGH

    # present_value is monotonically decreasing in yield, so the root is
    # bracketed iff the low end prices above the target and the high end below.
    ceiling = present_value(flows, low, bond.frequency)
    if ceiling < target:
        raise ValueError(
            f"no yield above {low:.0%} reproduces a dirty price of {target:,.2f}; "
            f"the highest price in the search range is {ceiling:,.2f}"
        )

    for _ in range(_BISECTION_STEPS):
        mid = (low + high) / 2
        if present_value(flows, mid, bond.frequency) > target:
            low = mid
        else:
            high = mid

    return _quote(bond, settlement, target, (low + high) / 2)
