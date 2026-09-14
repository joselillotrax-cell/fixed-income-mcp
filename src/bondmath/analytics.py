"""Interest-rate sensitivity.

Modified duration is computed by dividing Macaulay duration by one plus the
*periodic* yield. Dividing by one plus the *annual* yield is the standard
mistake on any bond paying more than once a year, and it understates the
figure by roughly the coupon frequency's share of the yield.

Because that error is the reason this library exists, `Analytics` also carries
an effective duration computed by central differences. Effective duration
involves no annualisation at all, so it cannot inherit a convention error and
can arbitrate whenever the analytic figure looks wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .bond import Bond, cash_flows
from .pricing import present_value

__all__ = ["Analytics", "Scenario", "analytics", "scenario"]

_BUMP = 0.0001  # 1 bp, used for the DV01 definition
_EFFECTIVE_BUMP = 0.01  # 100 bp, wide enough to stay clear of float noise


@dataclass(frozen=True)
class Analytics:
    """Sensitivity measures at a given yield.

    Attributes:
        macaulay: Macaulay duration in years — the present-value-weighted
            average time to receipt.
        modified: Modified duration in years. The percentage price change for
            a one-unit change in yield.
        effective: Duration from central differences on the pricing function.
            Convention-free, and therefore the check on `modified`.
        dv01: Currency change in dirty price per basis point.
        convexity: Convexity in years squared, from the second derivative of
            price with respect to yield.
        naive_modified: Macaulay divided by one plus the *annual* yield — the
            wrong divisor for a bond paying more than once a year. Reported so
            the error can be detected rather than merely avoided. Equal to
            `modified` when the frequency is annual.
    """

    macaulay: float
    modified: float
    effective: float
    dv01: float
    convexity: float
    naive_modified: float

    @property
    def annualisation_gap(self) -> float:
        """Relative distance between the correct and naive divisors, as a fraction.

        Zero for annual-pay bonds. Around 0.023 for a semi-annual bond yielding
        near 4.7%.
        """
        if self.naive_modified == 0:
            return 0.0
        return (self.modified - self.naive_modified) / self.naive_modified


@dataclass(frozen=True)
class Scenario:
    """A parallel shift in yield, priced two ways.

    Attributes:
        shock_bp: Size of the shift in basis points.
        exact: Dirty price from discounting every flow at the new yield.
        duration_only: First-order estimate.
        duration_convexity: Second-order estimate.
        residual_bp: Distance from the second-order estimate to the exact
            price, in basis points of the original price. This is the genuine
            third-and-higher-order term — for a short bond and a 100 bp move
            it is a fraction of a basis point, which is why a residual of
            several basis points indicates a bad duration rather than a
            truncated Taylor series.
        convexity_contribution_bp: What the convexity term added over the
            first-order estimate, in basis points.
    """

    shock_bp: float
    exact: float
    duration_only: float
    duration_convexity: float
    residual_bp: float
    convexity_contribution_bp: float


def analytics(bond: Bond, settlement: date, ytm: float) -> Analytics:
    """Duration, DV01 and convexity at a given yield.

    Args:
        bond: The instrument.
        settlement: Valuation date.
        ytm: Nominal annual yield, compounded at the coupon frequency.
    """
    flows = cash_flows(bond, settlement)
    m = bond.frequency
    periodic = ytm / m

    price = present_value(flows, ytm, m)
    if price <= 0:
        raise ValueError(f"price must be positive to compute duration, got {price}")

    weighted_time = 0.0
    convexity_sum = 0.0
    for f in flows:
        discounted = f.amount / (1 + periodic) ** f.periods
        weighted_time += f.years * discounted
        convexity_sum += (
            f.years * (f.years + 1 / m) * f.amount / (1 + periodic) ** (f.periods + 2)
        )

    macaulay = weighted_time / price
    modified = macaulay / (1 + periodic)
    convexity = convexity_sum / price

    up = present_value(flows, ytm + _EFFECTIVE_BUMP, m)
    down = present_value(flows, ytm - _EFFECTIVE_BUMP, m)
    effective = (down - up) / (2 * price * _EFFECTIVE_BUMP)

    return Analytics(
        macaulay=macaulay,
        modified=modified,
        effective=effective,
        dv01=modified * price * _BUMP,
        convexity=convexity,
        naive_modified=macaulay / (1 + ytm),
    )


def scenario(bond: Bond, settlement: date, ytm: float, shock_bp: float) -> Scenario:
    """Reprice after a parallel yield shift, exactly and by approximation.

    Args:
        bond: The instrument.
        settlement: Valuation date.
        ytm: Starting nominal annual yield.
        shock_bp: Parallel shift in basis points. Positive is a rise.
    """
    flows = cash_flows(bond, settlement)
    m = bond.frequency
    price = present_value(flows, ytm, m)
    metrics = analytics(bond, settlement, ytm)

    delta = shock_bp / 10_000
    exact = present_value(flows, ytm + delta, m)
    first_order = price * (1 - metrics.modified * delta)
    second_order = price * (
        1 - metrics.modified * delta + 0.5 * metrics.convexity * delta**2
    )

    return Scenario(
        shock_bp=shock_bp,
        exact=exact,
        duration_only=first_order,
        duration_convexity=second_order,
        residual_bp=(second_order - exact) / price * 10_000,
        convexity_contribution_bp=(second_order - first_order) / price * 10_000,
    )
