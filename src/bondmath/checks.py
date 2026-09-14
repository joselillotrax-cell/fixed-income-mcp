"""Consistency bounds.

Every quantity here could be recomputed from the pricing functions, so these
checks add no new mathematics. What they add is a floor: relationships that
must hold for any correct answer, stated explicitly and verified on every
valuation.

The motivating case: a language model asked for the yield of a bond trading at
97.80% of par returned 4.01%. Rejecting that needs no iteration. A bond priced
below par yields more than its coupon, and more than its current yield of
4.09%. The answer was not close — it was outside the range of possible
answers, and two comparisons would have caught it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .analytics import Analytics
from .bond import Bond
from .pricing import Quote

__all__ = ["Check", "ConsistencyReport", "check_consistency"]

_PAR_TOLERANCE = 5e-5  # treat within 0.005% of par as "at par"
_DURATION_TOLERANCE = 0.01  # years, between analytic and effective duration


@dataclass(frozen=True)
class Check:
    """One bound and its outcome.

    Attributes:
        name: Short identifier for the bound.
        passed: Whether the relationship holds.
        detail: Human-readable statement of what was compared, including the
            actual figures, so a failure explains itself.
    """

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ConsistencyReport:
    """The full set of bounds for one valuation."""

    checks: tuple[Check, ...]

    @property
    def passed(self) -> bool:
        """True when every bound holds."""
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> tuple[Check, ...]:
        """Only the bounds that were violated."""
        return tuple(c for c in self.checks if not c.passed)

    def __str__(self) -> str:
        lines = []
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"[{mark}] {c.name}: {c.detail}")
        return "\n".join(lines)


def check_consistency(
    bond: Bond,
    settlement: date,
    quote: Quote,
    metrics: Analytics | None = None,
) -> ConsistencyReport:
    """Verify the relationships any correct valuation must satisfy.

    Args:
        bond: The instrument.
        settlement: Valuation date.
        quote: The valuation to check.
        metrics: Sensitivity measures. When supplied, the analytic duration is
            also checked against the convention-free effective duration.

    Returns:
        A report whose `passed` property is True only if every bound holds.
    """
    checks: list[Check] = []

    current_yield = bond.annual_coupon / quote.clean
    coupon = bond.coupon_rate
    ytm = quote.ytm
    par_gap = quote.clean / bond.face - 1

    # --- price, coupon and yield must be ordered consistently ---------------
    if abs(par_gap) < _PAR_TOLERANCE:
        ok = abs(ytm - coupon) < 1e-4
        checks.append(
            Check(
                "par_pricing",
                ok,
                f"trading at par, so YTM should equal the coupon: "
                f"YTM {ytm:.4%} vs coupon {coupon:.4%}",
            )
        )
    elif par_gap < 0:
        ok = coupon < current_yield < ytm
        checks.append(
            Check(
                "below_par_ordering",
                ok,
                f"trading below par at {quote.clean_pct:.4f}%, so "
                f"coupon < current yield < YTM must hold: "
                f"{coupon:.4%} < {current_yield:.4%} < {ytm:.4%}",
            )
        )
    else:
        ok = ytm < current_yield < coupon
        checks.append(
            Check(
                "above_par_ordering",
                ok,
                f"trading above par at {quote.clean_pct:.4f}%, so "
                f"YTM < current yield < coupon must hold: "
                f"{ytm:.4%} < {current_yield:.4%} < {coupon:.4%}",
            )
        )

    # --- dirty price must exceed clean price -------------------------------
    checks.append(
        Check(
            "accrued_sign",
            quote.accrued >= 0 and quote.dirty >= quote.clean,
            f"accrued interest is never negative: "
            f"clean {quote.clean:,.4f} + accrued {quote.accrued:,.4f} "
            f"= dirty {quote.dirty:,.4f}",
        )
    )

    # --- price must not exceed the undiscounted sum of remaining flows -----
    from .bond import cash_flows

    undiscounted = sum(f.amount for f in cash_flows(bond, settlement))
    checks.append(
        Check(
            "price_ceiling",
            quote.dirty <= undiscounted or ytm < 0,
            f"a positive yield cannot price above the undiscounted flows: "
            f"dirty {quote.dirty:,.2f} vs flows {undiscounted:,.2f}",
        )
    )

    # --- analytic duration must agree with the numerical one ---------------
    if metrics is not None:
        gap = abs(metrics.modified - metrics.effective)
        checks.append(
            Check(
                "duration_cross_check",
                gap < _DURATION_TOLERANCE,
                f"analytic modified duration {metrics.modified:.4f} vs "
                f"effective duration {metrics.effective:.4f} "
                f"(difference {gap:.4f} years)",
            )
        )

        checks.append(
            Check(
                "convexity_sign",
                metrics.convexity > 0,
                f"a bullet bond without optionality has positive convexity: "
                f"{metrics.convexity:.4f}",
            )
        )

    return ConsistencyReport(checks=tuple(checks))
