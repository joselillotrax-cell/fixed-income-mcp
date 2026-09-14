"""MCP server exposing bondmath as agent tools.

Design notes, since they are what make these tools usable by a model rather
than merely callable:

**Units live in the parameter names.** A field called `coupon_rate` invites
the question of whether 4% is `4` or `0.04`, and a model that guesses wrong
produces an answer off by a factor of a hundred while looking perfectly
reasonable. Every rate here is named `*_pct` and every price
`*_pct_of_face`, so there is nothing to guess.

**Constraints live in the schema, not in the error path.** Day counts and
payment frequencies are `Literal` types, so they arrive at the model as
enumerations. An invalid value is unrepresentable rather than merely rejected,
which removes a round-trip and a chance to guess again.

**Every valuation returns its own consistency checks.** The tools do not just
compute; they report whether the result satisfies the bounds any correct
answer must satisfy. A model reading the output sees the verdict, not only the
number.

**Durations arrive with their numerical cross-check.** `modified_duration`
comes alongside `effective_duration`, computed by central differences and free
of any annualisation convention. When they agree, the analytic figure is
trustworthy. `naive_modified_duration` is the figure the common wrong divisor
produces, included so the error can be named rather than silently avoided.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from . import __version__
from .analytics import analytics as _analytics
from .analytics import scenario as _scenario
from .bond import Bond, accrued_interest, cash_flows
from .checks import check_consistency as _check
from .daycount import DayCount
from .pricing import Quote, price_from_yield, yield_from_price

__all__ = ["server", "main"]

server = MCPServer(
    name="fixed-income",
    title="Fixed Income Analytics",
    version=__version__,
    instructions=(
        "Bond pricing and interest-rate sensitivity with built-in verification.\n\n"
        "Use these tools instead of computing bond figures yourself: yields, "
        "durations and accrued interest depend on day-count conventions and on "
        "compounding frequency, and small convention errors produce plausible "
        "but wrong numbers.\n\n"
        "Every result carries a `checks` list. If `all_checks_passed` is false, "
        "report the failure rather than the number. Modified duration arrives "
        "with an `effective_duration` computed numerically; if those two "
        "disagree by more than about 0.01 years, something is wrong with the "
        "inputs.\n\n"
        "All rates are percentages: pass 4.0 for a 4% coupon, not 0.04."
    ),
)

DayCountName = Literal["ACT/ACT ICMA", "30/360", "ACT/365", "ACT/360"]

_BASIS: dict[str, DayCount] = {
    "ACT/ACT ICMA": DayCount.ACT_ACT_ICMA,
    "30/360": DayCount.THIRTY_360,
    "ACT/365": DayCount.ACT_365,
    "ACT/360": DayCount.ACT_360,
}

# --------------------------------------------------------------------------
# Parameter types.
#
# Declared once and reused, so every tool presents the same contract for the
# same concept. The descriptions here are what a model actually reads when
# deciding how to fill a field.
# --------------------------------------------------------------------------

FaceValue = Annotated[
    float,
    Field(gt=0, description="Redemption amount in currency units, e.g. 1000."),
]

CouponPct = Annotated[
    float,
    Field(
        ge=0,
        description=(
            "Annual coupon rate as a PERCENTAGE. Pass 4.0 for a 4% coupon, "
            "not 0.04. Zero for a zero-coupon bond."
        ),
    ),
]

PaymentsPerYear = Annotated[
    Literal[1, 2, 4, 12],
    Field(
        description=(
            "Coupon payments per year: 1 annual, 2 semi-annual, 4 quarterly, "
            "12 monthly. Most government bonds pay semi-annually."
        )
    ),
]

IssueDate = Annotated[
    str,
    Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Issue (dated) date in ISO format, e.g. 2024-03-15.",
    ),
]

MaturityDate = Annotated[
    str,
    Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Redemption date in ISO format, e.g. 2029-03-15.",
    ),
]

SettlementDate = Annotated[
    str,
    Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description=(
            "Valuation date in ISO format. Must fall between issue and "
            "maturity, e.g. 2025-09-11."
        ),
    ),
]

CleanPricePct = Annotated[
    float,
    Field(
        gt=0,
        description=(
            "Quoted clean price as a PERCENTAGE OF FACE, excluding accrued "
            "interest. Pass 97.80 for a bond quoted at 97.80%. This is not a "
            "currency amount."
        ),
    ),
]

YtmPct = Annotated[
    float,
    Field(
        description=(
            "Nominal annual yield to maturity as a PERCENTAGE, compounded at "
            "the coupon frequency. Pass 4.6867 for 4.6867%."
        )
    ),
]

Basis = Annotated[
    DayCountName,
    Field(
        description=(
            "Day-count convention used for accrual. ACT/ACT ICMA is the "
            "standard for most government and corporate bonds."
        )
    ),
]

OptionalCleanPricePct = Annotated[
    float | None,
    Field(
        default=None,
        description=(
            "Quoted clean price as a percentage of face, e.g. 97.80. Supply "
            "either this or ytm_pct, never both."
        ),
    ),
]

OptionalYtmPct = Annotated[
    float | None,
    Field(
        default=None,
        description=(
            "Nominal annual yield as a percentage, e.g. 4.6867. Supply either "
            "this or clean_price_pct_of_face, never both."
        ),
    ),
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"{field} must be an ISO date like 2025-09-11, got {value!r}"
        ) from exc


def _build(
    face_value: float,
    coupon_rate_pct: float,
    payments_per_year: int,
    issue_date: str,
    maturity_date: str,
    day_count: str,
) -> Bond:
    if day_count not in _BASIS:
        raise ValueError(
            f"day_count must be one of {', '.join(_BASIS)}, got {day_count!r}"
        )
    return Bond(
        face=face_value,
        coupon_rate=coupon_rate_pct / 100,
        frequency=payments_per_year,
        issue=_parse_date(issue_date, "issue_date"),
        maturity=_parse_date(maturity_date, "maturity_date"),
        basis=_BASIS[day_count],
    )


def _checks_payload(report) -> dict[str, Any]:
    return {
        "all_checks_passed": report.passed,
        "checks": [
            {"name": c.name, "passed": c.passed, "detail": c.detail}
            for c in report.checks
        ],
    }


def _quote_payload(bond: Bond, settlement: date, quote: Quote) -> dict[str, Any]:
    accrual = accrued_interest(bond, settlement)
    return {
        "clean_price": round(quote.clean, 6),
        "clean_price_pct_of_face": round(quote.clean_pct, 6),
        "accrued_interest": round(quote.accrued, 6),
        "dirty_price": round(quote.dirty, 6),
        "dirty_price_pct_of_face": round(quote.dirty_pct, 6),
        "ytm_pct": round(quote.ytm * 100, 6),
        "effective_annual_yield_pct": round(quote.effective_annual_yield * 100, 6),
        "current_yield_pct": round(bond.annual_coupon / quote.clean * 100, 6),
        "accrual": {
            "period_start": accrual.period_start.isoformat(),
            "period_end": accrual.period_end.isoformat(),
            "days_elapsed": accrual.days_elapsed,
            "days_in_period": accrual.days_in_period,
            "fraction_of_period": round(accrual.fraction, 9),
        },
    }


def _resolve(
    bond: Bond,
    settlement: date,
    clean_price_pct_of_face: float | None,
    ytm_pct: float | None,
) -> Quote:
    """Value the bond from whichever of price or yield was supplied."""
    if (clean_price_pct_of_face is None) == (ytm_pct is None):
        raise ValueError(
            "supply exactly one of clean_price_pct_of_face or ytm_pct, not both "
            "and not neither"
        )
    if ytm_pct is not None:
        return price_from_yield(bond, settlement, ytm_pct / 100)
    clean = clean_price_pct_of_face / 100 * bond.face  # type: ignore[operator]
    return yield_from_price(bond, settlement, clean)


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


@server.tool(title="Price a bond from its yield")
def price_from_yield_tool(
    face_value: FaceValue,
    coupon_rate_pct: CouponPct,
    payments_per_year: PaymentsPerYear,
    issue_date: IssueDate,
    maturity_date: MaturityDate,
    settlement_date: SettlementDate,
    ytm_pct: YtmPct,
    day_count: Basis = "ACT/ACT ICMA",
) -> dict[str, Any]:
    """Value a fixed-coupon bond at a given yield to maturity.

    Returns clean and dirty price, accrued interest with the day counts behind
    it, and the consistency checks the valuation satisfies.

    Use this when the yield is known and the price is wanted. For the reverse,
    use yield_from_price_tool.
    """
    bond = _build(
        face_value, coupon_rate_pct, payments_per_year,
        issue_date, maturity_date, day_count,
    )
    settle = _parse_date(settlement_date, "settlement_date")
    quote = price_from_yield(bond, settle, ytm_pct / 100)
    metrics = _analytics(bond, settle, quote.ytm)
    return {
        **_quote_payload(bond, settle, quote),
        **_checks_payload(_check(bond, settle, quote, metrics)),
    }


@server.tool(title="Solve a bond's yield from its price")
def yield_from_price_tool(
    face_value: FaceValue,
    coupon_rate_pct: CouponPct,
    payments_per_year: PaymentsPerYear,
    issue_date: IssueDate,
    maturity_date: MaturityDate,
    settlement_date: SettlementDate,
    clean_price_pct_of_face: CleanPricePct,
    day_count: Basis = "ACT/ACT ICMA",
) -> dict[str, Any]:
    """Solve for the yield to maturity implied by a market price.

    Solved by bisection, which cannot converge on a value that satisfies the
    iteration without solving the pricing equation. The returned checks verify
    the ordering of coupon, current yield and yield to maturity, which alone
    rules out a large class of wrong answers.

    This is the usual starting point: market prices are observable, yields are
    derived.
    """
    bond = _build(
        face_value, coupon_rate_pct, payments_per_year,
        issue_date, maturity_date, day_count,
    )
    settle = _parse_date(settlement_date, "settlement_date")
    clean = clean_price_pct_of_face / 100 * bond.face
    quote = yield_from_price(bond, settle, clean)
    metrics = _analytics(bond, settle, quote.ytm)
    return {
        **_quote_payload(bond, settle, quote),
        **_checks_payload(_check(bond, settle, quote, metrics)),
    }


@server.tool(title="Measure interest-rate sensitivity")
def bond_analytics_tool(
    face_value: FaceValue,
    coupon_rate_pct: CouponPct,
    payments_per_year: PaymentsPerYear,
    issue_date: IssueDate,
    maturity_date: MaturityDate,
    settlement_date: SettlementDate,
    clean_price_pct_of_face: OptionalCleanPricePct = None,
    ytm_pct: OptionalYtmPct = None,
    day_count: Basis = "ACT/ACT ICMA",
) -> dict[str, Any]:
    """Macaulay duration, modified duration, DV01 and convexity.

    Supply either a price or a yield, not both.

    `modified_duration_years` divides Macaulay duration by one plus the
    PERIODIC yield. `naive_modified_duration_years` shows what dividing by one
    plus the ANNUAL yield would give — the standard error on any bond paying
    more than once a year. `effective_duration_years` is computed numerically
    by central differences and uses no annualisation, so it arbitrates between
    them. Report the modified figure; the other two are there to prove it.
    """
    bond = _build(
        face_value, coupon_rate_pct, payments_per_year,
        issue_date, maturity_date, day_count,
    )
    settle = _parse_date(settlement_date, "settlement_date")
    quote = _resolve(bond, settle, clean_price_pct_of_face, ytm_pct)
    m = _analytics(bond, settle, quote.ytm)

    return {
        "ytm_pct": round(quote.ytm * 100, 6),
        "dirty_price": round(quote.dirty, 6),
        "macaulay_duration_years": round(m.macaulay, 6),
        "modified_duration_years": round(m.modified, 6),
        "effective_duration_years": round(m.effective, 6),
        "naive_modified_duration_years": round(m.naive_modified, 6),
        "annualisation_gap_pct": round(m.annualisation_gap * 100, 4),
        "dv01": round(m.dv01, 6),
        "convexity_years_squared": round(m.convexity, 6),
        "duration_note": (
            "modified_duration_years is the correct figure. "
            "naive_modified_duration_years shows what the wrong divisor gives; "
            "it is reported only so the error can be recognised. "
            "effective_duration_years is the convention-free numerical check."
        ),
        **_checks_payload(_check(bond, settle, quote, m)),
    }


@server.tool(title="List a bond's remaining cash flows")
def cashflow_schedule_tool(
    face_value: FaceValue,
    coupon_rate_pct: CouponPct,
    payments_per_year: PaymentsPerYear,
    issue_date: IssueDate,
    maturity_date: MaturityDate,
    settlement_date: SettlementDate,
    ytm_pct: OptionalYtmPct = None,
    day_count: Basis = "ACT/ACT ICMA",
) -> dict[str, Any]:
    """Remaining payments, with discount factors when a yield is supplied.

    The first flow sits a fraction of a period away rather than a whole one.
    That exponent is what makes an ACT/ACT ICMA yield differ from a naive one,
    so it is reported explicitly as periods_from_settlement.
    """
    bond = _build(
        face_value, coupon_rate_pct, payments_per_year,
        issue_date, maturity_date, day_count,
    )
    settle = _parse_date(settlement_date, "settlement_date")
    flows = cash_flows(bond, settle)

    rows: list[dict[str, Any]] = []
    total_pv = 0.0
    for f in flows:
        row: dict[str, Any] = {
            "date": f.date.isoformat(),
            "amount": round(f.amount, 6),
            "periods_from_settlement": round(f.periods, 9),
            "years_from_settlement": round(f.years, 9),
        }
        if ytm_pct is not None:
            factor = 1 / (1 + ytm_pct / 100 / bond.frequency) ** f.periods
            pv = f.amount * factor
            total_pv += pv
            row["discount_factor"] = round(factor, 9)
            row["present_value"] = round(pv, 6)
        rows.append(row)

    out: dict[str, Any] = {
        "coupon_amount": round(bond.coupon, 6),
        "flow_count": len(rows),
        "flows": rows,
    }
    if ytm_pct is not None:
        out["total_present_value"] = round(total_pv, 6)
    return out


@server.tool(title="Reprice after a parallel yield shift")
def scenario_shock_tool(
    face_value: FaceValue,
    coupon_rate_pct: CouponPct,
    payments_per_year: PaymentsPerYear,
    issue_date: IssueDate,
    maturity_date: MaturityDate,
    settlement_date: SettlementDate,
    shock_bp: Annotated[
        float,
        Field(
            description=(
                "Parallel shift in the yield curve, in BASIS POINTS. Positive "
                "is a rise. Pass 100 for a one-percentage-point rise."
            )
        ),
    ],
    clean_price_pct_of_face: OptionalCleanPricePct = None,
    ytm_pct: OptionalYtmPct = None,
    day_count: Basis = "ACT/ACT ICMA",
) -> dict[str, Any]:
    """Compare an exact repricing against first- and second-order estimates.

    `residual_bp` is the genuine third-and-higher-order term. For a short bond
    and a 100 bp move it is a fraction of a basis point. A residual of several
    basis points means the duration is wrong, not that the Taylor expansion
    was truncated — a distinction worth making explicitly, because attributing
    such a gap to higher-order terms is a common and confident mistake.

    Supply either a price or a yield, not both.
    """
    bond = _build(
        face_value, coupon_rate_pct, payments_per_year,
        issue_date, maturity_date, day_count,
    )
    settle = _parse_date(settlement_date, "settlement_date")
    quote = _resolve(bond, settle, clean_price_pct_of_face, ytm_pct)
    s = _scenario(bond, settle, quote.ytm, shock_bp)

    return {
        "starting_ytm_pct": round(quote.ytm * 100, 6),
        "shocked_ytm_pct": round(quote.ytm * 100 + shock_bp / 100, 6),
        "starting_dirty_price": round(quote.dirty, 6),
        "exact_repricing": round(s.exact, 6),
        "duration_only_estimate": round(s.duration_only, 6),
        "duration_plus_convexity_estimate": round(s.duration_convexity, 6),
        "convexity_contribution_bp": round(s.convexity_contribution_bp, 4),
        "residual_bp": round(s.residual_bp, 4),
        "residual_note": (
            "residual_bp is the true higher-order term left over after the "
            "second-order approximation. Values above roughly 1 bp on a short "
            "bond indicate an incorrect duration rather than a truncated "
            "expansion."
        ),
        "price_change_pct": round((s.exact - quote.dirty) / quote.dirty * 100, 6),
    }


@server.tool(title="Check a bond valuation for consistency")
def check_consistency_tool(
    face_value: FaceValue,
    coupon_rate_pct: CouponPct,
    payments_per_year: PaymentsPerYear,
    issue_date: IssueDate,
    maturity_date: MaturityDate,
    settlement_date: SettlementDate,
    clean_price_pct_of_face: CleanPricePct,
    claimed_ytm_pct: Annotated[
        float,
        Field(
            description=(
                "The yield to audit, as a percentage. This is the figure to "
                "be tested, not one to compute."
            )
        ),
    ],
    day_count: Basis = "ACT/ACT ICMA",
) -> dict[str, Any]:
    """Test a yield someone has already produced against the required bounds.

    Use this to audit a figure rather than compute one — for example when
    checking work, or when a yield appears in a document and its plausibility
    matters.

    A bond priced below par must yield more than its coupon and more than its
    current yield; above par, the ordering reverses. These bounds need no
    iteration and reject a whole class of wrong answers immediately. The
    correctly solved yield is returned too, so the size of any error is
    visible.
    """
    bond = _build(
        face_value, coupon_rate_pct, payments_per_year,
        issue_date, maturity_date, day_count,
    )
    settle = _parse_date(settlement_date, "settlement_date")
    clean = clean_price_pct_of_face / 100 * bond.face

    solved = yield_from_price(bond, settle, clean)
    claimed = Quote(
        clean=solved.clean,
        dirty=solved.dirty,
        accrued=solved.accrued,
        clean_pct=solved.clean_pct,
        dirty_pct=solved.dirty_pct,
        ytm=claimed_ytm_pct / 100,
        effective_annual_yield=(
            (1 + claimed_ytm_pct / 100 / bond.frequency) ** bond.frequency - 1
        ),
    )

    report = _check(bond, settle, claimed)
    error_bp = (claimed_ytm_pct - solved.ytm * 100) * 100

    return {
        "claimed_ytm_pct": claimed_ytm_pct,
        "correct_ytm_pct": round(solved.ytm * 100, 6),
        "error_bp": round(error_bp, 2),
        "current_yield_pct": round(bond.annual_coupon / solved.clean * 100, 6),
        "coupon_pct": coupon_rate_pct,
        **_checks_payload(report),
    }


def main() -> None:
    """Run the server over stdio."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
