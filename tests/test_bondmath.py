"""Tests for bondmath.

The reference instrument throughout is the bond that motivated the project: a
Spanish government bond, 4% semi-annual, issued 15/03/2024, maturing
15/03/2029, quoted at 97.80% clean on 11/09/2025 under ACT/ACT ICMA.

Its published values, verified independently by central differences:

    accrued interest      19.5652
    dirty price          997.5652   (99.7565%)
    YTM                    4.6867%
    Macaulay duration      3.2428 years
    modified duration      3.1686 years
    DV01                   0.3161 per bp
    convexity             12.1678 years^2

Run with:  python -m pytest -q
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from bondmath import (
    Bond,
    DayCount,
    accrued_interest,
    add_months,
    analytics,
    cash_flows,
    check_consistency,
    coupon_schedule,
    present_value,
    price_from_yield,
    scenario,
    yield_from_price,
)

SETTLE = date(2025, 9, 11)
CLEAN = 978.00


@pytest.fixture
def bond() -> Bond:
    return Bond(
        face=1000.0,
        coupon_rate=0.04,
        frequency=2,
        issue=date(2024, 3, 15),
        maturity=date(2029, 3, 15),
        basis=DayCount.ACT_ACT_ICMA,
    )


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------


def test_add_months_clamps_to_month_end():
    assert add_months(date(2025, 3, 31), -1) == date(2025, 2, 28)
    assert add_months(date(2024, 3, 31), -1) == date(2024, 2, 29)  # leap year
    assert add_months(date(2025, 1, 15), 13) == date(2026, 2, 15)


def test_schedule_runs_backward_from_maturity(bond: Bond):
    schedule = bond.schedule()
    assert schedule[0] == bond.issue
    assert schedule[-1] == bond.maturity
    assert schedule == sorted(schedule)
    # 5 years semi-annual, issue coinciding with a coupon date
    assert len(schedule) == 11


def test_schedule_handles_irregular_first_period():
    odd = Bond(
        face=1000,
        coupon_rate=0.03,
        frequency=2,
        issue=date(2024, 5, 2),  # off-cycle
        maturity=date(2029, 3, 15),
    )
    schedule = odd.schedule()
    assert schedule[0] == date(2024, 5, 2)
    assert schedule[1] == date(2024, 9, 15)  # anchored to maturity day
    assert schedule[-1] == date(2029, 3, 15)


def test_schedule_rejects_bad_frequency():
    with pytest.raises(ValueError, match="divide 12"):
        coupon_schedule(date(2024, 1, 1), date(2029, 1, 1), 5)


# --------------------------------------------------------------------------
# accrual
# --------------------------------------------------------------------------


def test_accrued_interest_reference(bond: Bond):
    acc = accrued_interest(bond, SETTLE)
    assert acc.period_start == date(2025, 3, 15)
    assert acc.period_end == date(2025, 9, 15)
    assert acc.days_elapsed == 180
    assert acc.days_in_period == 184
    assert acc.amount == pytest.approx(19.5652, abs=1e-4)


def test_accrual_is_zero_on_a_coupon_date(bond: Bond):
    assert accrued_interest(bond, date(2025, 3, 15)).amount == pytest.approx(0.0)


def test_accrual_reaches_a_full_coupon_just_before_the_next(bond: Bond):
    acc = accrued_interest(bond, date(2025, 9, 14))
    assert acc.amount < bond.coupon
    assert acc.amount == pytest.approx(bond.coupon * 183 / 184, abs=1e-9)


def test_thirty_360_differs_from_act_act(bond: Bond):
    thirty = Bond(**{**bond.__dict__, "basis": DayCount.THIRTY_360})
    assert accrued_interest(thirty, SETTLE).amount != pytest.approx(
        accrued_interest(bond, SETTLE).amount, abs=1e-6
    )


def test_settlement_outside_schedule_is_rejected(bond: Bond):
    with pytest.raises(ValueError, match="outside the schedule"):
        accrued_interest(bond, date(2030, 1, 1))


# --------------------------------------------------------------------------
# cash flows
# --------------------------------------------------------------------------


def test_cash_flows_count_and_final_payment(bond: Bond):
    flows = cash_flows(bond, SETTLE)
    assert len(flows) == 8
    assert flows[-1].date == bond.maturity
    assert flows[-1].amount == pytest.approx(bond.coupon + bond.face)
    assert all(f.amount == pytest.approx(bond.coupon) for f in flows[:-1])


def test_first_flow_sits_at_a_fraction_of_a_period(bond: Bond):
    flows = cash_flows(bond, SETTLE)
    assert flows[0].periods == pytest.approx(1 - 180 / 184, abs=1e-9)
    assert flows[1].periods == pytest.approx(flows[0].periods + 1)
    assert flows[0].years == pytest.approx(flows[0].periods / 2)


# --------------------------------------------------------------------------
# pricing
# --------------------------------------------------------------------------


def test_yield_from_price_reference(bond: Bond):
    quote = yield_from_price(bond, SETTLE, CLEAN)
    assert quote.accrued == pytest.approx(19.5652, abs=1e-4)
    assert quote.dirty == pytest.approx(997.5652, abs=1e-4)
    assert quote.dirty_pct == pytest.approx(99.7565, abs=1e-4)
    assert quote.ytm == pytest.approx(0.046867, abs=1e-6)
    assert quote.effective_annual_yield == pytest.approx(0.047416, abs=1e-6)


def test_price_and_yield_are_inverse(bond: Bond):
    quote = yield_from_price(bond, SETTLE, CLEAN)
    back = price_from_yield(bond, SETTLE, quote.ytm)
    assert back.clean == pytest.approx(CLEAN, abs=1e-6)
    assert back.dirty == pytest.approx(quote.dirty, abs=1e-6)


def test_par_bond_yields_its_coupon(bond: Bond):
    quote = yield_from_price(bond, date(2025, 3, 15), 1000.0)
    assert quote.ytm == pytest.approx(bond.coupon_rate, abs=1e-6)


def test_present_value_is_decreasing_in_yield(bond: Bond):
    flows = cash_flows(bond, SETTLE)
    prices = [present_value(flows, y, 2) for y in (0.02, 0.04, 0.06, 0.08)]
    assert prices == sorted(prices, reverse=True)


def test_price_above_the_undiscounted_total_still_solves(bond: Bond):
    """A price above the sum of the flows is reachable — at a negative yield.

    Worth pinning: the intuition that the undiscounted total caps the price
    only holds for non-negative yields, and the solver searches below zero.
    """
    flows_total = sum(f.amount for f in cash_flows(bond, SETTLE))
    quote = yield_from_price(bond, SETTLE, flows_total * 2)
    assert quote.ytm < 0
    assert price_from_yield(bond, SETTLE, quote.ytm).clean == pytest.approx(
        flows_total * 2, abs=1e-6
    )


def test_unreachable_price_is_rejected(bond: Bond):
    """Above the price at the search floor there is no solution to find."""
    with pytest.raises(ValueError, match="no yield above"):
        yield_from_price(bond, SETTLE, 1e9)


# --------------------------------------------------------------------------
# analytics — the core of the project
# --------------------------------------------------------------------------


def test_duration_reference_values(bond: Bond):
    ytm = yield_from_price(bond, SETTLE, CLEAN).ytm
    a = analytics(bond, SETTLE, ytm)
    assert a.macaulay == pytest.approx(3.2428, abs=1e-4)
    assert a.modified == pytest.approx(3.1686, abs=1e-4)
    assert a.dv01 == pytest.approx(0.3161, abs=1e-4)
    assert a.convexity == pytest.approx(12.1678, abs=1e-3)


def test_analytic_duration_matches_effective_duration(bond: Bond):
    """The central guarantee: the closed form agrees with the numerical one."""
    ytm = yield_from_price(bond, SETTLE, CLEAN).ytm
    a = analytics(bond, SETTLE, ytm)
    assert abs(a.modified - a.effective) < 0.005


def test_naive_divisor_is_the_one_that_disagrees(bond: Bond):
    """The naive figure must sit measurably further from the numerical truth.

    This is the regression test for the bug the project documents: if someone
    ever swaps the divisor, `modified` drifts toward `naive_modified` and this
    test fails.
    """
    ytm = yield_from_price(bond, SETTLE, CLEAN).ytm
    a = analytics(bond, SETTLE, ytm)
    assert a.naive_modified == pytest.approx(3.0976, abs=1e-4)
    assert abs(a.naive_modified - a.effective) > 10 * abs(a.modified - a.effective)
    assert a.annualisation_gap == pytest.approx(0.0229, abs=1e-3)


def test_annual_bond_has_no_annualisation_gap():
    annual = Bond(
        face=1000,
        coupon_rate=0.04,
        frequency=1,
        issue=date(2024, 3, 15),
        maturity=date(2029, 3, 15),
    )
    ytm = yield_from_price(annual, SETTLE, CLEAN).ytm
    a = analytics(annual, SETTLE, ytm)
    assert a.modified == pytest.approx(a.naive_modified, abs=1e-12)
    assert a.annualisation_gap == pytest.approx(0.0, abs=1e-12)


def test_zero_coupon_macaulay_equals_time_to_maturity():
    zero = Bond(
        face=1000,
        coupon_rate=0.0,
        frequency=1,
        issue=date(2024, 1, 1),
        maturity=date(2029, 1, 1),
    )
    settle = date(2025, 1, 1)
    a = analytics(zero, settle, 0.05)
    assert a.macaulay == pytest.approx(4.0, abs=1e-9)


def test_duration_falls_as_yield_rises(bond: Bond):
    low = analytics(bond, SETTLE, 0.02).modified
    high = analytics(bond, SETTLE, 0.08).modified
    assert high < low


# --------------------------------------------------------------------------
# scenario
# --------------------------------------------------------------------------


def test_second_order_residual_is_under_one_basis_point(bond: Bond):
    """The claim at the centre of the write-up.

    For a bond of this maturity and a 100 bp move, the genuine third-order
    term is a fraction of a basis point — so a residual of several basis
    points signals a bad duration, not a truncated expansion.
    """
    ytm = yield_from_price(bond, SETTLE, CLEAN).ytm
    s = scenario(bond, SETTLE, ytm, 100)
    assert abs(s.residual_bp) < 1.0
    assert s.exact == pytest.approx(966.55, abs=0.01)


def test_convexity_always_helps_the_holder(bond: Bond):
    """Positive convexity cushions a sell-off and amplifies a rally."""
    ytm = yield_from_price(bond, SETTLE, CLEAN).ytm
    for shock in (-200, -100, 100, 200):
        s = scenario(bond, SETTLE, ytm, shock)
        assert s.duration_convexity > s.duration_only


def test_naive_duration_would_misprice_by_several_basis_points(bond: Bond):
    """Quantifies the cost of the bug, and pins the number in the write-up."""
    ytm = yield_from_price(bond, SETTLE, CLEAN).ytm
    a = analytics(bond, SETTLE, ytm)
    s = scenario(bond, SETTLE, ytm, 100)
    price = yield_from_price(bond, SETTLE, CLEAN).dirty

    delta = 0.01
    naive_price = price * (
        1 - a.naive_modified * delta + 0.5 * a.convexity * delta**2
    )
    naive_error_bp = (naive_price - s.exact) / price * 10_000

    assert naive_error_bp == pytest.approx(7.18, abs=0.1)
    assert abs(naive_error_bp) > 50 * abs(s.residual_bp)


# --------------------------------------------------------------------------
# consistency bounds
# --------------------------------------------------------------------------


def test_reference_valuation_passes_every_bound(bond: Bond):
    quote = yield_from_price(bond, SETTLE, CLEAN)
    a = analytics(bond, SETTLE, quote.ytm)
    report = check_consistency(bond, SETTLE, quote, a)
    assert report.passed, str(report)


def test_below_par_ordering_is_enforced(bond: Bond):
    """The 4.01% case: reject a yield below the current yield on a discount bond."""
    quote = yield_from_price(bond, SETTLE, CLEAN)
    impossible = type(quote)(
        clean=quote.clean,
        dirty=quote.dirty,
        accrued=quote.accrued,
        clean_pct=quote.clean_pct,
        dirty_pct=quote.dirty_pct,
        ytm=0.0401,
        effective_annual_yield=0.0405,
    )
    report = check_consistency(bond, SETTLE, impossible)
    assert not report.passed
    assert any(c.name == "below_par_ordering" for c in report.failures)


def test_above_par_ordering_is_enforced(bond: Bond):
    quote = yield_from_price(bond, SETTLE, 1050.0)
    report = check_consistency(bond, SETTLE, quote)
    assert report.passed, str(report)
    assert quote.ytm < bond.coupon_rate


def test_report_renders_readable_lines(bond: Bond):
    quote = yield_from_price(bond, SETTLE, CLEAN)
    text = str(check_consistency(bond, SETTLE, quote))
    assert "[PASS]" in text
    assert "below_par_ordering" in text


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"face": -1000}, "face must be positive"),
        ({"coupon_rate": -0.01}, "coupon_rate must not be negative"),
        ({"frequency": 5}, "divisor of 12"),
        ({"maturity": date(2023, 1, 1)}, "must be after issue"),
    ],
)
def test_bond_rejects_invalid_input(kwargs, message):
    base = dict(
        face=1000.0,
        coupon_rate=0.04,
        frequency=2,
        issue=date(2024, 3, 15),
        maturity=date(2029, 3, 15),
    )
    with pytest.raises(ValueError, match=message):
        Bond(**{**base, **kwargs})


def test_all_conventions_produce_finite_yields(bond: Bond):
    for basis in DayCount:
        variant = Bond(**{**bond.__dict__, "basis": basis})
        quote = yield_from_price(variant, SETTLE, CLEAN)
        assert math.isfinite(quote.ytm)
        assert 0.03 < quote.ytm < 0.07
