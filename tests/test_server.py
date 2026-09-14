"""Tests for the MCP tool layer.

These exercise the tool functions directly rather than over a transport. What
they verify is the contract an agent depends on: units are interpreted as
documented, every valuation carries its verdict, and bad input fails with a
message that says how to fix it.
"""

from __future__ import annotations

import pytest

from bondmath.server import (
    bond_analytics_tool,
    cashflow_schedule_tool,
    check_consistency_tool,
    price_from_yield_tool,
    scenario_shock_tool,
    server,
    yield_from_price_tool,
)

REFERENCE = dict(
    face_value=1000.0,
    coupon_rate_pct=4.0,
    payments_per_year=2,
    issue_date="2024-03-15",
    maturity_date="2029-03-15",
    settlement_date="2025-09-11",
)


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_all_six_tools_are_registered():
    names = {t.name for t in await server.list_tools()}
    assert len(names) == 6
    for expected in (
        "price_from_yield_tool",
        "yield_from_price_tool",
        "bond_analytics_tool",
        "cashflow_schedule_tool",
        "scenario_shock_tool",
        "check_consistency_tool",
    ):
        assert expected in names


@pytest.mark.anyio
async def test_every_tool_documents_itself():
    """Descriptions become the model's only guide to calling these correctly."""
    for tool in await server.list_tools():
        assert tool.description
        assert len(tool.description) > 80


# --------------------------------------------------------------------------
# pricing tools
# --------------------------------------------------------------------------


def test_yield_from_price_reference():
    out = yield_from_price_tool(**REFERENCE, clean_price_pct_of_face=97.80)
    assert out["dirty_price"] == pytest.approx(997.5652, abs=1e-3)
    assert out["ytm_pct"] == pytest.approx(4.6867, abs=1e-3)
    assert out["current_yield_pct"] == pytest.approx(4.0900, abs=1e-3)
    assert out["all_checks_passed"] is True


def test_accrual_detail_is_reported():
    out = yield_from_price_tool(**REFERENCE, clean_price_pct_of_face=97.80)
    acc = out["accrual"]
    assert acc["period_start"] == "2025-03-15"
    assert acc["period_end"] == "2025-09-15"
    assert acc["days_elapsed"] == 180
    assert acc["days_in_period"] == 184


def test_price_and_yield_tools_are_inverse():
    priced = price_from_yield_tool(**REFERENCE, ytm_pct=4.6867)
    assert priced["clean_price_pct_of_face"] == pytest.approx(97.80, abs=1e-3)


def test_rates_are_read_as_percentages_not_decimals():
    """A 4.0 coupon must mean 4%, never 400%."""
    out = yield_from_price_tool(**REFERENCE, clean_price_pct_of_face=100.0)
    assert out["ytm_pct"] == pytest.approx(4.0, abs=0.01)


# --------------------------------------------------------------------------
# analytics tool — the differentiator
# --------------------------------------------------------------------------


def test_analytics_reports_correct_and_naive_duration():
    out = bond_analytics_tool(**REFERENCE, clean_price_pct_of_face=97.80)
    assert out["modified_duration_years"] == pytest.approx(3.1686, abs=1e-3)
    assert out["naive_modified_duration_years"] == pytest.approx(3.0976, abs=1e-3)
    assert out["effective_duration_years"] == pytest.approx(3.1694, abs=1e-3)
    assert out["dv01"] == pytest.approx(0.3161, abs=1e-3)


def test_effective_duration_backs_the_analytic_one_not_the_naive_one():
    out = bond_analytics_tool(**REFERENCE, clean_price_pct_of_face=97.80)
    eff = out["effective_duration_years"]
    assert abs(out["modified_duration_years"] - eff) < 0.01
    assert abs(out["naive_modified_duration_years"] - eff) > 0.05


def test_analytics_explains_the_two_duration_figures():
    out = bond_analytics_tool(**REFERENCE, clean_price_pct_of_face=97.80)
    assert "correct figure" in out["duration_note"]


def test_analytics_requires_exactly_one_of_price_or_yield():
    with pytest.raises(ValueError, match="exactly one"):
        bond_analytics_tool(**REFERENCE)
    with pytest.raises(ValueError, match="exactly one"):
        bond_analytics_tool(**REFERENCE, clean_price_pct_of_face=97.8, ytm_pct=4.7)


# --------------------------------------------------------------------------
# cash flows
# --------------------------------------------------------------------------


def test_cashflow_schedule_without_yield_omits_discounting():
    out = cashflow_schedule_tool(**REFERENCE)
    assert out["flow_count"] == 8
    assert out["coupon_amount"] == pytest.approx(20.0)
    assert out["flows"][-1]["amount"] == pytest.approx(1020.0)
    assert "discount_factor" not in out["flows"][0]


def test_cashflow_schedule_with_yield_sums_to_the_dirty_price():
    out = cashflow_schedule_tool(**REFERENCE, ytm_pct=4.6867)
    assert out["total_present_value"] == pytest.approx(997.5652, abs=1e-2)
    assert out["flows"][0]["discount_factor"] < 1


# --------------------------------------------------------------------------
# scenario
# --------------------------------------------------------------------------


def test_scenario_residual_is_a_fraction_of_a_basis_point():
    out = scenario_shock_tool(
        **REFERENCE, clean_price_pct_of_face=97.80, shock_bp=100
    )
    assert abs(out["residual_bp"]) < 1.0
    assert out["exact_repricing"] == pytest.approx(966.55, abs=0.05)
    assert out["duration_plus_convexity_estimate"] > out["duration_only_estimate"]


def test_scenario_names_what_a_large_residual_means():
    out = scenario_shock_tool(
        **REFERENCE, clean_price_pct_of_face=97.80, shock_bp=100
    )
    assert "incorrect duration" in out["residual_note"]


def test_falling_yields_raise_the_price():
    out = scenario_shock_tool(
        **REFERENCE, clean_price_pct_of_face=97.80, shock_bp=-100
    )
    assert out["price_change_pct"] > 0


# --------------------------------------------------------------------------
# audit tool
# --------------------------------------------------------------------------


def test_audit_rejects_the_impossible_yield():
    """The 4.01% case, reproduced as a regression test."""
    out = check_consistency_tool(
        **REFERENCE, clean_price_pct_of_face=97.80, claimed_ytm_pct=4.01
    )
    assert out["all_checks_passed"] is False
    failed = [c["name"] for c in out["checks"] if not c["passed"]]
    assert "below_par_ordering" in failed
    assert out["correct_ytm_pct"] == pytest.approx(4.6867, abs=1e-3)
    assert out["error_bp"] == pytest.approx(-67.67, abs=1.0)


def test_audit_accepts_the_correct_yield():
    out = check_consistency_tool(
        **REFERENCE, clean_price_pct_of_face=97.80, claimed_ytm_pct=4.6867
    )
    assert out["all_checks_passed"] is True
    assert abs(out["error_bp"]) < 0.1


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


def test_bad_date_says_what_the_format_should_be():
    with pytest.raises(ValueError, match="ISO date like"):
        yield_from_price_tool(
            **{**REFERENCE, "settlement_date": "11/09/2025"},
            clean_price_pct_of_face=97.80,
        )


def test_bad_day_count_lists_the_valid_options():
    with pytest.raises(ValueError, match="ACT/ACT ICMA"):
        yield_from_price_tool(
            **REFERENCE, clean_price_pct_of_face=97.80, day_count="ACT/ACT ISDA"
        )


def test_settlement_after_maturity_is_rejected():
    with pytest.raises(ValueError, match="outside the schedule"):
        yield_from_price_tool(
            **{**REFERENCE, "settlement_date": "2030-01-01"},
            clean_price_pct_of_face=97.80,
        )


@pytest.fixture
def anyio_backend():
    return "asyncio"
