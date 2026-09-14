# fixed-income-mcp

Fixed-income analytics for AI agents, with the checks that catch the errors
models actually make.

Prices fixed-coupon bonds and measures their interest-rate sensitivity. Two
things separate it from a calculator: every analytic duration is validated
against a convention-free numerical one, and every valuation is tested against
the bounds a correct answer has to satisfy.

Pure standard library. No dependencies.

---

## Why

A language model asked for the modified duration of an ordinary bond returned
3.0942 years. The correct value is 3.1686. It had divided Macaulay duration by
one plus the *annual* yield instead of one plus the *periodic* yield — the
wrong divisor for a bond paying twice a year.

Then it did something more interesting. Its own duration-plus-convexity
approximation stopped matching an exact repricing, and it explained the gap as
"third-order terms of the Taylor series." For a bond of that maturity and a
100 bp move, the genuine third-order term is 0.09 bp. The residual it was
explaining came from its own miscalculated duration, handed back as theory.

A separate session returned a 4.01% yield for a bond trading at 97.80% of par
— below the 4.09% current yield, and therefore outside the range of possible
answers.

The full write-up is at
[joselillotrax-cell/bond-desk](https://github.com/joselillotrax-cell/bond-desk).

This library is the answer to it: give the agent a tool that does the
arithmetic correctly, and have the tool say so out loud.

## Design

**Bisection, not Newton-Raphson.** Newton converges faster but can diverge, or
settle on a value that satisfies the iteration without solving the pricing
equation. That is exactly the 4.01% failure. Bisection cannot do it.

**Effective duration ships alongside the analytic one.** Central differences
on the pricing function involve no annualisation, so the numerical figure
cannot inherit a convention error. When the two disagree, the analytic one is
wrong.

**The wrong answer is computed on purpose.** `Analytics.naive_modified`
carries the figure the bad divisor produces. Unusual for a library, but it is
what lets the tools flag the error rather than silently avoid it.

## Use

```python
from datetime import date
from bondmath import Bond, DayCount, yield_from_price, analytics, check_consistency

bond = Bond(
    face=1000,
    coupon_rate=0.04,
    frequency=2,
    issue=date(2024, 3, 15),
    maturity=date(2029, 3, 15),
    basis=DayCount.ACT_ACT_ICMA,
)
settle = date(2025, 9, 11)

quote = yield_from_price(bond, settle, 978.00)   # 97.80% of par
print(f"{quote.dirty:.4f}")          # 997.5652
print(f"{quote.ytm:.4%}")            # 4.6867%

metrics = analytics(bond, settle, quote.ytm)
print(f"{metrics.modified:.4f}")     # 3.1686  <- correct
print(f"{metrics.naive_modified:.4f}")  # 3.0976  <- the common error
print(f"{metrics.effective:.4f}")    # 3.1694  <- numerical arbiter

print(check_consistency(bond, settle, quote, metrics))
# [PASS] below_par_ordering: trading below par at 97.8000%, so
#        coupon < current yield < YTM must hold: 4.0000% < 4.0900% < 4.6867%
# ...
```

Scenario analysis isolates the true higher-order residual:

```python
from bondmath import scenario

s = scenario(bond, settle, quote.ytm, shock_bp=100)
print(f"{s.exact:.2f}")          # 966.55
print(f"{s.residual_bp:+.2f}")   # +0.09 bp, not the six the model claimed
```

## Conventions

ACT/ACT ICMA, 30/360 (US bond basis), ACT/365 and ACT/360. Annual,
semi-annual, quarterly and monthly coupons. Schedules are generated backward
from maturity, so an off-cycle issue produces a correctly handled irregular
first period.

## Install

```bash
pip install -e ".[dev]"
```

Tests:

```bash
python -m pytest -q
```

35 tests, including regression tests that pin the numbers in the write-up. If
someone ever swaps the divisor back, `test_naive_divisor_is_the_one_that_disagrees`
fails.

## Status

The `bondmath` core is complete and tested. The MCP server layer is in
progress — it will expose `price_from_yield`, `yield_from_price`,
`bond_analytics`, `cashflow_schedule`, `scenario_shock` and
`check_consistency` as agent tools.

## Not covered

Embedded optionality, floating coupons, sinking funds, amortising structures,
ex-dividend conventions, credit risk and settlement lag. A teaching and
checking instrument, not a trading system.

## License

MIT.
