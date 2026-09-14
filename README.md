# fixed-income-mcp

<!-- mcp-name: io.github.joselillotrax-cell/fixed-income-mcp -->

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

## As an MCP server

Six tools, so an agent can stop doing this arithmetic in its head.

| Tool | Does |
|---|---|
| `yield_from_price_tool` | Solves yield to maturity from a market price |
| `price_from_yield_tool` | Clean and dirty price at a given yield |
| `bond_analytics_tool` | Duration, DV01, convexity — with both cross-checks |
| `cashflow_schedule_tool` | Remaining flows, discount factors, present values |
| `scenario_shock_tool` | Exact repricing vs first- and second-order estimates |
| `check_consistency_tool` | Audits a yield someone else produced |

Two decisions make these usable by a model rather than merely callable.

**Units live in the parameter names.** A field called `coupon_rate` invites
the question of whether 4% is `4` or `0.04`, and a wrong guess is off by a
factor of a hundred while looking reasonable. Everything here is
`coupon_rate_pct`, `clean_price_pct_of_face`, `ytm_pct`. Nothing to guess.

**Every valuation returns its own verdict.** Results carry an
`all_checks_passed` flag and the list of bounds behind it, so the model sees
whether the number is admissible, not just what it is.

### Install

```bash
pip install "bondmath[mcp]"
```

Then register it. In Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fixed-income": {
      "command": "fixed-income-mcp"
    }
  }
}
```

In Claude Code:

```bash
claude mcp add fixed-income -- fixed-income-mcp
```

**If the command isn't found:** the pip-installed script may not be on the
launching process's `PATH`. Use the absolute path instead — find it with
`which fixed-income-mcp` (inside the environment you installed into) and put
that full path in `command`.

**If you're on macOS and the server shows "disconnected" with no clear
error:** check `~/Library/Logs/Claude/mcp-server-fixed-income.log` before
anything else — the panel's error message is not the real one. Two macOS
specifics bit this project during development: Python 3.14 silently skips
hidden `.pth` files, and iCloud Drive can mark files inside `~/Desktop` as
hidden without warning; separately, `~/Desktop`, `~/Documents` and
`~/Downloads` require an explicit permission grant for a launched
subprocess to read from at all. Installing outside those folders (`~/dev`,
`~/code`, anywhere not cloud-synced) avoids both.

### In practice

Three exchanges from testing this against Claude Desktop, exercised with
natural-language questions rather than pre-filled parameters and cross-checked
independently against each tool's own output:

> *"Un compañero me dice que un bono al 4% semestral... cotizando al 97,80%,
> rinde un 4,01%. ¿Tiene sentido?"*
> — called `check_consistency_tool` directly rather than recomputing from
> scratch, correctly identified the 67.67 bp error, and explained why without
> needing to iterate: *"el bono cotiza bajo par, así que obligatoriamente
> cupón < rendimiento corriente < TIR."*

> *"Un bono al 3,5% que vence en 2032 cotiza a 96,4. ¿Qué rentabilidad me da?"*
> — face value, issue date and payment frequency were all missing. The
> schema's required `issue_date` field rejected the first call attempt
> (`MCP error -32602: invalid_type`); the model disclosed the assumption it
> then made rather than silently inventing it, and computed both semi-annual
> and annual scenarios to show the frequency assumption barely moved the
> answer.

> *"¿Cuál tiene más riesgo de tipos: el A (4%, vence 2029, cotiza a 98) o el B
> (2%, vence 2035, cotiza a 85)?"*
> — two chained tool calls, correct verdict (B, 7.43 years vs 2.14), and an
> explanation that separated the two forces at work: longer maturity *and* a
> low coupon that pushes more of the bond's value into the final principal
> payment. It also flagged, unprompted, that comparing DV01 in currency terms
> gives a different ratio than comparing modified duration in percentage
> terms, since the two bonds don't trade at the same price.

### Auditing a suspect figure

`check_consistency_tool` exists for the case that started this project —
checking a number someone already produced:

```
claimed_ytm_pct        4.01
correct_ytm_pct        4.686736
error_bp               -67.67
all_checks_passed      false

[FAIL] below_par_ordering: trading below par at 97.8000%, so
       coupon < current yield < YTM must hold:
       4.0000% < 4.0900% < 4.0100%
```

## Install for development

```bash
pip install -e ".[dev]"
python -m pytest -q
```

59 tests. Some pin the numbers in the write-up: if someone ever swaps the
divisor back, `test_naive_divisor_is_the_one_that_disagrees` fails. Others
guard the MCP schema itself — `test_every_parameter_carries_a_description`
exists because an earlier version of this server shipped with none, which a
model could only have discovered by guessing.

## Not covered

Embedded optionality, floating coupons, sinking funds, amortising structures,
ex-dividend conventions, credit risk and settlement lag. A teaching and
checking instrument, not a trading system.

## License

MIT.
