"""Part 3, Question 1: construct ten portfolios; no performance backtest.

Run this file using the project's .venv interpreter in PyCharm, or from root:
.venv/Scripts/python.exe "Part3/Part3_1_Different Portfolios/Part3_1_Different Portfolios.py"
Requires numpy, pandas and scipy. A readable weight table is saved as HTML.
The sole financial input is HW_Prices.csv. All output is saved beside this file.
"""

import csv
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "Files_Homework" / "HW_Prices.csv"
CUTOFF = pd.Period("2014-06", freq="M")
PORTFOLIOS = ["P1 Cap A proxy", "P2 Cap B proxy", "P3 Alpha higher",
              "P4 Alpha lower", "P5 Tangency", "P6 Min variance",
              "P7 Inverse vol", "P8 Most diversified", "P9 Group budget", "P10 ERC"]


def load_data():
    # Read metadata as CSV rows, not a normal DataFrame header. Every field is
    # matched by POSITION to the ticker row; DJIA's metadata fields are blank.
    with SOURCE.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))
    expected = ["Name", "Major Group", "Shares Jan 2004", "Shares Dec 2024", "Date"]
    if [row[0].strip() for row in rows[:5]] != expected:
        raise ValueError("Unexpected metadata rows; inspect the CSV before proceeding")
    width = len(rows[4])
    if any(len(row) != width for row in rows if row):
        raise ValueError("Inconsistent CSV row widths")
    tickers = [value.strip() for value in rows[4]]
    if tickers.count("DJIA") != 1:
        raise ValueError("Expected exactly one separate DJIA column")
    positions = [i for i in range(1, width) if tickers[i] != "DJIA"]
    if len(positions) != 30 or len({tickers[i] for i in positions}) != 30:
        raise ValueError("Expected exactly thirty unique stock tickers")
    metadata = pd.DataFrame({
        "Ticker": [tickers[i] for i in positions],
        "Company": [rows[0][i].strip() for i in positions],
        "Major Group": [rows[1][i].strip() for i in positions],
        "Shares Jan 2004": [rows[2][i] for i in positions],
        "Shares Dec 2024": [rows[3][i] for i in positions],
    }).set_index("Ticker")
    for column in ["Shares Jan 2004", "Shares Dec 2024"]:
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")
        if not np.isfinite(metadata[column]).all() or (metadata[column] <= 0).any():
            raise ValueError(f"Invalid share counts in {column}")
    if (metadata[["Company", "Major Group"]] == "").any().any():
        raise ValueError("Missing company names or major groups")
    if metadata["Major Group"].nunique() != 5:
        raise ValueError("Expected five major groups")

    # The fifth row is the price-table header. The benchmark is validated but
    # excluded from all stock return estimates, stock weights and optimizations.
    prices = pd.read_csv(SOURCE, skiprows=4, dtype=str).set_index("Date")
    if not prices.index.str.fullmatch(r"\d{6}").all():
        raise ValueError("Price dates must be YYYYMM")
    dates = pd.to_datetime(prices.index, format="%Y%m", errors="coerce")
    if dates.isna().any():
        raise ValueError("Failed price-date parsing")
    prices.index = pd.PeriodIndex(dates, freq="M", name="Month")
    prices = prices.apply(pd.to_numeric, errors="coerce").sort_index()
    if prices.index.has_duplicates:
        raise ValueError("Duplicate price months")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Missing, nonfinite or nonpositive prices")
    expected_months = pd.period_range(prices.index[0], prices.index[-1], freq="M")
    if not prices.index.equals(expected_months):
        raise ValueError("Price dates are not consecutive calendar months")
    prices = prices[metadata.index]
    for month in [pd.Period("2024-01", freq="M"), CUTOFF]:
        if month not in prices.index:
            raise ValueError(f"Required price snapshot missing: {month}")
    # Successive-month SIMPLE price returns, in decimals. These are not
    # established total returns: dividend and corporate-action treatment is
    # not documented in the supplied file. No silent split corrections.
    returns = prices.pct_change(fill_method=None).iloc[1:]
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Invalid computed stock returns")
    extreme_rows = []
    # 40% absolute change is a transparent screening threshold, not a finding
    # that an observation is erroneous. Real market moves may also exceed it.
    for month in returns.index:
        for ticker in returns.columns:
            value = returns.loc[month, ticker]
            if abs(value) > 0.40:
                extreme_rows.append({"Month": month, "Ticker": ticker,
                                     "Price return (%)": 100 * value,
                                     "In estimation sample": month <= CUTOFF})
    extremes = pd.DataFrame(extreme_rows,
                            columns=["Month", "Ticker", "Price return (%)", "In estimation sample"])
    estimation = returns.loc[:CUTOFF]
    if len(estimation) <= 30:
        raise ValueError("Insufficient estimation history")
    return metadata, prices, estimation, extremes


def optimize_portfolios(mu, covariance):
    """Long-only, fully invested optimization with gradients and multiple starts."""
    n = len(mu)
    sigma = np.sqrt(np.diag(covariance))
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues.min() <= 0 or not np.isfinite(covariance).all():
        raise ValueError("Covariance must be finite and positive definite")
    # Scale inputs for numerically well-behaved objectives without changing
    # the optimum. Scaling both mu and volatility leaves Sharpe unchanged.
    scale = np.sqrt(np.mean(np.diag(covariance)))
    cov = covariance / scale**2
    mean = mu / scale
    vol = sigma / scale
    equal = np.full(n, 1 / n)
    inverse = (1 / sigma) / (1 / sigma).sum()
    rng = np.random.default_rng(2026)
    starts = [equal, inverse, *rng.dirichlet(np.ones(n), size=3)]
    constraint = {"type": "eq", "fun": lambda w: w.sum() - 1,
                  "jac": lambda w: np.ones(n)}
    bounds = [(0, 1)] * n
    diagnostics = []

    def solve(label, objective, gradient):
        candidates = []
        for number, initial in enumerate(starts, start=1):
            result = minimize(objective, initial, jac=gradient, method="SLSQP",
                              bounds=bounds, constraints=constraint,
                              options={"ftol": 1e-12, "maxiter": 3000})
            valid = (result.success and np.isfinite(result.x).all()
                     and abs(result.x.sum() - 1) < 1e-8 and result.x.min() >= -1e-10)
            diagnostics.append({"Portfolio": label, "Start": number,
                                "Success": bool(result.success), "Valid": bool(valid),
                                "Objective": result.fun, "Message": result.message})
            if valid:
                candidates.append(result)
        if not candidates:
            raise RuntimeError(f"All optimizations failed for {label}: {diagnostics[-5:]}")
        best = min(candidates, key=lambda result: result.fun)
        # Remove only numerical boundary noise, then verify objective stability.
        weights = np.maximum(best.x, 0)
        weights /= weights.sum()
        if abs(objective(weights) - best.fun) > 1e-8:
            raise RuntimeError(f"Normalization materially changed {label}")
        # First-order simplex check: positive holdings share a Lagrange
        # multiplier; zero holdings cannot lower the objective by entering.
        g = gradient(weights)
        active = weights > 1e-7
        multiplier = g[active].mean()
        violation = max(np.max(np.abs(g[active] - multiplier)),
                        max(0, multiplier - g[~active].min()) if (~active).any() else 0)
        if violation > 2e-5:
            raise RuntimeError(f"First-order optimality check failed for {label}: {violation}")
        print(f"{label}: {len(candidates)}/{len(starts)} valid starts; "
              f"first-order residual {violation:.2e}.")
        return weights

    def ratio_objective(w, numerator):
        return -(w @ numerator) / np.sqrt(w @ cov @ w)

    def ratio_gradient(w, numerator):
        variance = w @ cov @ w
        return -numerator / np.sqrt(variance) + (w @ numerator) * (cov @ w) / variance**1.5

    # P5: maximize monthly Sharpe with rf=0; P6: minimize full portfolio variance.
    tangency = solve("P5 Tangency", lambda w: ratio_objective(w, mean),
                     lambda w: ratio_gradient(w, mean))
    min_variance = solve("P6 Min variance", lambda w: w @ cov @ w,
                         lambda w: 2 * cov @ w)
    # P8: diversification ratio = weighted average individual volatility /
    # portfolio volatility. Correlations enter through the full covariance.
    diversified = solve("P8 Most diversified", lambda w: ratio_objective(w, vol),
                         lambda w: ratio_gradient(w, vol))

    # P10: solve a strictly convex risk-budget problem in positive x:
    # min 0.5*x'Cov*x - sum_i b_i*log(x_i), where b_i=1/30.
    # First-order conditions give x_i*(Cov*x)_i=b_i. Normalizing x to sum to
    # one preserves each stock's share of variance. This uses ALL covariances.
    budget = np.full(n, 1 / n)
    def erc_objective(x):
        return 0.5 * x @ cov @ x - budget @ np.log(x)
    def erc_gradient(x):
        return cov @ x - budget / x
    erc_candidates = []
    for number, initial in enumerate([equal, inverse], start=1):
        initial = initial / np.sqrt(initial @ cov @ initial)
        result = minimize(erc_objective, initial, jac=erc_gradient,
                          method="L-BFGS-B", bounds=[(1e-12, None)] * n,
                          options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 10000,
                                   "maxls": 100})
        w = result.x / result.x.sum()
        rc = w * (covariance @ w) / (w @ covariance @ w)
        error = np.max(np.abs(rc - budget))
        valid = result.success and np.isfinite(w).all() and (w > 0).all() and error < 1e-6
        diagnostics.append({"Portfolio": "P10 ERC", "Start": number,
                            "Success": bool(result.success), "Valid": bool(valid),
                            "Objective": result.fun, "Message": result.message})
        if valid:
            erc_candidates.append((error, w))
    if not erc_candidates:
        raise RuntimeError("ERC solver failed success or risk-contribution checks")
    erc_error, erc = min(erc_candidates, key=lambda pair: pair[0])
    print(f"P10 ERC: {len(erc_candidates)}/2 valid starts; maximum risk-share error {erc_error:.2e}.")
    return tangency, min_variance, inverse, diversified, erc, pd.DataFrame(diagnostics)


def main():
    # Print the PDF's evidence and the assumptions needed where inputs are absent.
    import sys
    project_root = next(p for p in Path(__file__).resolve().parents if (p / "Files_Homework").is_dir())
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from homework_assumptions import print_assumptions
    print_assumptions("P3Q1")
    print("PART 3, QUESTION 1: DIFFERENT PORTFOLIOS -- CONSTRUCTION ONLY")
    metadata, prices, estimation, extremes = load_data()
    print("\nIMPORTANT: JANUARY 2024 SHARE COUNTS ARE NOT IN THIS FILE.")
    print("It supplies Shares Jan 2004 and Shares Dec 2024 only. P1/P2 use the")
    print("DECEMBER 2024 counts as an explicit proxy, multiplied by January 2024")
    print("and June 2014 prices respectively. Both are approximations, with future")
    print("share information relative to those price dates; neither is an investable historical backtest.")
    print("Price/share adjustment compatibility is also undocumented: supplied prices")
    print("may not be on the same split-adjustment basis as shares. No corrections are invented.")
    print(f"\nPrice history: {prices.index[0]} to {prices.index[-1]}, {len(prices)} months; 30 stocks, DJIA excluded.")
    print(f"Estimation returns: {estimation.index[0]} to {estimation.index[-1]}, {len(estimation)} monthly returns.")
    print("January 2004 supplies the first lagged price; June 2014's return is included.")
    print("Means and sample covariance (ddof=1) use these identical observations for all stocks.")
    print("Returns are simple price returns in decimals; dividend treatment is undocumented.")
    print("\nPRICE-CHANGE SCREEN: absolute monthly changes >40%; no automatic adjustment")
    print(extremes.to_string(index=False, float_format=lambda v: f"{v:.3f}") if len(extremes)
          else "No changes exceed the threshold.")
    print("These may reflect corporate actions or genuine market moves. Supplied data")
    print("are used unchanged; affected estimation moments and resulting weights require caution.")

    mu = estimation.mean().to_numpy()
    covariance = estimation.cov().to_numpy()
    print(f"Covariance condition number: {np.linalg.cond(covariance):.2f}; "
          f"minimum eigenvalue: {np.linalg.eigvalsh(covariance).min():.8f}.")
    print("\nOPTIMIZATION ASSUMPTION: fully invested, long-only, no leverage; rf=0.")
    print("The assignment does not specify these constraints; they are our consistent baseline.")
    tangent, minimum, inverse, diverse, erc, solver_log = optimize_portfolios(mu, covariance)
    weights = pd.DataFrame(0.0, index=metadata.index, columns=PORTFOLIOS)
    shares = metadata["Shares Dec 2024"]
    # P1/P2: supplied price times Dec-2024 shares, normalized across 30 stocks.
    for label, month in [(PORTFOLIOS[0], "2024-01"), (PORTFOLIOS[1], "2014-06")]:
        market_values = shares * prices.loc[month]
        weights[label] = market_values / market_values.sum()
    # P3/P4: sort COMPANY names, not ticker symbols. Case-insensitive literal
    # alphabetical order retains spaces/punctuation; ticker breaks any name tie.
    ordered = sorted(metadata.index, key=lambda t: (metadata.loc[t, "Company"].casefold(), t))
    lower, higher = ordered[:15], ordered[15:]
    weights.loc[higher, PORTFOLIOS[2]] = 1 / 15
    weights.loc[lower, PORTFOLIOS[3]] = 1 / 15
    print("\nFIRST 15 COMPANY NAMES (P4 lower half):")
    print(metadata.loc[lower, ["Company"]].to_string())
    print("\nLAST 15 COMPANY NAMES (P3 higher half):")
    print(metadata.loc[higher, ["Company"]].to_string())
    for label, vector in zip([PORTFOLIOS[i] for i in [4, 5, 6, 7, 9]],
                             [tangent, minimum, inverse, diverse, erc]):
        weights[label] = vector
    # P9: a fifth of the total budget per major group, equal within the group.
    counts = metadata["Major Group"].value_counts().sort_index()
    weights[PORTFOLIOS[8]] = metadata["Major Group"].map(lambda group: 0.2 / counts[group])
    groups = counts.rename("Stock count").to_frame()
    groups["Group weight (%)"] = 20.0
    groups["Each stock weight (%)"] = 20 / groups["Stock count"]
    print("\nFIVE MAJOR GROUPS (P9):")
    print(groups.to_string(float_format=lambda v: f"{v:.4f}"))
    np.testing.assert_allclose(weights[PORTFOLIOS[8]].groupby(metadata["Major Group"]).sum(), 0.2, atol=1e-12)

    # Validate every portfolio before saving or reporting its weights.
    if not np.isfinite(weights.to_numpy()).all() or (weights < 0).any().any():
        raise RuntimeError("Invalid or negative portfolio weights")
    np.testing.assert_allclose(weights.sum(), 1, rtol=0, atol=1e-10)
    validation = pd.DataFrame({"Weight sum": weights.sum(),
                               "Negative weights": (weights < 0).sum(),
                               "Holdings > 0.00001%": (weights > 1e-7).sum()})
    table = metadata[["Company", "Major Group"]].join(weights)
    display = metadata[["Company", "Major Group"]].join(weights * 100)
    print("\nALL 30 STOCK WEIGHTS: PERCENT OF TOTAL PORTFOLIO")
    print("P1/P2 are share-count PROXIES. Full-precision FRACTIONAL weights are saved in the CSV.")
    print(display.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nVALIDATION (sums in fractions, target=1):")
    print(validation.to_string(float_format=lambda v: f"{v:.12f}"))

    estimated_rows = []
    for label in [PORTFOLIOS[i] for i in [4, 5, 7, 9]]:
        w = weights[label].to_numpy()
        expected_return = w @ mu
        volatility = np.sqrt(w @ covariance @ w)
        estimated_rows.append({"Portfolio": label, "Monthly expected return (%)": 100 * expected_return,
                               "Monthly volatility (%)": 100 * volatility,
                               "Monthly Sharpe (rf=0)": expected_return / volatility,
                               "Diversification ratio": (w @ np.sqrt(np.diag(covariance))) / volatility})
    estimates = pd.DataFrame(estimated_rows).set_index("Portfolio")
    print("\nOPTIMIZED PORTFOLIOS: IN-SAMPLE MONTHLY ESTIMATES THROUGH JUNE 2014")
    print(estimates.to_string(float_format=lambda v: f"{v:.4f}"))
    print("These are estimation-sample moments, NOT realized future performance.")

    erc_variance = erc @ covariance @ erc
    risk = metadata[["Company"]].copy()
    risk["ERC weight"] = erc
    risk["Variance contribution fraction"] = erc * (covariance @ erc) / erc_variance
    risk["Target fraction"] = 1 / 30
    risk["Deviation"] = risk["Variance contribution fraction"] - 1 / 30
    np.testing.assert_allclose(risk["Variance contribution fraction"].sum(), 1, atol=1e-12)
    max_deviation = risk["Deviation"].abs().max()
    if max_deviation > 1e-6:
        raise RuntimeError("ERC contributions are not sufficiently equal")
    print("\nP10 wording '30 stocks groups' is ambiguous: equal risk contribution is")
    print("assigned to each of the 30 STOCKS (3.333333% of variance each), not five groups.")
    print(f"Maximum absolute deviation from 1/30: {max_deviation:.3e} "
          f"({100 * max_deviation:.3e} percentage points of total variance).")
    inv_rc = inverse * (covariance @ inverse) / (inverse @ covariance @ inverse)
    print(f"For comparison, P7 inverse-volatility risk shares range from {100*inv_rc.min():.3f}% "
          f"to {100*inv_rc.max():.3f}%; correlations prevent exact equal contributions.")

    explanations = [
        "Largest supplied Dec-2024 shares times Jan-2024 prices dominate; proxy and price/share-basis cautions apply.",
        "Same future share proxy times June-2014 prices; changing the price snapshot changes relative sizes.",
        "Exactly the last 15 company names receive 6.6667% each; alphabetical position drives selection.",
        "Exactly the first 15 company names receive 6.6667% each; no return estimates enter.",
        "High estimated return relative to covariance risk attracts weight; sample means can make this concentrated.",
        "Low variance and helpful covariance attract weight; estimated mean returns are not an objective input.",
        "Lower individual volatility receives more weight; correlations do not enter this formula.",
        "Holdings maximize weighted individual volatility relative to portfolio volatility using correlations.",
        "Each group receives 20%; stocks in smaller groups receive more each, irrespective of size or volatility.",
        "Different weights offset different volatilities and correlations to equalize stock variance contributions.",
    ]
    print("\nLARGEST HOLDINGS AND METHOD")
    for label, explanation in zip(PORTFOLIOS, explanations):
        ranked = weights[label].sort_values(ascending=False, kind="stable")
        print(label + ": " + "; ".join(f"{ticker} {100*value:.2f}%" for ticker, value in ranked.head(5).items()))
        print("  " + explanation)
        if label in [PORTFOLIOS[2], PORTFOLIOS[3], PORTFOLIOS[8]]:
            print("  Displayed top five can be tied with additional holdings; consult the full table.")
    print("\nDATES: P1 uses Jan-2024 prices; P2 uses June-2014 prices; both use Dec-2024 share proxies.")
    print("P5/P6/P7/P8/P10 use estimates through June 2014 (hypothetically available afterward).")
    print("P3/P4/P9 use the supplied names/groups and universe, without historical membership verification.")
    print("There is NO common genuinely investable inception date for all ten constructions.")
    print("These are formula-based snapshots under stated assumptions, not historical backtests.")
    print("No realized performance comparison, utility or later Part 3 questions are computed.")

    outputs = {"weights": table, "weights_percent": display, "validation": validation,
               "metadata": metadata, "group_budgets": groups, "estimated_moments": estimates,
               "ERC_risk_contributions": risk, "price_change_flags": extremes,
               "solver_diagnostics": solver_log}
    for suffix, frame in outputs.items():
        frame.to_csv(HERE / f"Part3_1_{suffix}.csv", index=suffix not in ["price_change_flags", "solver_diagnostics"])
    # A standalone HTML table is easier to inspect than a wide console line.
    # Sticky identifiers and horizontally scrollable output preserve all 30x10 cells.
    html = """<!doctype html><html><head><meta charset="utf-8"><title>Part 3 Q1 weights</title>
<style>body{font:14px Arial;margin:24px}table{border-collapse:collapse;white-space:nowrap}
th,td{padding:8px;border:1px solid #ddd;text-align:right}thead th{background:#eaf0f5;position:sticky;top:0}
tbody th{position:sticky;left:0;background:#fff}tbody tr:nth-child(even){background:#f5f7f9}</style>
</head><body><h1>Part 3 Question 1: portfolio weights (%)</h1>
<p>P1/P2 use December 2024 shares as a proxy: look-ahead and price/share adjustment limitations apply.
P1 prices: January 2024; P2 prices and estimation cutoff: June 2014. No common investable inception date.</p>
<p>All weights long-only and fully invested. CSV weights are fractions; this table is percent.</p>"""
    html += display.to_html(float_format=lambda v: f"{v:.4f}") + "</body></html>"
    (HERE / "Part3_1_weights.html").write_text(html, encoding="utf-8")
    print(f"\nFull fractional-weight CSV: {HERE / 'Part3_1_weights.csv'}")
    print(f"Readable 30-row weight table: {HERE / 'Part3_1_weights.html'}")
    print(f"Diagnostics and text report directory: {HERE}")


if __name__ == "__main__":
    report = StringIO()
    with redirect_stdout(report):
        main()
    text = report.getvalue()
    print(text, end="")
    (HERE / "Part3_1_results.txt").write_text(text, encoding="utf-8")
