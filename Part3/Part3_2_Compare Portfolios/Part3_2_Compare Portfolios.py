"""Part 3, Question 2: evidence-based portfolio comparability, not a backtest.

Run with the project .venv Python interpreter. Reads Question 1's saved weights
and reuses its parsing/optimization checks against HW_Prices.csv. Writes only
Question 2 outputs. Requires numpy, pandas and scipy; no external financial data.
"""

from contextlib import redirect_stdout
from io import StringIO
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
Q1_DIR = HERE.parent / "Part3_1_Different Portfolios"
Q1_FILE = Q1_DIR / "Part3_1_Different Portfolios.py"
spec = importlib.util.spec_from_file_location("question1", Q1_FILE)
q1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q1)  # Import functions without running or overwriting Q1.


def verify_weights(metadata, prices, estimation):
    """Check that saved portfolios really use the inputs described in the table."""
    saved = pd.read_csv(Q1_DIR / "Part3_1_weights.csv", index_col="Ticker")
    if len(saved) != 30 or saved.index.has_duplicates or set(saved.index) != set(metadata.index):
        raise ValueError("Q1 weights do not have the expected thirty stocks")
    saved = saved.loc[metadata.index]
    pd.testing.assert_frame_equal(saved[["Company", "Major Group"]],
                                  metadata[["Company", "Major Group"]])
    weights = saved[q1.PORTFOLIOS]
    if not np.isfinite(weights.to_numpy()).all() or (weights < 0).any().any():
        raise ValueError("Invalid saved weights")
    np.testing.assert_allclose(weights.sum(), 1, rtol=0, atol=1e-10)

    # Reconstruct proxies from their actual share date, not the assignment's
    # unavailable January-2024 share counts. This is a provenance check only.
    for label, month in [(q1.PORTFOLIOS[0], "2024-01"), (q1.PORTFOLIOS[1], "2014-06")]:
        values = metadata["Shares Dec 2024"] * prices.loc[month]
        np.testing.assert_allclose(weights[label], values / values.sum(), atol=1e-12)
    ordered = sorted(metadata.index, key=lambda t: (metadata.loc[t, "Company"].casefold(), t))
    for label, selected in [(q1.PORTFOLIOS[2], ordered[15:]), (q1.PORTFOLIOS[3], ordered[:15])]:
        expected = pd.Series(0.0, index=metadata.index)
        expected.loc[selected] = 1 / 15
        np.testing.assert_allclose(weights[label], expected, atol=1e-12)
    counts = metadata["Major Group"].value_counts()
    group_weights = metadata["Major Group"].map(lambda group: 0.2 / counts[group])
    np.testing.assert_allclose(weights[q1.PORTFOLIOS[8]], group_weights, atol=1e-12)

    # Re-estimate with data through June 2014 only and compare to the saved
    # vectors. Successful reproduction is evidence about actual implementation,
    # beyond merely labeling a portfolio 'out of sample'.
    tangent, minimum, inverse, diverse, erc, solver_log = q1.optimize_portfolios(
        estimation.mean().to_numpy(), estimation.cov().to_numpy())
    for position, vector in zip([4, 5, 6, 7, 9], [tangent, minimum, inverse, diverse, erc]):
        np.testing.assert_allclose(weights[q1.PORTFOLIOS[position]], vector,
                                   rtol=1e-6, atol=1e-7)
    summary = pd.DataFrame({"Weight sum": weights.sum(),
                            "Negative weights": (weights < 0).sum(),
                            "Largest holding": weights.idxmax(),
                            "Largest weight (%)": 100 * weights.max(),
                            "Concentration HHI": (weights**2).sum()})
    return weights, summary


def classification(estimation):
    """Separate earliest conditional eligibility from verified investability."""
    window = f"{estimation.index[0]} to {estimation.index[-1]} ({len(estimation)} returns)"
    rows = [
        ["(i)", "Market cap A proxy", "Dec-2024 shares x Jan-2024 prices",
         "Not before Dec-2024 share data are published", "None; Jan-2024 price snapshot",
         "Yes for Jul-2014 or Jan-2024", "Jan-2025 lower bound; later if shares published later; outside file",
         "Descriptive weights only in supplied history"],
        ["(ii)", "Market cap B proxy", "Dec-2024 shares x Jun-2014 prices",
         "Not before Dec-2024 share data are published", "None; Jun-2014 price snapshot",
         "Yes for Jul-2014 or Jan-2024", "Jan-2025 lower bound; later if shares published later; outside file",
         "Descriptive weights only in supplied history"],
        ["(iii)", "Alphabetically higher", "Fixed universe; last 15 company names",
         "Names/universe metadata have no as-of date", "None",
         "Unknown metadata history; no future returns used", "Feb-2004 if known Jan-2004; join Jul-2014 if known Jun-2014",
         "Conditional addition to July-2014 comparison"],
        ["(iv)", "Alphabetically lower", "Fixed universe; first 15 company names",
         "Names/universe metadata have no as-of date", "None",
         "Unknown metadata history; no future returns used", "Feb-2004 if known Jan-2004; join Jul-2014 if known Jun-2014",
         "Conditional addition to July-2014 comparison"],
    ]
    definitions = [("(v)", "Tangency", "Sample means and full covariance"),
                   ("(vi)", "Minimum variance", "Full sample covariance"),
                   ("(vii)", "Inverse volatility", "Sample individual volatilities"),
                   ("(viii)", "Most diversified", "Volatilities and full covariance")]
    for numeral, label, inputs in definitions:
        rows.append([numeral, label, inputs, "After Jun-2014 month-end prices", window,
                     "No post-Jun-2014 prices; universe history unverified",
                     "Jul-2014, conditional on price quality/universe availability",
                     "Same estimation-information set; shared data caveats"])
    rows.append(["(ix)", "Equal group budget", "Fixed universe and five major-group labels",
                 "Group/universe metadata have no as-of date", "None",
                 "Unknown metadata history; no future returns used", "Feb-2004 if known Jan-2004; join Jul-2014 if known Jun-2014",
                 "Conditional addition to July-2014 comparison"])
    rows.append(["(x)", "Equal risk contribution", "Full covariance; equal risk budgets per stock",
                 "After Jun-2014 month-end prices", window,
                 "No post-Jun-2014 prices; universe history unverified",
                 "Jul-2014, conditional on price quality/universe availability",
                 "Same estimation-information set; shared data caveats"])
    return pd.DataFrame(rows, columns=["Item", "Portfolio", "Weight-setting inputs",
                                      "Input availability", "Estimation window / snapshot",
                                      "Future information?", "Earliest defensible comparison start",
                                      "Classification"]).set_index("Item")


def main():
    print("PART 3, QUESTION 2: WHICH PORTFOLIOS ARE COMPARABLE?")
    metadata, prices, estimation, flags = q1.load_data()
    weights, evidence = verify_weights(metadata, prices, estimation)
    table = classification(estimation)
    # Read/check stock returns to assess the data, without constructing a new
    # portfolio performance series. Q1 produces weights and in-sample moments,
    # not realized portfolio returns; the user's conditional backtest request
    # therefore does not apply to the present Q1 implementation.
    stock_returns = prices.pct_change(fill_method=None).iloc[1:]
    candidate_months = stock_returns.index[stock_returns.index > q1.CUTOFF]
    expected = pd.period_range(q1.CUTOFF + 1, prices.index[-1], freq="M")
    pd.testing.assert_index_equal(candidate_months, expected, check_names=False)
    print(f"\nPrice history: {prices.index[0]} to {prices.index[-1]}, {len(prices)} observations.")
    print(f"Estimation: {estimation.index[0]} to {estimation.index[-1]}, {len(estimation)} monthly returns.")
    print(f"Candidate common out-of-sample dates: {candidate_months[0]} to {candidate_months[-1]}, "
          f"{len(candidate_months)} months (dates only; no performance claim).")
    print("Saved weights reproduce from their stated Q1 inputs; all ten are long-only and sum to one.")
    print("Q1 estimates moments and weights only; it does not produce realized portfolio return series.")
    print("Accordingly Q2 classifies comparability and audits returns, without adding a performance backtest.")
    print("\nCOMPARABILITY TABLE (also saved as a readable HTML table)")
    print(table.to_string())
    print("\nDESCRIPTIVE WEIGHT EVIDENCE -- NOT PERFORMANCE")
    print(evidence.to_string(float_format=lambda v: f"{v:.6f}"))
    first, second = weights.iloc[:, 0], weights.iloc[:, 1]
    distance = 0.5 * (first - second).abs().sum()
    print(f"P1/P2 half-L1 weight distance: {100*distance:.2f}% of capital.")
    print("This measures allocation differences between two vectors, not realized turnover or trading profit.")
    print(f"AAPL weights: P1 {100*first.loc['AAPL']:.2f}%, P2 {100*second.loc['AAPL']:.2f}%.")
    print("Their concentrations, stock/group allocations and differences can be compared descriptively.")

    # Augment the existing 40% price-change screen with actual prices and phase.
    # No detected move is automatically identified as a split or corrected.
    flags = flags.copy()
    flags["Previous price"] = [prices.loc[row.Month - 1, row.Ticker] for row in flags.itertuples()]
    flags["Current price"] = [prices.loc[row.Month, row.Ticker] for row in flags.itertuples()]
    flags["Phase"] = np.where(flags["In estimation sample"], "Estimation", "Candidate OOS")
    print("\nPRICE-DATA AUDIT: absolute monthly changes >40%")
    print(flags.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"{int(flags['In estimation sample'].sum())} flags affect estimation; "
          f"{int((~flags['In estimation sample']).sum())} occur after the estimation cutoff.")
    print("No missing/nonpositive prices or calendar gaps were found. This does not verify corporate-action adjustment.")
    print("Large moves may be genuine market changes or possible splits/corporate actions;")
    print("the CSV supplies no adjustment factors, dividend treatment or authoritative correction source.")
    print("For example, estimation includes AAPL Feb-2005, UNH May-2005 and CAT Jul-2005;")
    print("the candidate OOS period includes HPQ Nov-2015. These issues can affect both weights and returns.")
    print("The fixed list of 30 stocks has no historical membership, listing or classification history.")
    print("Selection/survivorship and historical availability therefore cannot be verified.")
    print("Reliable investable/total-return performance is not established by this file alone.")
    print("Neither adjusted prices nor corrections are invented; the classification remains complete despite that limitation.")

    print("\nFAIR COMPARISON FRAMEWORK")
    print("1. Freeze the Q1 targets estimated through June 2014; first earn July 2014 returns.")
    print("   Use the same July-2014 to December-2024 monthly observations for all eligible portfolios.")
    print("   A common return period alone does not remove future information from weight construction.")
    print("2. Hold the long-only, fully invested constraints, zero risk-free rate, and zero costs fixed.")
    print("   Objectives differ by design: tangency uses means; minimum variance/ERC use covariance;")
    print("   inverse volatility ignores correlations; most-diversified uses volatility and covariance.")
    print("   Their identical estimation dates make these rule differences comparable.")
    print("3. Monthly rebalancing means restoring the SAME frozen target weights before each month's return.")
    print("   It does not mean re-estimating parameters each month. Buy-and-hold would let weights drift;")
    print("   rolling re-estimation would be a new strategy requiring a separately specified information rule.")
    print("4. Alphabetical/group rules may join only if the supplied universe, names and groups are")
    print("   treated as known at June 2014; their metadata dates are unverified, not demonstrably future-free.")
    print("5. P1 uses Jan-2024 prices; P2 uses Jun-2014 prices. BOTH use Dec-2024 share proxies.")
    print("   Neither supports an investable July-2014 or January-2024 backtest. Even Jan-2024 closing")
    print("   prices are not known at the beginning of January. Publication dates for the share counts")
    print("   are unknown: January 2025 is only a lower-bound hypothetical start, with no subsequent prices here.")
    print("   The file's Jan-2004 shares could define another strategy, but substituting them changes Q1.")
    print("6. DJIA is a separate price-weighted index series, excluded from the 30 stock vectors.")
    print("   Comparing its price changes on identical dates can be informative, but it has a different")
    print("   weighting/membership methodology. It is not an ETF total-return series; dividend comparability")
    print("   and stock price adjustments must be established before claiming like-for-like performance.")

    print("\nSUBMISSION SUMMARY")
    print("Directly comparable in construction and estimation information: (v), (vi), (vii), (viii), (x).")
    print("Their 125 returns end June 2014, so a common July-2014 start is conditionally feasible.")
    print("All still share unresolved price-adjustment and fixed-universe limitations; this is not an unconditional investability claim.")
    print("Comparable with additional qualifications: (iii), (iv), (ix), if names/groups and universe were known at that start.")
    print("Not a fair historical investable comparison as specified: (i), (ii), because Dec-2024 shares")
    print("look ahead relative to their Jan-2024/Jun-2014 price snapshots. Compare their weight vectors only.")
    print("DJIA is an informative external index benchmark, not the same portfolio or a supplied ETF total return.")
    outputs = {"classification": table, "weight_evidence": evidence, "price_audit": flags}
    for suffix, frame in outputs.items():
        frame.to_csv(HERE / f"Part3_2_{suffix}.csv", index=suffix != "price_audit")
    html = """<!doctype html><html><head><meta charset="utf-8"><title>Portfolio comparability</title>
<style>body{font:14px Arial;margin:24px}table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ccc;padding:8px;text-align:left;vertical-align:top}
thead th{background:#eaf0f5}tbody tr:nth-child(even){background:#f5f7fa}</style></head><body>
<h1>Part 3 Question 2: comparability</h1><p>Common candidate period: July 2014–December 2024.
This is a classification, not a performance backtest. All historical claims are subject to unresolved
corporate-action adjustment and fixed-universe limitations.</p>"""
    html += table.to_html() + "<h2>Descriptive weight evidence</h2>" + evidence.to_html(float_format=lambda v: f"{v:.6f}")
    html += "<h2>Uncorrected price-change flags</h2>" + flags.to_html(index=False, float_format=lambda v: f"{v:.3f}")
    html += "</body></html>"
    (HERE / "Part3_2_comparison.html").write_text(html, encoding="utf-8")
    print(f"\nReadable comparison: {HERE / 'Part3_2_comparison.html'}")
    print(f"Classification CSV: {HERE / 'Part3_2_classification.csv'}")
    print(f"Full submission explanation and audit: {HERE / 'Part3_2_results.txt'}")


if __name__ == "__main__":
    report = StringIO()
    with redirect_stdout(report):
        main()
    text = report.getvalue()
    print(text, end="")
    (HERE / "Part3_2_results.txt").write_text(text, encoding="utf-8")
