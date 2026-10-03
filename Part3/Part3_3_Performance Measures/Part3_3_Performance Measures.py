"""Part 3, Question 3: descriptive performance of the ten Q1 portfolios and DJIA.

Run with the project .venv interpreter in PyCharm, or from the project root:
.venv/Scripts/python.exe "Part3/Part3_3_Performance Measures/Part3_3_Performance Measures.py"
Requires numpy, pandas, scipy and matplotlib. Uses HW_Prices.csv and the saved
Q1 weight definitions. Professor Q4/Q5 requires original alignment and
supplied-price returns. June-2014 weights create look-ahead in the earlier sample.
"""

from contextlib import redirect_stdout
from io import StringIO
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
Q2_FILE = HERE.parent / "Part3_2_Compare Portfolios" / "Part3_2_Compare Portfolios.py"
spec = importlib.util.spec_from_file_location("question2", Q2_FILE)
q2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q2)
q1 = q2.q1
GAMMA = 5.0


def portfolio_returns(prices, weights):
    """Apply fixed targets with costless monthly rebalancing, including DJIA."""
    # Q1's parser already checked dates, stock/DJIA prices, gaps and positivity.
    # Read DJIA separately: it is not a 31st stock. Professor Q4 requires
    # retaining its ORIGINAL row dates; do not shift it by six months.
    raw = pd.read_csv(q1.SOURCE, skiprows=4, dtype=str).set_index("Date")
    raw.index = pd.PeriodIndex(pd.to_datetime(raw.index, format="%Y%m"), freq="M", name="Month")
    djia = pd.to_numeric(raw["DJIA"], errors="raise").sort_index()
    pd.testing.assert_index_equal(djia.index, prices.index)
    if not np.isfinite(djia).all() or (djia <= 0).any():
        raise ValueError("Invalid DJIA index levels")

    # Both series are SIMPLE PRICE returns in decimals. January 2004 is only
    # the first price: absent December 2003 prices, no January return exists.
    stocks = prices.pct_change(fill_method=None).iloc[1:]
    djia_returns = djia.pct_change(fill_method=None).iloc[1:]
    if not (stocks.index - 1).equals(prices.index[:-1]):
        raise ValueError("A computed return spans more than one calendar month")
    pd.testing.assert_index_equal(stocks.index, djia_returns.index)
    # Matrix multiplication uses the same stock order in both matrices. The
    # saved target vector is fixed over the ENTIRE descriptive evaluation.
    # Each month's start restores that vector: R_p,t = sum_i w_i * R_i,t.
    # There is no buy-and-hold drift and no monthly parameter re-estimation.
    result = stocks[weights.index].dot(weights)
    result["DJIA price index"] = djia_returns
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Missing or nonfinite portfolio/index returns")
    if (result <= -1).any().any():
        raise ValueError("A return at or below -100% prevents positive growth-of-1 plotting")
    if len(result) != len(prices) - 1:
        raise ValueError("The common sample has lost observations")
    return result


def performance_table(returns):
    # All calculations use decimal returns, including variance in utility.
    # Only display columns labelled (%) are multiplied by 100 afterwards.
    mean = returns.mean()
    variance = returns.var(ddof=1)  # Sample variance, denominator n-1.
    volatility = np.sqrt(variance)
    # pandas skew is the bias-corrected Fisher-Pearson sample skewness.
    # Skewness and Sharpe are undefined for constant series: show NaN, not zero.
    valid_sd = volatility > 0
    sharpe = mean.div(volatility.where(valid_sd))  # Monthly RF is zero.
    skew = returns.skew().where(valid_sd & (len(returns) >= 3))
    utility = mean - (GAMMA / 2) * variance
    table = pd.DataFrame({
        "Mean monthly (%)": 100 * mean,
        "Monthly SD (%)": 100 * volatility,
        "Skewness": skew,
        "Worst month (%)": 100 * returns.min(),
        "Worst date": returns.idxmin().astype(str),
        "Monthly Sharpe": sharpe,
        "Monthly utility (decimal)": utility,
        "N": len(returns),
    })
    table.index.name = "Series"
    return table.sort_values("Monthly utility (decimal)", ascending=False, kind="stable")


def growth_and_plot(returns):
    # Initial wealth is explicitly 1 at the January-2004 price observation,
    # before February's first computable return. Do not insert a fake return.
    initial_month = returns.index[0] - 1
    initial = pd.DataFrame(1.0, index=pd.PeriodIndex([initial_month], freq="M"),
                           columns=returns.columns)
    wealth = pd.concat([initial, (1 + returns).cumprod()])
    wealth.index.name = "Month"
    if not np.isfinite(wealth.to_numpy()).all():
        raise ValueError("Nonfinite compounded wealth")
    # Show all eleven lines twice: linear scale for absolute wealth and log
    # scale for the lower-valued paths. No arbitrary y-limits or clipping.
    fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, 11))
    dates = wealth.index.to_timestamp(how="end")
    for number, name in enumerate(wealth.columns):
        style = {"color": "black", "linestyle": "--", "linewidth": 2} if name == "DJIA price index" else {
            "color": colors[number], "linewidth": 1.6}
        for ax in axes:
            ax.plot(dates, wealth[name], label=name, **style)
    axes[0].set_title("Linear scale: absolute compounded wealth")
    axes[1].set_title("Logarithmic scale: all paths remain visible")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_ylabel("Growth of 1 (price returns only)")
        ax.grid(alpha=0.25, which="major")
        ax.margins(y=0.06)
    axes[1].set_xlabel("Date")
    fig.suptitle("Part 3 Q3: full-sample DESCRIPTIVE performance\n"
                 f"Returns {returns.index[0]} to {returns.index[-1]} | fixed targets, monthly rebalancing, zero costs\n"
                 "Original column alignment and supplied-price returns per professor Q4/Q5; includes future-information weights", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 0.92))
    path = HERE / "Part3_3_growth_of_1.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return wealth, path


def explain(table, wealth):
    winner = table.index[0]
    row = table.loc[winner]
    print("\nOBSERVED PERFORMANCE -- HINDSIGHT, NOT AN EX-ANTE RANKING")
    print(f"Highest retrospective monthly utility: {winner}, U={row['Monthly utility (decimal)']:.6f}.")
    print(f"Its monthly mean is {row['Mean monthly (%)']:.3f}%, SD {row['Monthly SD (%)']:.3f}%,")
    print(f"Sharpe {row['Monthly Sharpe']:.3f}, skewness {row['Skewness']:.3f}, and worst month")
    print(f"{row['Worst month (%)']:.3f}% in {row['Worst date']}. Final wealth: {wealth[winner].iloc[-1]:.3f}.")
    high_return = table["Mean monthly (%)"].idxmax()
    low_risk = table["Monthly SD (%)"].idxmin()
    worst = table["Worst month (%)"].idxmin()
    print(f"Highest arithmetic monthly mean: {high_return} ({table.loc[high_return, 'Mean monthly (%)']:.3f}%).")
    print(f"Lowest monthly volatility: {low_risk} ({table.loc[low_risk, 'Monthly SD (%)']:.3f}%).")
    print(f"Most severe single-month loss: {worst}, {table.loc[worst, 'Worst month (%)']:.3f}% in "
          f"{table.loc[worst, 'Worst date']}.")
    negative_skew = table.index[table["Skewness"] < 0].tolist()
    print("Negative sample skewness: " + (", ".join(negative_skew) or "none") + ".")
    print("Negative skewness indicates more pronounced left-tail asymmetry in this sample;")
    print("positive skewness indicates the reverse. Neither establishes a future tail distribution.")
    print("Minimum monthly return is one observed month, not peak-to-trough drawdown.")
    print("Utility trades monthly mean against 2.5 times monthly variance; it does not")
    print("incorporate skewness or other tail risks. A high final wealth need not maximize utility.")
    benchmark = table.loc["DJIA price index"]
    print(f"DJIA price index: monthly mean {benchmark['Mean monthly (%)']:.3f}%, SD "
          f"{benchmark['Monthly SD (%)']:.3f}%, utility {benchmark['Monthly utility (decimal)']:.6f}.")
    print("DJIA is a separate price-weighted index, not an ETF or a supplied total-return investment.")

    print("\nWHAT COULD BE CHOSEN ON JANUARY 1, 2004?")
    print(f"Hypothetical hindsight choice under the stated utility: {winner}.")
    print("This does not establish that it could have been selected or constructed on January 1, 2004.")
    print("P5/P6/P7/P8/P10 use means, covariances or volatilities through June 2014;")
    print("P1 now uses January 2004 shares and January 2004 month-end prices, as corrected by the professor.")
    print("Those month-end prices were unavailable on January 1, 2004. P2 uses the same January 2004")
    print("shares with June 2014 prices; its prices were unavailable throughout the earlier sample.")
    print("The full-sample curves of P2 and the estimated portfolios remain retrospective calculations.")
    print("P3/P4 (alphabetical halves) and P9 (equal group budgets) could in principle be")
    print("specified without future return estimates, if the universe, names and groups were known then.")
    print("The file does not establish those historical classifications or constituent availability.")
    print("A DJIA-tracking idea is also distinct from an executable investment in the supplied index levels.")
    print("There is NO unique defensible ex-ante utility-maximizing choice from these data alone.")
    print("An ex-ante decision needs pre-2004 return history/estimates, point-in-time constituents")
    print("and classifications, and tradable prices before/at the decision. December 2003 prices")
    print("would also be needed to measure January 2004's return. January's supplied month-end")
    print("price is not known on January 1. These are starting requirements, not facts supplied here.")


def main():
    # Print the PDF's evidence and the assumptions needed where inputs are absent.
    import sys
    project_root = next(p for p in Path(__file__).resolve().parents if (p / "Files_Homework").is_dir())
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from homework_assumptions import print_assumptions
    print_assumptions("P3Q3")
    print("PART 3, QUESTION 3: PERFORMANCE MEASURES")
    metadata, prices, estimation, flags = q1.load_data()
    # Reuse Q2's provenance checks: all thirty stock weights, budget/positivity,
    # company matching, corrected cap formulas, and reproduction of Q1's optimizations.
    weights, validation = q2.verify_weights(metadata, prices, estimation)
    returns = portfolio_returns(prices, weights)
    print(f"\nActual common return sample: {returns.index[0]} to {returns.index[-1]}, {len(returns)} months.")
    print(f"Initial price/wealth observation: {prices.index[0]}. No January 2004 return is invented.")
    print("All ten weight vectors remain fixed; restore targets monthly, with no transaction costs.")
    print("This is the requested full-sample DESCRIPTIVE comparison; look-ahead is not removed by matching dates.")
    print("\nWEIGHT VALIDATION")
    print(validation[["Weight sum", "Negative weights"]].to_string(float_format=lambda v: f"{v:.12f}"))
    print("\nPRICE-QUALITY FLAGS (absolute stock return >40%; supplied prices unchanged)")
    print(flags.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("No missing/nonpositive values or calendar gaps were found.")
    print("Professor Q4: retain original dates and column alignment, including DJIA and CRM.")
    print("Professor Q5: calculate P_t/P_(t-1)-1 directly from the supplied prices.")
    print("No dividends, coupons, distributions, external price adjustments or date corrections are added.")
    print("Large supplied-price moves remain in the calculations, as required; flags are descriptive only.")
    print("Historical fixed-universe availability remains unverified. The price-return policy is resolved.")
    table = performance_table(returns)
    print("\nMONTHLY PERFORMANCE, SORTED BY UTILITY (gamma=5, RF=0)")
    print("Mean, SD and worst return: percent. SD/variance use ddof=1 (sample convention).")
    print("Skewness: bias-corrected Fisher-Pearson sample skewness; Sharpe and skewness are unitless.")
    print("Sharpe = mean(decimal R)/SD(decimal R). Utility = mean(decimal R)-2.5*variance(decimal R).")
    print("Utility is reported in monthly DECIMAL return units, not percent. Undefined metrics appear as NaN.")
    print(table.to_string(float_format=lambda v: f"{v:.6f}", na_rep="NaN (undefined)"))
    wealth, plot_path = growth_and_plot(returns)
    explain(table, wealth)
    outputs = {"performance": table, "returns_decimal": returns, "wealth": wealth,
               "weight_validation": validation, "price_flags": flags}
    for suffix, frame in outputs.items():
        frame.to_csv(HERE / f"Part3_3_{suffix}.csv", index=suffix != "price_flags", na_rep="NaN")
    html = """<!doctype html><html><head><meta charset="utf-8"><title>Part 3 Q3 performance</title>
<style>body{font:14px Arial;margin:24px}table{border-collapse:collapse;white-space:nowrap}
th,td{border:1px solid #ddd;padding:8px;text-align:right}thead{background:#eaf0f5}</style></head><body>
<h1>Part 3 Q3: retrospective performance</h1><p>February 2004–December 2024; 251 returns.
Fixed Q1 targets, costless monthly rebalancing, RF=0. Includes future-information weights.</p>
<p>Professor Q4/Q5: original dates and alignment, including DJIA and CRM, are retained.
Returns are calculated directly from supplied prices; no additional dividend or coupon data are required.</p>
<p>Monthly mean, sample SD and worst return are percent; utility is DECIMAL (mean - 2.5 variance).
Skewness is adjusted Fisher-Pearson; Sharpe is monthly. Results are supplied-price returns.</p>"""
    html += table.to_html(float_format=lambda v: f"{v:.6f}", na_rep="NaN (undefined)")
    html += '<p><img src="Part3_3_growth_of_1.png" style="max-width:100%" alt="Growth of 1: linear and log scales"></p></body></html>'
    (HERE / "Part3_3_performance.html").write_text(html, encoding="utf-8")
    print(f"\nGrowth-of-1 plot: {plot_path}")
    print(f"Performance table: {HERE / 'Part3_3_performance.csv'}")
    print(f"Readable table and chart: {HERE / 'Part3_3_performance.html'}")
    print(f"Full report: {HERE / 'Part3_3_results.txt'}")


if __name__ == "__main__":
    report = StringIO()
    with redirect_stdout(report):
        main()
    text = report.getvalue()
    print(text, end="")
    (HERE / "Part3_3_results.txt").write_text(text, encoding="utf-8")
