"""Part 5 Question 1: thirty-year buy-and-hold policy portfolios.

Run with the project .venv interpreter, or from the project root:
.venv/Scripts/python.exe "Part5/Part5_1_Policy Portfolios/Part5_1_Policy Portfolios.py"
Requires numpy, pandas and matplotlib. Only HW_Factors.csv supplies data.
Results are historical scenarios, not forecasts or currency-converted outcomes.
"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "Files_Homework" / "HW_Factors.csv"
INITIAL = 100_000.0
HORIZON = 360
EQUITY_WEIGHTS = np.arange(11) / 10
STARTS = pd.period_range("1927-01", "1996-04", freq="M", name="Windfall month")
BENCHMARK_MONTHLY = 1.01**(1 / 12) - 1


def load_returns():
    raw = pd.read_csv(SOURCE, dtype=str)
    dates = raw.iloc[:, 0].str.strip()
    if not dates.str.fullmatch(r"\d{6}", na=False).all():
        raise ValueError("Factor dates must be YYYYMM")
    parsed = pd.to_datetime(dates, format="%Y%m", errors="coerce")
    if parsed.isna().any():
        raise ValueError("Failed factor date parsing")
    factors = raw[["Mkt-RF", "RF"]].apply(pd.to_numeric, errors="coerce")
    factors.index = pd.PeriodIndex(parsed, freq="M", name="Month")
    factors = factors.sort_index()
    if factors.index.has_duplicates or not np.isfinite(factors.to_numpy()).all():
        raise ValueError("Duplicate dates or missing/nonfinite factor values")
    expected = pd.period_range(factors.index[0], factors.index[-1], freq="M")
    if not factors.index.equals(expected):
        raise ValueError("Missing calendar months in factor data")
    # Both input columns are percentage points. Reconstruct TOTAL market return
    # first, then divide by 100 exactly once for decimal compounding.
    returns = pd.DataFrame({"Market total": (factors["Mkt-RF"] + factors["RF"]) / 100,
                            "Risk-free proxy": factors["RF"] / 100})
    if (returns <= -1).any().any():
        raise ValueError("A sleeve return is at/below -100%; positive wealth cannot be assumed")
    if len(STARTS) != 832 or not STARTS.isin(returns.index).all():
        raise ValueError("Expected all 832 windfall dates from January 1927 to April 1996")
    required = pd.period_range(STARTS[0] + 1, STARTS[-1] + HORIZON, freq="M")
    if not required.isin(returns.index).all():
        raise ValueError("The factor file does not cover every required subsequent 360-month window")
    return returns


def scenario_paths(window):
    """Compound two sleeves without trading between them; return all 11 paths."""
    if len(window) != HORIZON:
        raise ValueError("Each scenario must contain exactly 360 subsequent returns")
    # Row zero is wealth at the windfall month's END, before any return is earned.
    # Thereafter each sleeve compounds its OWN returns. No target-weight reset.
    growth = np.vstack([np.ones(2), np.cumprod(1 + window.to_numpy(), axis=0)])
    equity = INITIAL * growth[:, [0]] * EQUITY_WEIGHTS
    risk_free = INITIAL * growth[:, [1]] * (1 - EQUITY_WEIGHTS)
    wealth = equity + risk_free
    actual_returns = wealth[1:] / wealth[:-1] - 1
    # Beginning-of-month realized weights drift with relative sleeve values.
    drift_weights = equity / wealth
    weighted_check = (drift_weights[:-1] * window["Market total"].to_numpy()[:, None]
                      + (1 - drift_weights[:-1]) * window["Risk-free proxy"].to_numpy()[:, None])
    np.testing.assert_allclose(actual_returns, weighted_check, rtol=1e-10, atol=1e-14)
    # Endpoint validation checks the ENTIRE path, not just terminal wealth.
    np.testing.assert_allclose(actual_returns[:, 0], window["Risk-free proxy"], atol=1e-14)
    np.testing.assert_allclose(actual_returns[:, -1], window["Market total"], atol=1e-14)
    np.testing.assert_allclose(wealth[:, 0], INITIAL * growth[:, 1], rtol=1e-12)
    np.testing.assert_allclose(wealth[:, -1], INITIAL * growth[:, 0], rtol=1e-12)
    return wealth, actual_returns, drift_weights


def calculate_scenarios(returns):
    records = []
    example = None
    for start in STARTS:
        months = pd.period_range(start + 1, start + HORIZON, freq="M")
        window = returns.loc[months]
        # Explicit checks exclude the windfall month's own return and include
        # exactly the next thirty years. No centered or overlapping signal logic.
        if len(window) != 360 or window.index[0] != start + 1 or window.index[-1] != start + 360:
            raise ValueError(f"Invalid forward window for {start}")
        wealth, actual, drift = scenario_paths(window)
        mean = actual.mean(axis=0)
        sd = actual.std(axis=0, ddof=1)  # Sample SD, denominator 359.
        # The 1% benchmark is used ONLY here. RF actually earned by the
        # risk-free sleeve is the historical, time-varying series in the file.
        sharpe = np.full(11, np.nan)
        nonzero_sd = sd > 1e-15  # Numerical tolerance for a genuinely constant path.
        sharpe[nonzero_sd] = np.sqrt(12) * (mean[nonzero_sd] - BENCHMARK_MONTHLY) / sd[nonzero_sd]
        cagr = (wealth[-1] / INITIAL)**(1 / 30) - 1
        for i, weight in enumerate(EQUITY_WEIGHTS):
            records.append({
                "Windfall month": start, "First return month": months[0],
                "Last return month": months[-1], "Starting equity weight": weight,
                "Terminal balance": wealth[-1, i],
                "Annual arithmetic return (%)": 100 * 12 * mean[i],
                "Annual volatility (%)": 100 * np.sqrt(12) * sd[i],
                "Annual Sharpe": sharpe[i], "CAGR (%)": 100 * cagr[i],
                "Final equity weight": drift[-1, i], "N months": len(actual),
            })
        if start == STARTS[0]:
            # Independent intermediate check: multiply each set of gross
            # returns directly, then add the two starting sleeve balances.
            check = (INITIAL * 0.6 * np.prod(1 + window["Market total"])
                     + INITIAL * 0.4 * np.prod(1 + window["Risk-free proxy"]))
            np.testing.assert_allclose(wealth[-1, 6], check, rtol=1e-12)
            example = pd.DataFrame({"Wealth": wealth[:, 6], "Equity weight": drift[:, 6],
                                    "Risk-free weight": 1 - drift[:, 6]},
                                   index=pd.period_range(start, start + HORIZON, freq="M", name="Month"))
    results = pd.DataFrame(records)
    if len(results) != 832 * 11 or not (results["N months"] == 360).all():
        raise ValueError("Unexpected number or length of scenarios")
    return results, example


def summarize(results):
    # Equal probability over the 832 specified historical starting months.
    # Crucially, average the per-path Sharpes already computed above; do not
    # divide an average return by an average standard deviation instead.
    summary = results.groupby("Starting equity weight", sort=True).agg(
        **{"Mean terminal balance": ("Terminal balance", "mean"),
           "Median terminal balance": ("Terminal balance", "median"),
           "10th percentile balance": ("Terminal balance", lambda x: x.quantile(0.10)),
           "90th percentile balance": ("Terminal balance", lambda x: x.quantile(0.90)),
           "Mean annual arithmetic (%)": ("Annual arithmetic return (%)", "mean"),
           "Mean annual volatility (%)": ("Annual volatility (%)", "mean"),
           "Mean annual Sharpe": ("Annual Sharpe", "mean"),
           "Mean CAGR (%)": ("CAGR (%)", "mean"),
           "Scenarios": ("Terminal balance", "size"),
           "Defined Sharpes": ("Annual Sharpe", "count")})
    if not (summary["Scenarios"] == 832).all():
        raise ValueError("Every allocation must have 832 scenarios")
    return summary


def save_plots(results):
    specifications = [
        ("Terminal balance", "30-year terminal balance (millions of starting monetary units)",
         "terminal_balance", 1e6),
        ("Annual arithmetic return (%)", "Annualized arithmetic return (%)", "annual_return", 1),
        ("Annual volatility (%)", "Annualized volatility (%)", "annual_volatility", 1),
        ("Annual Sharpe", "Annualized Sharpe (constant 1% annual benchmark)", "annual_sharpe", 1),
    ]
    paths = []
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, 11))
    for metric, ylabel, suffix, divisor in specifications:
        values = results.pivot(index="Windfall month", columns="Starting equity weight", values=metric)
        fig, ax = plt.subplots(figsize=(13, 7))
        for i, weight in enumerate(EQUITY_WEIGHTS):
            ax.plot(values.index.to_timestamp(how="end"), values[weight] / divisor,
                    color=colors[i], linewidth=1.8 if i in [0, 10] else 1.1,
                    label=f"{weight:.0%} equity / {1-weight:.0%} RF")
        ax.set_title(f"Part 5 Q1: {metric} by windfall date\n"
                     "Each start: next 360 returns; buy and hold with drifting weights")
        ax.set_xlabel("Windfall month (investment at month end)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        path = HERE / f"Part5_1_{suffix}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    # Heatmap supplements eleven overlapping terminal-balance curves, keeping
    # every allocation and historical start directly inspectable.
    matrix = results.pivot(index="Starting equity weight", columns="Windfall month", values="Terminal balance")
    fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
    mesh = ax.imshow(matrix.to_numpy() / 1e6, aspect="auto", origin="lower", cmap="viridis")
    ticks = np.linspace(0, len(matrix.columns)-1, 8, dtype=int)
    ax.set_xticks(ticks, [str(matrix.columns[i]) for i in ticks])
    ax.set_yticks(np.arange(11), [f"{w:.0%}" for w in matrix.index])
    ax.set_xlabel("Windfall month")
    ax.set_ylabel("Initial equity allocation")
    ax.set_title("30-year buy-and-hold terminal balances | initial capital 100,000 | no FX conversion")
    fig.colorbar(mesh, ax=ax, label="Terminal balance (millions of starting monetary units)")
    path = HERE / "Part5_1_terminal_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [*paths, path]


def explain(results, summary, example):
    print("\nINTERPRETATION")
    for weight in [0.0, 0.6, 1.0]:
        row = summary.loc[weight]
        print(f"Starting {weight:.0%} equity: mean terminal {row['Mean terminal balance']:,.0f}; "
              f"10th-90th percentiles {row['10th percentile balance']:,.0f}-{row['90th percentile balance']:,.0f}.")
        print(f"  Mean annual arithmetic return {row['Mean annual arithmetic (%)']:.2f}%, "
              f"volatility {row['Mean annual volatility (%)']:.2f}%, Sharpe {row['Mean annual Sharpe']:.3f}.")
    for column, label in [("Mean terminal balance", "mean terminal wealth"),
                          ("Mean annual volatility (%)", "mean annualized volatility")]:
        increasing = (summary[column].diff().dropna() >= 0).all()
        print(f"Increasing starting equity {'increases' if increasing else 'does not monotonically increase'} {label} in these historical scenarios.")
    best_sharpe = summary["Mean annual Sharpe"].idxmax()
    print(f"Highest average path Sharpe occurs at {best_sharpe:.0%} starting equity.")
    print("The all-RF path can have a defined historical volatility because the monthly")
    print("RF yield changes over decades. A high Sharpe against the constant 1% benchmark")
    print("is not evidence that a long-duration bond fund would earn the same result.")
    for weight in [0.6, 1.0]:
        subset = results[results["Starting equity weight"] == weight]
        weak = subset.loc[subset["Terminal balance"].idxmin()]
        strong = subset.loc[subset["Terminal balance"].idxmax()]
        print(f"{weight:.0%} equity weakest terminal outcome: start {weak['Windfall month']}, "
              f"end {weak['Last return month']}, balance {weak['Terminal balance']:,.0f}.")
        print(f"{weight:.0%} equity strongest: start {strong['Windfall month']}, "
              f"end {strong['Last return month']}, balance {strong['Terminal balance']:,.0f}.")
    print("These are observed historical start dates, not a market-timing prediction.")
    print("Adjacent 30-year windows overlap heavily; 832 scenarios are not 832 independent observations.")
    print("Equal weighting of historical start months is the assignment's assumption, not a future probability model.")
    print("Terminal percentiles show horizon outcomes but hide losses along the way.")
    print("A long horizon may help an investor endure volatility, but an inability to tolerate")
    print("interim losses or a need for earlier cash can make a high-equity allocation unsuitable.")
    print("\nHYPOTHETICAL CHOICE TODAY")
    print("For a hypothetical investor with a genuine 30-year horizon, separate emergency")
    print("savings and tolerance for substantial fluctuations, my illustrative preference is")
    print("60% equity / 40% risk-free initially: retain substantial growth exposure while")
    print("starting with a lower volatility profile than all-equity in this historical analysis.")
    print(f"This is a STARTING allocation: in the example it drifts to {example['Equity weight'].iloc[-1]:.1%} equity.")
    print("It does not maintain a 60/40 risk profile, guarantee a floor, or maximize every investor's utility.")
    print("Someone less able to bear losses could prefer less equity; the largest historical")
    print("average terminal balance is not automatically best. A September 2026 investment's")
    print("30-year outcome is unknown and is not contained in these data.")


def main():
    print("PART 5, QUESTION 1: POLICY PORTFOLIOS")
    returns = load_returns()
    print(f"Input months: {returns.index[0]} to {returns.index[-1]}.")
    print(f"Eligible windfall months: {STARTS[0]} to {STARTS[-1]}: {len(STARTS)} starts.")
    print(f"First window: invest end {STARTS[0]}, earn {STARTS[0]+1} through {STARTS[0]+360}.")
    print(f"Last window: invest end {STARTS[-1]}, earn {STARTS[-1]+1} through {STARTS[-1]+360}.")
    print("Each window contains exactly 360 subsequent monthly returns; no missing dates/values.")
    print("Market total = Mkt-RF + RF; percentage points converted to decimals for compounding.")
    print("Initial capital: EUR 100,000 as a numerical starting amount. Returns are USD-based;")
    print("no EUR/USD conversion is available. Balances are not currency-converted euro outcomes.")
    print("The risk-free sleeve rolls the supplied monthly RF proxy, not a long-duration bond fund.")
    print("No contributions, withdrawals or rebalancing between sleeves; weights drift.")
    print(f"Sharpe benchmark ONLY: constant annual 1%, monthly {BENCHMARK_MONTHLY:.9f} decimal.")
    results, example = calculate_scenarios(returns)
    summary = summarize(results)
    print("Edge-case path checks and independent 60/40 terminal-sleeve check passed.")
    print("\nBUY-AND-HOLD EXAMPLE: JANUARY 1927 WINDFALL, INITIAL 60/40")
    print(example.iloc[[0,-1]].to_string(float_format=lambda v: f"{v:,.6f}"))
    print("\nAVERAGES ACROSS START MONTHS (not metrics of an averaged return path)")
    print("Annual arithmetic return = 12*monthly mean; volatility = sqrt(12)*sample SD (ddof=1).")
    print("CAGR = (terminal/100000)^(1/30)-1. Percentiles use linear interpolation.")
    print("Undefined Sharpe paths remain NaN; average Sharpe uses defined paths, counted explicitly.")
    print(summary.to_string(float_format=lambda v: f"{v:,.4f}", na_rep="NaN (undefined)"))
    print(f"Undefined path Sharpes: {results['Annual Sharpe'].isna().sum()}.")
    explain(results, summary, example)
    paths = save_plots(results)
    results.to_csv(HERE / "Part5_1_per_start_results.csv", index=False, na_rep="NaN")
    summary.to_csv(HERE / "Part5_1_summary.csv", na_rep="NaN")
    example.to_csv(HERE / "Part5_1_drift_example.csv")
    html = """<!doctype html><html><head><meta charset="utf-8"><title>Policy portfolios</title>
<style>body{font:14px Arial;margin:24px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd;text-align:right}</style>
</head><body><h1>Part 5 Q1: 30-year buy-and-hold policy portfolios</h1>
<p>832 starts, January 1927–April 1996; each earns the next 360 monthly returns.
Initial amount 100,000; US-dollar-based return scenarios, no EUR/USD conversion.
Annualized arithmetic returns and CAGR are separate measures; Sharpe uses a constant 1% annual benchmark.</p>"""
    html += summary.to_html(float_format=lambda v: f"{v:,.4f}")
    for path in paths:
        html += f'<p><img src="{path.name}" style="max-width:100%" alt="Policy allocation outcomes by start month"></p>'
    (HERE / "Part5_1_results.html").write_text(html + "</body></html>", encoding="utf-8")
    print("\nSAVED FILES")
    for name in ["Part5_1_summary.csv", "Part5_1_per_start_results.csv", "Part5_1_drift_example.csv", "Part5_1_results.html", "Part5_1_results.txt"]:
        print(HERE / name)
    for path in paths:
        print(path)


if __name__ == "__main__":
    report = StringIO()
    with redirect_stdout(report):
        main()
    text = report.getvalue()
    print(text, end="")
    (HERE / "Part5_1_results.txt").write_text(text, encoding="utf-8")
