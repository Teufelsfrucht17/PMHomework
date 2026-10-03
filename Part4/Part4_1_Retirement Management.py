"""Part 4: seven retirement portfolios, on one common ETF price-return sample.

Run with the project's .venv interpreter, or from the project root:
.venv/Scripts/python.exe "Part4/Part4_1_Retirement Management.py"
Requires numpy, pandas and matplotlib. Only HW_ETFs.csv supplies financial data.
All results are saved in Part4. Professor Q4/Q5 requires original alignment
(including PSP) and returns from supplied prices without added distributions.
"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "Files_Homework" / "HW_ETFs.csv"
TICKERS = ["SPY", "IWM", "FEZ", "EEM", "TLT", "AGG", "JNK", "LQD",
           "MTUM", "SIZE", "VLUE", "USMV", "USRT", "QAI", "IGF", "PSP"]
PORTFOLIOS = ["P1 Baseline 60/40", "P2 Regional equities", "P3 Three bonds",
              "P4 Four equities/four bonds", "P5 Equity factors",
              "P6 Alternatives", "P7 Equal weight"]
RF_ANNUAL = 0.01
RF_MONTHLY = (1 + RF_ANNUAL)**(1 / 12) - 1


def load_etfs():
    raw = pd.read_csv(SOURCE, dtype=str)
    descriptions = list(raw.columns[1:])
    # Extract each ticker from parentheses; trailing '$' is part of the
    # descriptive header, not a return unit and not part of the ticker.
    mapped = []
    for description in descriptions:
        matches = re.findall(r"\(([A-Z]+)\)", description)
        if len(matches) != 1:
            raise ValueError(f"Cannot unambiguously identify ticker: {description}")
        mapped.append(matches[0])
    if len(mapped) != 16 or len(set(mapped)) != 16 or set(mapped) != set(TICKERS):
        raise ValueError(f"Expected sixteen specified ETF tickers; found {mapped}")
    dates = raw.iloc[:, 0].str.strip()
    if not dates.str.fullmatch(r"\d{6}", na=False).all():
        raise ValueError("ETF dates must have YYYYMM format")
    parsed = pd.to_datetime(dates, format="%Y%m", errors="coerce")
    if parsed.isna().any():
        raise ValueError("Invalid monthly dates")
    numeric = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    # Missing pre-inception prices are legitimate; malformed nonempty cells
    # are not silently converted into acceptable missing values.
    invalid = raw.iloc[:, 1:].notna() & numeric.isna()
    if invalid.any().any():
        raise ValueError("Nonempty nonnumeric price cells")
    numeric.columns = mapped
    numeric.index = pd.PeriodIndex(parsed, freq="M", name="Month")
    prices = numeric[TICKERS].sort_index()
    if prices.index.has_duplicates:
        raise ValueError("Duplicate dates")
    if not prices.index.equals(pd.period_range(prices.index[0], prices.index[-1], freq="M")):
        raise ValueError("Missing calendar months in the price table")
    if (prices <= 0).any().any() or np.isinf(prices.to_numpy()).any():
        raise ValueError("Nonpositive or infinite observed prices")
    audit = []
    for ticker in TICKERS:
        series = prices[ticker]
        first, last = series.first_valid_index(), series.last_valid_index()
        if first is None:
            raise ValueError(f"No observed prices for {ticker}")
        # Reject internal holes: do not bridge them or forward-fill them.
        if series.loc[first:last].isna().any():
            raise ValueError(f"Missing prices within {ticker}'s observed history")
        audit.append({"Ticker": ticker, "Description": descriptions[mapped.index(ticker)],
                      "First price": first, "Last price": last,
                      "Missing prices": int(series.isna().sum())})
    # Professor Q4: preserve PSP and all other columns on their supplied dates.
    # Q5: simple returns from supplied prices; no external distribution data.
    # Price levels are NOT returns. fill_method=None prevents creating returns
    # from pre-inception missing values or carrying prices forward.
    all_returns = prices.pct_change(fill_method=None)
    common = all_returns.dropna(how="any")
    if len(common) < 2 or not np.isfinite(common.to_numpy()).all():
        raise ValueError("Insufficient or invalid common monthly returns")
    expected = pd.period_range(common.index[0], common.index[-1], freq="M")
    if not common.index.equals(expected):
        raise ValueError("Common returns contain a calendar gap")
    # Confirm each common return has both its current and previous-month price.
    if prices.loc[common.index - 1].isna().any().any():
        raise ValueError("Missing lagged prices for a common return")
    audit = pd.DataFrame(audit).set_index("Ticker")
    audit["First return"] = [all_returns[t].first_valid_index() for t in audit.index]
    if audit.loc["USRT", "First price"] == pd.Period("2016-11", freq="M"):
        if common.index[0] != pd.Period("2016-12", freq="M"):
            raise ValueError("Common start differs from the expected December 2016; investigate")
    return common, audit, prices


def target_weights():
    # All target fractions are of the TOTAL portfolio. In P2 the regional
    # proportions 50/25/25 are multiplied by the 60% equity sleeve.
    weights = pd.DataFrame(0.0, index=pd.Index(TICKERS, name="Ticker"), columns=PORTFOLIOS)
    weights.loc[["SPY", "TLT"], PORTFOLIOS[0]] = [0.60, 0.40]
    weights.loc[["IWM", "FEZ", "EEM", "TLT"], PORTFOLIOS[1]] = [0.30, 0.15, 0.15, 0.40]
    weights.loc["SPY", PORTFOLIOS[2]] = 0.60
    weights.loc[["AGG", "JNK", "LQD"], PORTFOLIOS[2]] = 0.40 / 3
    weights.loc[["SPY", "IWM", "FEZ", "EEM"], PORTFOLIOS[3]] = 0.60 / 4
    weights.loc[["TLT", "AGG", "JNK", "LQD"], PORTFOLIOS[3]] = 0.40 / 4
    weights.loc[["MTUM", "SIZE", "VLUE", "USMV"], PORTFOLIOS[4]] = 0.60 / 4
    weights.loc["TLT", PORTFOLIOS[4]] = 0.40
    weights.loc["SPY", PORTFOLIOS[5]] = 0.60
    weights.loc[["USRT", "QAI", "IGF", "PSP"], PORTFOLIOS[5]] = 0.40 / 4
    weights[PORTFOLIOS[6]] = 1 / 16
    if (weights < 0).any().any():
        raise ValueError("Negative target weight")
    np.testing.assert_allclose(weights.sum(), 1, rtol=0, atol=1e-14)
    return weights


def measure_performance(returns):
    # Initial wealth 1 is present BEFORE the first investment month, so a
    # first-month loss counts toward drawdown. Compound realized portfolio
    # returns, not constituent price levels or arithmetic cumulative sums.
    initial = pd.DataFrame(1.0, index=pd.PeriodIndex([returns.index[0] - 1], freq="M"),
                           columns=returns.columns)
    wealth = pd.concat([initial, (1 + returns).cumprod()])
    wealth.index.name = "Month"
    rows = []
    for name in returns:
        r = returns[name]
        sd = r.std(ddof=1)  # Sample SD with denominator n-1.
        drawdown = wealth[name] / wealth[name].cummax() - 1
        # All calculations below use decimals; percentage columns multiply by
        # 100 only for display. CAGR is separate from arithmetic annualization.
        rows.append({"Portfolio": name,
                     "Mean monthly (%)": 100 * r.mean(),
                     "Annual arithmetic (%)": 100 * 12 * r.mean(),
                     "CAGR (%)": 100 * (wealth[name].iloc[-1]**(12 / len(r)) - 1),
                     "Annual volatility (%)": 100 * np.sqrt(12) * sd,
                     "Max drawdown (%)": 100 * drawdown.min(),
                     "Annual Sharpe": np.sqrt(12) * (r - RF_MONTHLY).mean() / sd if sd > 0 else np.nan,
                     "Final wealth": wealth[name].iloc[-1], "N": len(r)})
    return pd.DataFrame(rows).set_index("Portfolio"), wealth


def save_plots(wealth, performance):
    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
    colors = plt.get_cmap("tab10").colors
    for i, name in enumerate(wealth):
        # Emphasize the baseline, while keeping all seven paths visible.
        ax.plot(wealth.index.to_timestamp(how="end"), wealth[name], label=name,
                color="black" if i == 0 else colors[i], linewidth=2.8 if i == 0 else 1.7)
    ax.set_title(f"Part 4: retirement portfolios, PRICE-return growth of 1\n"
                 f"{wealth.index[1]} to {wealth.index[-1]} | common sample, monthly rebalancing, zero costs")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth (initial investment = 1)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
    growth_path = HERE / "Part4_1_growth_of_1.png"
    fig.savefig(growth_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    metrics = ["Annual arithmetic (%)", "Annual volatility (%)", "Max drawdown (%)", "Annual Sharpe"]
    for ax, metric in zip(axes.flat, metrics):
        values = performance[metric]
        bars = ax.barh(np.arange(7), values, color=["black", *colors[1:7]])
        ax.set_yticks(np.arange(7), [name.split(" ", 1)[0] for name in performance.index])
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.axvline(0, color="gray", linewidth=0.6)
        ax.set_title(metric)
        ax.margins(x=0.25)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Common-sample PRICE-return comparison | P1 is the 60/40 baseline\n"
                 "P2 Regional; P3 Three bonds; P4 Four/four; P5 Factors; P6 Alternatives; P7 Equal weight")
    comparison_path = HERE / "Part4_1_metric_comparison.png"
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)
    return growth_path, comparison_path


def explain(performance):
    baseline = performance.loc[PORTFOLIOS[0]]
    print("\nDATA-DRIVEN COMPARISON WITH THE 60/40 BASELINE")
    print(f"Baseline CAGR {baseline['CAGR (%)']:.2f}%, annual volatility "
          f"{baseline['Annual volatility (%)']:.2f}%, max drawdown {baseline['Max drawdown (%)']:.2f}%, "
          f"Sharpe {baseline['Annual Sharpe']:.3f}.")
    exposures = [
        "P2 replaces US large-cap equity with US small caps (IWM), euro-area equities (FEZ) and emerging markets (EEM), retaining 40% TLT.",
        "P3 replaces long-duration Treasuries with aggregate (AGG), high-yield (JNK) and investment-grade corporate bonds (LQD). Duration and credit risks differ.",
        "P4 spreads equity across four regions/size exposures and bonds across all four bond ETFs; TLT falls to 10% of the portfolio.",
        "P5 holds momentum, size, value and minimum-volatility equity tilts; these remain equity exposures, with 40% TLT unchanged.",
        "P6 replaces bonds with REITs (USRT), a hedge-strategy tracker (QAI), infrastructure (IGF) and listed private equity (PSP). These are not bond substitutes with guaranteed protection.",
        "P7 holds 25% regional/broad equities, 25% equity factor ETFs, 25% bond ETFs and 25% alternatives. It is not a 60/40 allocation; alternatives can carry equity-like risk.",
    ]
    for name, exposure in zip(PORTFOLIOS[1:], exposures):
        row = performance.loc[name]
        print(f"\n{name}: annual arithmetic return difference {row['Annual arithmetic (%)'] - baseline['Annual arithmetic (%)']:+.2f} pp;")
        print(f"  CAGR difference {row['CAGR (%)']-baseline['CAGR (%)']:+.2f} pp; volatility difference "
              f"{row['Annual volatility (%)']-baseline['Annual volatility (%)']:+.2f} pp;")
        print(f"  drawdown {row['Max drawdown (%)']:.2f}% versus {baseline['Max drawdown (%)']:.2f}%; "
              f"Sharpe {row['Annual Sharpe']:.3f} versus {baseline['Annual Sharpe']:.3f}.")
        print("  " + exposure)
    print("\nExposures offer plausible reasons for differences: regional equity performance,")
    print("interest-rate sensitivity of long-duration Treasuries, corporate credit risk,")
    print("factor tilts and equity-like alternative risks. This comparison does not isolate causal effects.")
    best = performance["Annual Sharpe"].idxmax()
    fastest = performance["CAGR (%)"].idxmax()
    shallowest = performance["Max drawdown (%)"].idxmax()
    deepest = performance["Max drawdown (%)"].idxmin()
    print(f"Highest observed Sharpe: {best}; highest price CAGR: {fastest}.")
    print(f"Smallest drawdown magnitude: {shallowest} ({performance.loc[shallowest, 'Max drawdown (%)']:.2f}%);")
    print(f"largest: {deepest} ({performance.loc[deepest, 'Max drawdown (%)']:.2f}%).")
    print("Holding more ETFs did not automatically establish better diversification; assess")
    print("the actual volatility and drawdown, not just the number of holdings.")
    print("For retirement, large losses can be especially damaging when withdrawals force")
    print("sales during a downturn. This growth-of-1 exercise has no contributions or")
    print("withdrawals and does not simulate retirement withdrawal outcomes.")
    print("The common period excludes earlier market episodes; observed rankings are not")
    print("guarantees of future performance or a personalized portfolio recommendation.")
    print("Professor Q4/Q5 explicitly requires the supplied prices and original column alignment.")
    print("PSP retains its original row dates; no two-month correction is applied.")
    print("Results are PRICE returns as requested; dividend, coupon and distribution data are not required.")


def main():
    # Print the PDF's evidence and the assumptions needed where inputs are absent.
    import sys
    project_root = next(p for p in Path(__file__).resolve().parents if (p / "Files_Homework").is_dir())
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from homework_assumptions import print_assumptions
    print_assumptions("P4")
    print("PART 4: RETIREMENT MANAGEMENT")
    common, audit, prices = load_etfs()
    weights = target_weights()
    print(f"Price table: {prices.index[0]} to {prices.index[-1]}.")
    print("\nETF MAPPING AND AVAILABILITY (missing pre-inception prices are not filled)")
    print(audit.to_string())
    print(f"\nCOMMON RETURN SAMPLE: {common.index[0]} to {common.index[-1]}, {len(common)} months for ALL seven portfolios.")
    print("USRT first price: November 2016; first common return confirmed as December 2016.")
    print("No internal missing prices, date duplicates, calendar gaps or nonpositive observed prices.")
    print("Professor Q4/Q5: original ETF alignment, including PSP, is retained; all returns use supplied prices.")
    print("\nTARGET WEIGHTS (fractions; all zero holdings explicitly shown)")
    print(weights.to_string(float_format=lambda v: f"{v:.6f}"))
    print("Column sums:")
    print(weights.sum().to_string(float_format=lambda v: f"{v:.12f}"))
    # Rebalance to the same targets before every month, then apply that month's
    # ETF returns. This implements costless rebalancing, not drifting holdings.
    returns = common[TICKERS].dot(weights)
    if not np.isfinite(returns.to_numpy()).all() or (returns <= -1).any().any():
        raise ValueError("Invalid portfolio returns")
    performance, wealth = measure_performance(returns)
    print("\nCONVENTIONS: decimal returns internally; monthly rebalancing, zero costs.")
    print("Annual arithmetic mean = 12*monthly mean; CAGR is a separate compounded growth rate.")
    print("Annual volatility = sqrt(12)*sample monthly SD (ddof=1). These annualizations do not adjust for serial dependence.")
    print(f"RF annual = 1%; RF monthly = (1.01)^(1/12)-1 = {RF_MONTHLY:.9f} decimal ({100*RF_MONTHLY:.6f}%).")
    print("Annual Sharpe = sqrt(12)*mean(R-RF_monthly)/SD(R). Initial wealth of 1 is included in drawdowns.")
    print("\nCOMMON-SAMPLE PRICE-RETURN PERFORMANCE (percent columns labelled; Sharpe unitless)")
    print(performance.to_string(float_format=lambda v: f"{v:.4f}", na_rep="NaN (undefined)"))
    plot_paths = save_plots(wealth, performance)
    explain(performance)
    outputs = {"weights": weights, "performance": performance, "ETF_availability": audit,
               "ETF_returns_decimal": common, "portfolio_returns_decimal": returns, "wealth": wealth}
    for suffix, frame in outputs.items():
        frame.to_csv(HERE / f"Part4_1_{suffix}.csv", na_rep="NaN")
    html = """<!doctype html><html><head><meta charset="utf-8"><title>Retirement portfolios</title>
<style>body{font:14px Arial;margin:24px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd;text-align:right}thead{background:#eaf0f5}</style>
</head><body><h1>Part 4: retirement portfolios</h1><p>Supplied-price return performance, as required by professor Q5.
Professor Q4: original dates and column alignment are retained, including PSP.
No additional dividend, coupon or distribution data are required.
Common sample December 2016–June 2026; 115 months. Monthly rebalancing; zero costs; 1% annual RF.</p>
<h2>Target weights (fractions)</h2>"""
    html += weights.to_html(float_format=lambda v: f"{v:.6f}")
    html += "<h2>Performance</h2>" + performance.to_html(float_format=lambda v: f"{v:.4f}")
    for path in plot_paths:
        html += f'<p><img src="{path.name}" style="max-width:100%" alt="Portfolio comparison"></p>'
    (HERE / "Part4_1_results.html").write_text(html + "</body></html>", encoding="utf-8")
    print("\nSAVED OUTPUTS")
    for path in plot_paths:
        print(path)
    for suffix in outputs:
        print(HERE / f"Part4_1_{suffix}.csv")
    print(HERE / "Part4_1_results.html")
    print(HERE / "Part4_1_results.txt")


if __name__ == "__main__":
    report = StringIO()
    with redirect_stdout(report):
        main()
    text = report.getvalue()
    print(text, end="")
    (HERE / "Part4_1_results.txt").write_text(text, encoding="utf-8")
