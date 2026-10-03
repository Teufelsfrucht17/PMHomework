"""Part 2: monthly international momentum, with the specified long-only weights.

Run with the project's .venv interpreter in PyCharm, or from the project root:
.venv/Scripts/python.exe "Part2/Part2_1_International Momentum.py"

Requires numpy, pandas, matplotlib and statsmodels. Only HW_World.csv and
HW_Factors.csv supply data. Output CSVs, figure and report go in Part2.
"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "Files_Homework"
COUNTRIES = ["Austria", "Australia", "Belgium", "Canada", "Denmark", "Finland",
             "France", "Germany", "HongKong", "Ireland", "Italy", "Japan",
             "Netherlands", "New Zealand", "Norway", "Singapore", "Spain",
             "Sweden", "Switzerland", "UK"]
FACTORS = ["Mkt-RF", "SMB", "HML", "Mom"]


def read_monthly_pp(filename, columns):
    """Read numerical percentage points, with explicit date/value validation."""
    frame = pd.read_csv(DATA / filename, dtype=str)
    if len(frame.columns) != len(columns) + 1 or set(frame.columns[1:]) != set(columns):
        raise ValueError(f"Unexpected columns in {filename}: {frame.columns.tolist()}")
    raw_dates = frame.iloc[:, 0].str.strip()
    if not raw_dates.str.fullmatch(r"\d{6}", na=False).all():
        raise ValueError(f"Dates must be YYYYMM in {filename}")
    dates = pd.to_datetime(raw_dates, format="%Y%m", errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Failed date parsing in {filename}")
    frame = frame[columns].apply(pd.to_numeric, errors="coerce")
    frame.index = pd.PeriodIndex(dates, freq="M", name="Month")
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate months in {filename}")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"Missing, nonnumeric or infinite values in {filename}")
    if frame.empty:
        raise ValueError(f"No observations in {filename}")
    expected = pd.period_range(frame.index.min(), frame.index.max(), freq="M")
    missing = expected.difference(frame.index)
    if len(missing):
        raise ValueError(f"Missing calendar months in {filename}: {missing.tolist()}")
    return frame


def rank_window(window_decimal):
    """Return compounded signals and deterministic, nonnegative weights."""
    # Compound 11 SIMPLE monthly returns. Summing them would be incorrect.
    if len(window_decimal) != 11:
        raise ValueError("A ranking window must have exactly eleven months")
    signal = (1 + window_decimal).prod(axis=0) - 1
    if not np.isfinite(signal.to_numpy()).all():
        raise ValueError("Nonfinite cumulative ranking signals")
    # Start alphabetically, then use a stable descending sort: exact ties keep
    # alphabetical order. In that order, the first 3 win and the last 3 lose.
    ranked = signal.sort_index().sort_values(ascending=False, kind="stable")
    # Express weights as exact integer numerators with denominator 210:
    # 4/15 = 56/210; 1/70 = 3/210. Thus 3*56 + 14*3 + 3*0 = 210 exactly.
    units = pd.Series(3, index=signal.index, dtype=int)
    units.loc[ranked.index[:3]] = 56
    units.loc[ranked.index[-3:]] = 0
    if units.sum() != 210 or (units < 0).any():
        raise ValueError("Weights do not form a fully invested long-only portfolio")
    weights = units / 210.0
    # Binary floating point cannot always represent rational weights exactly;
    # exact investment is checked above, with a tight floating-point check here.
    np.testing.assert_allclose(weights.sum(), 1, rtol=0, atol=1e-14)
    return signal, weights, ranked


def construct_strategy(world_decimal):
    """Form weights using only t-12 through t-2, then earn month-t returns."""
    if len(world_decimal) <= 12:
        raise ValueError("Insufficient history to form the first investment month")
    weights_rows, signal_rows, audit_rows = [], [], []
    trading_months = world_decimal.index[12:]
    for position in range(12, len(world_decimal)):
        month = world_decimal.index[position]
        # Python's slice endpoint is excluded: position-1 leaves the last
        # included row at position-2. Neither t-1 nor t enters this window.
        window = world_decimal.iloc[position - 12:position - 1]
        expected = pd.period_range(month - 12, month - 2, freq="M")
        if not window.index.equals(expected) or window.index.max() >= month - 1:
            raise ValueError(f"Look-ahead or incorrect ranking dates for {month}")
        signal, weights, ranked = rank_window(window)
        weights_rows.append(weights)
        signal_rows.append(signal)
        audit_rows.append({
            "Month": month, "Signal start": window.index[0],
            "Signal end": window.index[-1], "Skipped month": month - 1,
            "Top 3": ", ".join(ranked.index[:3]),
            "Bottom 3": ", ".join(ranked.index[-3:]),
        })
    weights = pd.DataFrame(weights_rows, index=trading_months)
    signals = pd.DataFrame(signal_rows, index=trading_months)
    audit = pd.DataFrame(audit_rows).set_index("Month")
    np.testing.assert_allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-14)
    if (weights < 0).any().any():
        raise ValueError("Unexpected short position")
    # Beginning-of-month weights multiply SAME-month simple returns. The next
    # month's weights are recalculated, rather than letting holdings drift.
    current_returns = world_decimal.loc[trading_months]
    portfolio = (weights * current_returns).sum(axis=1)
    equal_weight = current_returns.mean(axis=1)  # Rebalance all 20 to 5% monthly.
    returns = pd.DataFrame({"Strategy": portfolio, "Equal weight": equal_weight})

    # Independent timing check using the example explicitly requested in Q2.
    january = pd.Period("1992-01", freq="M")
    if january in weights.index:
        manual_window = world_decimal.loc["1991-01":"1991-11"]
        manual_signal, manual_weights, _ = rank_window(manual_window)
        np.testing.assert_allclose(signals.loc[january], manual_signal, rtol=0, atol=1e-14)
        np.testing.assert_allclose(weights.loc[january], manual_weights, rtol=0, atol=1e-14)
        print("Timing check passed: January 1992 uses January-November 1991; skips December 1991.")
    return returns, weights, signals, audit


def wealth_index(returns):
    # Include the initial wealth of 1 BEFORE the first investment month. This
    # ensures a loss in the first month counts toward maximum drawdown.
    if (returns <= -1).any().any():
        raise ValueError("A portfolio return is at or below -100%; inspect the data")
    initial = pd.DataFrame(1.0, index=pd.PeriodIndex([returns.index[0] - 1], freq="M"),
                           columns=returns.columns)
    return pd.concat([initial, (1 + returns).cumprod()])


def performance_table(returns, rf_decimal, wealth):
    rows = []
    for name in returns.columns:
        r = returns[name]
        excess = r - rf_decimal  # Match the risk-free return by investment month.
        monthly_sd = r.std(ddof=1)  # Sample standard deviation, n-1 denominator.
        monthly_excess_sd = excess.std(ddof=1)
        if monthly_sd <= 0 or monthly_excess_sd <= 0:
            raise ValueError(f"Zero return volatility for {name}")
        # Standard homework annualization: arithmetic mean *12, volatility
        # *sqrt(12), Sharpe sqrt(12)*mean(excess)/std(excess).
        # These square-root-of-time measures do not adjust for serial correlation.
        drawdown = wealth[name] / wealth[name].cummax() - 1
        rows.append({
            "Portfolio": name,
            "Mean monthly (%)": 100 * r.mean(),
            "Annual arithmetic (%)": 100 * 12 * r.mean(),
            "Annual volatility (%)": 100 * np.sqrt(12) * monthly_sd,
            "Max drawdown (%)": 100 * drawdown.min(),
            "Annual Sharpe": np.sqrt(12) * excess.mean() / monthly_excess_sd,
            "CAGR (%)": 100 * (wealth[name].iloc[-1] ** (12 / len(r)) - 1),
            "Final wealth": wealth[name].iloc[-1],
            "N": len(r),
        })
    return pd.DataFrame(rows).set_index("Portfolio")


def four_factor_model(returns, factors_pp):
    # Convert the strategy from decimals BACK to pp so alpha is monthly pp.
    # Factor values are already pp. Mkt-RF is already excess; do not subtract
    # RF from it (or from SMB/HML/Mom) a second time.
    y_pp = returns["Strategy"] * 100 - factors_pp["RF"]
    x_pp = sm.add_constant(factors_pp[FACTORS], has_constant="add")
    if len(y_pp) <= 5 or np.linalg.matrix_rank(x_pp.to_numpy()) != 5:
        raise ValueError("Insufficient data or collinear four-factor design")
    fit = sm.OLS(y_pp, x_pp, missing="raise").fit()
    # Verify that a consistent change from pp to decimal units only rescales
    # the intercept, leaving factor loadings, t-statistics and fit unchanged.
    decimal_fit = sm.OLS(y_pp / 100, sm.add_constant(factors_pp[FACTORS] / 100),
                         missing="raise").fit()
    np.testing.assert_allclose(fit.params[FACTORS], decimal_fit.params[FACTORS], atol=1e-12)
    np.testing.assert_allclose(fit.params["const"] / 100, decimal_fit.params["const"], atol=1e-12)
    np.testing.assert_allclose(fit.tvalues, decimal_fit.tvalues, atol=1e-10)
    table = pd.DataFrame({"Estimate": fit.params, "t-statistic": fit.tvalues,
                          "p-value": fit.pvalues})
    table.index = ["Alpha (monthly pp)", "MKT", "SMB", "HML", "MOM"]
    model_fit = pd.DataFrame({"R-squared": [fit.rsquared],
                              "Adjusted R-squared": [fit.rsquared_adj],
                              "N": [int(fit.nobs)]}, index=["Four factors"])
    return table, model_fit


def momentum_profit_analysis(returns, factors_pp):
    """Analyze the incremental return generated by the momentum allocation.

    The strategy is fully invested in equities, so its excess return contains
    ordinary equity-market performance. Strategy minus equal weight isolates
    the gain from changing country weights according to the momentum signal.
    No risk-free rate is subtracted from this zero-net-investment difference.
    """
    profit_decimal = returns["Strategy"] - returns["Equal weight"]
    monthly_sd = profit_decimal.std(ddof=1)
    mean_fit = sm.OLS(profit_decimal, np.ones((len(profit_decimal), 1))).fit()
    mean_t = float(mean_fit.tvalues.iloc[0])
    mean_p = float(mean_fit.pvalues.iloc[0])
    summary = pd.DataFrame({
        "Mean monthly (%)": [100 * profit_decimal.mean()],
        "Annual arithmetic (%)": [100 * 12 * profit_decimal.mean()],
        "Annual volatility (%)": [100 * np.sqrt(12) * monthly_sd],
        "Mean return t-statistic": [mean_t],
        "Mean return p-value": [mean_p],
        "N": [len(profit_decimal)],
    }, index=["Strategy minus equal weight"])

    # Regress the country-momentum gain itself on the supplied four factors.
    # This complements the regression of the funded strategy excess return.
    y_pp = profit_decimal * 100
    x_pp = sm.add_constant(factors_pp[FACTORS], has_constant="add")
    fit = sm.OLS(y_pp, x_pp, missing="raise").fit()
    decimal_fit = sm.OLS(y_pp / 100, sm.add_constant(factors_pp[FACTORS] / 100),
                         missing="raise").fit()
    np.testing.assert_allclose(fit.params[FACTORS], decimal_fit.params[FACTORS], atol=1e-12)
    np.testing.assert_allclose(fit.params["const"] / 100, decimal_fit.params["const"], atol=1e-12)
    np.testing.assert_allclose(fit.tvalues, decimal_fit.tvalues, atol=1e-10)
    coefficients = pd.DataFrame({"Estimate": fit.params,
                                 "t-statistic": fit.tvalues,
                                 "p-value": fit.pvalues})
    coefficients.index = ["Alpha (monthly pp)", "MKT", "SMB", "HML", "MOM"]
    model_fit = pd.DataFrame({"R-squared": [fit.rsquared],
                              "Adjusted R-squared": [fit.rsquared_adj],
                              "N": [int(fit.nobs)]},
                             index=["Momentum profit: strategy minus equal weight"])
    return summary, coefficients, model_fit


def factor_comparison(returns, factors_pp):
    # Compare excess/spread returns on identical months and in identical units.
    # Mom is a spread return, not a funded portfolio needing RF subtraction.
    series = pd.DataFrame({
        "Strategy excess": returns["Strategy"] * 100 - factors_pp["RF"],
        "Equal-weight excess": returns["Equal weight"] * 100 - factors_pp["RF"],
        "Momentum profit (strategy - equal weight)":
            (returns["Strategy"] - returns["Equal weight"]) * 100,
        "Market excess (Mkt-RF)": factors_pp["Mkt-RF"],
        "Size factor (SMB)": factors_pp["SMB"],
        "Value factor (HML)": factors_pp["HML"],
        "Momentum factor (Mom)": factors_pp["Mom"],
    })
    return pd.DataFrame({
        "Mean monthly excess/spread (pp)": series.mean(),
        "Annual arithmetic excess/spread (pp)": 12 * series.mean(),
        "Annual volatility of excess/spread (pp)": np.sqrt(12) * series.std(ddof=1),
        "N": len(series),
    })


def save_growth_plot(wealth):
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    # These are compounded realized portfolio returns, NOT ranking signals.
    # Plot only funded portfolios; long/short factor growth is not automatically
    # comparable because financing and capital conventions differ.
    dates = wealth.index.to_timestamp(how="end")
    for name in wealth.columns:
        ax.plot(dates, wealth[name], linewidth=1.8, label=name)
    ax.set_title("Part 2: International momentum versus equal weight\n"
                 f"Growth of 1 | {wealth.index[1]} to {wealth.index[-1]} | monthly rebalancing, zero costs")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth (initial investment = 1)")
    ax.grid(alpha=0.25)
    ax.legend()
    path = HERE / "Part2_1_growth_of_1.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def p_text(p):
    return f"{p:.2e}" if p < 0.0001 else f"{p:.4f}"


def explain(performance, coefficients, fit, profit_summary, profit_coefficients,
            profit_fit, comparison):
    print("\nDATA-DRIVEN CONCLUSION")
    strategy, benchmark = performance.loc["Strategy"], performance.loc["Equal weight"]
    print(f"Strategy: {strategy['Annual arithmetic (%)']:.2f}% annual arithmetic mean, "
          f"{strategy['Annual volatility (%)']:.2f}% annual volatility,")
    print(f"Sharpe {strategy['Annual Sharpe']:.3f}, maximum drawdown "
          f"{strategy['Max drawdown (%)']:.2f}%. One unit grows to {strategy['Final wealth']:.2f}.")
    print(f"Equal weight: {benchmark['Annual arithmetic (%)']:.2f}% annual arithmetic mean, "
          f"{benchmark['Annual volatility (%)']:.2f}% annual volatility,")
    print(f"Sharpe {benchmark['Annual Sharpe']:.3f}; final wealth {benchmark['Final wealth']:.2f}.")
    print(f"Strategy minus benchmark annual arithmetic return: "
          f"{strategy['Annual arithmetic (%)'] - benchmark['Annual arithmetic (%)']:+.2f} percentage points.")
    print(f"CAGR is separately {strategy['CAGR (%)']:.2f}% for the strategy; it is a geometric")
    print("growth rate, not the annualized arithmetic average.")
    profit = profit_summary.iloc[0]
    print(f"The momentum profit (strategy minus equal weight) averages "
          f"{profit['Annual arithmetic (%)']:.2f} pp/year, with "
          f"{profit['Annual volatility (%)']:.2f}% annual volatility and a mean-return "
          f"t-statistic of {profit['Mean return t-statistic']:.2f} "
          f"(p={p_text(profit['Mean return p-value'])}).")
    print("This benchmark-relative return isolates the effect of the momentum country weights.")
    print(f"\nFor the complete funded strategy, four factors explain "
          f"{fit.iloc[0]['R-squared']:.1%} of monthly excess-return variation")
    print(f"(adjusted R2={fit.iloc[0]['Adjusted R-squared']:.3f}).")
    meanings = {"MKT": "equity-market exposure", "SMB": "small minus large stocks",
                "HML": "value minus growth stocks", "MOM": "past winners minus past losers"}
    for name, meaning in meanings.items():
        row = coefficients.loc[name]
        status = "significant" if row["p-value"] < 0.05 else "not significant"
        if abs(abs(row["t-statistic"]) - 1.96) < 0.15:
            status += ", borderline"
        print(f"  {name} ({meaning}): loading {row['Estimate']:+.3f}, "
              f"t={row['t-statistic']:.2f}, p={p_text(row['p-value'])}: {status} at 5%.")
    print("Each loading is the associated percentage-point change in strategy excess")
    print("return for a one-percentage-point rise in that factor, holding others fixed.")
    print("Positive spread-factor loadings suggest the named exposure; negative loadings")
    print("suggest the opposite. Loading size and statistical significance are different.")
    alpha = coefficients.loc["Alpha (monthly pp)"]
    print(f"Alpha is {alpha['Estimate']:+.4f} pp/month (t={alpha['t-statistic']:.2f}, "
          f"p={p_text(alpha['p-value'])}).")
    if alpha["p-value"] < 0.05:
        print("Alpha differs statistically from zero under conventional OLS assumptions;")
        print("its sign matters, and the result alone does not establish investment skill.")
    else:
        print("Alpha is not statistically distinguishable from zero; that is not proof")
        print("that true alpha is exactly zero or that the model explains every return.")
    print("A positive raw return alone is not evidence of a distinct momentum profit.")
    mom = coefficients.loc["MOM"]
    print(f"The MOM loading of {mom['Estimate']:+.3f} measures momentum-factor association")
    print("after controlling for market, size and value; the strategy is long-only and")
    print("country based, while supplied Mom uses a different momentum construction.")
    print("Their returns therefore need not match.")
    print("\nDIRECT TEST OF THE INTERNATIONAL MOMENTUM PROFIT")
    print(f"The four factors explain {profit_fit.iloc[0]['R-squared']:.1%} of the monthly")
    print("strategy-minus-equal-weight return variation.")
    profit_alpha = profit_coefficients.loc["Alpha (monthly pp)"]
    profit_mom = profit_coefficients.loc["MOM"]
    print(f"Incremental alpha is {profit_alpha['Estimate']:+.4f} pp/month "
          f"(t={profit_alpha['t-statistic']:.2f}, p={p_text(profit_alpha['p-value'])}).")
    print(f"Its MOM loading is {profit_mom['Estimate']:+.3f} "
          f"(t={profit_mom['t-statistic']:.2f}, p={p_text(profit_mom['p-value'])}).")
    if profit_alpha["p-value"] < 0.05:
        print("The benchmark-relative alpha is statistically significant at 5%.")
    else:
        print("The benchmark-relative alpha is not statistically significant at 5%.")
    if profit_mom["p-value"] < 0.05:
        print("The supplied MOM factor is significantly related to the international")
        print("momentum profit. The insignificant residual alpha supports the conclusion")
        print("that the four-factor model explains the profit in the statistical-alpha sense,")
        print("although its R-squared need not be close to one.")
    else:
        print("The supplied MOM factor is not significantly related to the international")
        print("momentum profit at 5%; interpret any model-explanation claim cautiously.")
    print("\nMatched-sample annual excess/spread averages and risks:")
    for name, row in comparison.iterrows():
        print(f"  {name}: mean {row.iloc[1]:.2f} pp/year; volatility {row.iloc[2]:.2f} pp/year.")
    print("This comparison uses volatility of excess/spread returns. The required")
    print("portfolio Sharpe above also uses the volatility of monthly excess returns.")
    print("Mkt-RF is already excess; SMB, HML and Mom are factor spread returns. Their")
    print("growth of 1 is not automatically comparable to a funded long-only portfolio.")
    print("t-statistics divide estimates by their standard errors; |t| > 1.96 is an")
    print("approximate two-sided 5% guide. Conventional OLS inference can be affected")
    print("by heteroskedasticity and serial dependence. Results are historical, in-sample")
    print("associations, with no transaction costs; they do not establish causation or")
    print("future profitability. No Part 3 analysis is performed.")


def main():
    # Print the PDF's evidence and the assumptions needed where inputs are absent.
    import sys
    project_root = next(p for p in Path(__file__).resolve().parents if (p / "Files_Homework").is_dir())
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from homework_assumptions import print_assumptions
    print_assumptions("P2")
    print("PART 2: INTERNATIONAL MOMENTUM")
    world_pp = read_monthly_pp("HW_World.csv", COUNTRIES)
    factors_pp = read_monthly_pp("HW_Factors.csv", [*FACTORS, "RF"])
    # Input units are explicit, not inferred from magnitudes. Both files are
    # percentage points; convert world returns ONCE for compounding/portfolios.
    if (world_pp <= -100).any().any():
        raise ValueError("Country returns at/below -100%; inspect units and input")
    world_decimal = world_pp / 100.0
    np.testing.assert_allclose(world_decimal * 100, world_pp, atol=1e-12)
    print(f"World data: {world_pp.index.min()} to {world_pp.index.max()}, "
          f"{len(world_pp)} months, {len(world_pp.columns)} countries.")
    print("Dates/values validated: no duplicates, gaps, missing values or nonfinite values.")
    returns_all, weights, signals, audit = construct_strategy(world_decimal)
    # Form signals before aligning factors, so the lookback history is retained.
    # All reported performance, benchmarks and regressions then use ONE sample.
    common = returns_all.index.intersection(factors_pp.index).sort_values()
    if len(common) == 0:
        raise ValueError("No investment months overlap the factor data")
    if not common.equals(pd.period_range(common[0], common[-1], freq="M")):
        raise ValueError("Nonconsecutive investment months after factor alignment")
    returns = returns_all.loc[common]
    factors_pp = factors_pp.loc[common]
    pd.testing.assert_index_equal(returns.index, factors_pp.index)
    print(f"Trading/evaluation period: {common[0]} to {common[-1]}, {len(common)} months.")
    print(f"Investable months excluded for missing factors: {len(returns_all) - len(common)}.")
    print("The wording 'long/short' conflicts with the specified weights: this implements")
    print("the specified LONG-ONLY allocation (top 3: 4/15; middle 14: 1/70; bottom 3: 0).")
    print("Exact ties: alphabetical country order. Rational weight numerators sum to 210/210.")
    print("Monthly reranking/rebalancing; zero transaction costs.")
    print("\nFIRST THREE INVESTMENT MONTHS: TIMING AND RANKING")
    print(audit.loc[common].head(3).to_string())
    print("\nSAMPLE WEIGHTS (fractions; first three evaluated months)")
    print(weights.loc[common].head(3).T.to_string(float_format=lambda v: f"{v:.6f}"))
    print("Full signals and weights are saved for every evaluated month.")
    wealth = wealth_index(returns)
    performance = performance_table(returns, factors_pp["RF"] / 100, wealth)
    coefficients, model_fit = four_factor_model(returns, factors_pp)
    profit_summary, profit_coefficients, profit_model_fit = momentum_profit_analysis(
        returns, factors_pp
    )
    comparison = factor_comparison(returns, factors_pp)
    print("\nPERFORMANCE (returns/volatility/drawdown in percent; Sharpe unitless)")
    print("Annual arithmetic mean = 12*monthly mean; volatility = sqrt(12)*sample SD.")
    print("Sharpe = sqrt(12)*mean(R-RF)/sample SD(R-RF); drawdown includes initial wealth 1.")
    print(performance.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nFOUR-FACTOR OLS (alpha: monthly pp; loadings: unitless; conventional SEs)")
    print(coefficients.to_string(float_format=lambda v: f"{v:.4f}", formatters={"p-value": p_text}))
    print(model_fit.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nMOMENTUM PROFIT: STRATEGY MINUS EQUAL WEIGHT")
    print(profit_summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nFOUR-FACTOR OLS FOR MOMENTUM PROFIT")
    print("Dependent variable = strategy minus equal-weight return; alpha is monthly pp.")
    print(profit_coefficients.to_string(float_format=lambda v: f"{v:.4f}",
                                        formatters={"p-value": p_text}))
    print(profit_model_fit.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nFACTOR COMPARISON: IDENTICAL INVESTMENT MONTHS")
    print(comparison.to_string(float_format=lambda v: f"{v:.4f}"))
    plot_path = save_growth_plot(wealth)
    outputs = {"weights": weights.loc[common], "ranking_signals_decimal": signals.loc[common],
               "timing_audit": audit.loc[common], "returns_decimal": returns,
               "wealth": wealth, "performance": performance,
               "four_factor_coefficients": coefficients, "four_factor_fit": model_fit,
               "momentum_profit_summary": profit_summary,
               "momentum_profit_four_factor_coefficients": profit_coefficients,
               "momentum_profit_four_factor_fit": profit_model_fit,
               "factor_comparison": comparison}
    for suffix, table in outputs.items():
        table.to_csv(HERE / f"Part2_1_{suffix}.csv")
    explain(performance, coefficients, model_fit, profit_summary, profit_coefficients,
            profit_model_fit, comparison)
    print(f"\nGrowth-of-1 figure: {plot_path}")
    print("Saved CSV files:")
    for suffix in outputs:
        print(HERE / f"Part2_1_{suffix}.csv")
    print(f"Text report: {HERE / 'Part2_1_results.txt'}")


if __name__ == "__main__":
    report = StringIO()
    with redirect_stdout(report):
        main()
    text = report.getvalue()
    print(text, end="")
    (HERE / "Part2_1_results.txt").write_text(text, encoding="utf-8")
