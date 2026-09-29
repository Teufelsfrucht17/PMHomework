"""Part 1, Question 1: monthly CAPM for the eight hedge fund indices.

Run from the project root:
.venv/Scripts/python.exe Part1/Part1_1_CAPM/Part1_1_CAPM.py
Requires pandas, numpy, statsmodels, and matplotlib.
Only HW_Hedge Fund.csv and HW_Factors.csv supply data for this analysis.
"""

from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import matplotlib

matplotlib.use("Agg")  # Save a figure without requiring an interactive window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "Files_Homework"


def parse_months(raw, date_format, source):
    """Reject invalid or duplicate dates instead of silently losing observations."""
    # Explicit formats distinguish Jan-05 from 200501. A monthly Period is
    # independent of which day of the month a source uses.
    parsed = pd.to_datetime(raw.str.strip(), format=date_format, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Invalid/missing dates in {source}: {raw[parsed.isna()].tolist()}")
    months = parsed.dt.to_period("M")
    if months.duplicated().any():
        raise ValueError(f"Duplicate months in {source}")
    return months


def load_sample(factor_names=("Mkt-RF",), heading="PART 1, QUESTION 1: CAPM"):
    """Share the same date matching and units with the Question 2 analysis."""
    # Keep the first row (short names); skip only the second row (descriptions).
    funds = pd.read_csv(DATA / "HW_Hedge Fund.csv", skiprows=[1], dtype=str)
    funds = funds.rename(columns={funds.columns[0]: "Month"})
    names = funds.columns[1:].tolist()
    if len(names) != 8 or "Aggregate Index" not in names:
        raise ValueError(f"Expected eight short index names; found {names}")
    funds["Month"] = parse_months(funds["Month"], "%b-%y", "hedge fund file")

    # Work in percentage points throughout: '0.47%' becomes 0.47, NOT 0.0047.
    # This matches the factor file's numerical percentage-point convention.
    for name in names:
        raw = funds[name].str.strip()
        if not raw.str.endswith("%", na=False).all():
            raise ValueError(f"Missing return or missing percent sign in {name}")
        funds[name] = pd.to_numeric(raw.str.removesuffix("%"), errors="coerce")

    factors = pd.read_csv(DATA / "HW_Factors.csv", dtype=str)
    factors = factors.rename(columns={factors.columns[0]: "Month"})
    # Q1 requests only Mkt-RF; Q2 also requests SMB, HML, and Mom.
    # RF is needed in both cases to calculate hedge fund excess returns.
    factor_columns = list(dict.fromkeys([*factor_names, "RF"]))
    factors = factors[["Month", *factor_columns]].copy()
    factors["Month"] = parse_months(factors["Month"], "%Y%m", "factor file")
    for name in factor_columns:
        factors[name] = pd.to_numeric(factors[name], errors="coerce")

    for label, frame, columns in [
        ("hedge fund", funds, names),
        ("factor", factors, factor_columns),
    ]:
        if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            raise ValueError(f"Missing, nonnumeric, or infinite returns in {label} file")

    # An inner join keeps ONLY dates present in both files. No forward filling.
    sample = funds.merge(factors, on="Month", how="inner", validate="one_to_one")
    sample = sample.sort_values("Month").set_index("Month")
    if len(sample) < 3 or sample["Mkt-RF"].nunique() < 2:
        raise ValueError("Insufficient overlapping data or no market variation")
    expected = pd.period_range(sample.index.min(), sample.index.max(), freq="M")
    if not sample.index.equals(expected):
        raise ValueError("The common sample contains gaps in its monthly dates")

    print(heading)
    print(f"Common sample: {sample.index.min()} to {sample.index.max()} ({len(sample)} months)")
    print(f"Hedge fund months excluded for lack of factors: {len(funds) - len(sample)}")
    print(f"Factor months excluded for lack of hedge fund data: {len(factors) - len(sample)}")
    print("All returns and monthly alpha are in percentage points; beta is unitless.")
    print("t-statistics use conventional OLS standard errors (not HAC/robust).\n")
    return sample, names


def estimate_capm(sample, names):
    # Mkt-RF is ALREADY an excess return. Do not subtract RF from it again.
    # Add an intercept explicitly: y = alpha + beta * market_excess + error.
    design = sm.add_constant(sample[["Mkt-RF"]], has_constant="add")
    excess_returns = sample[names].sub(sample["RF"], axis=0)
    rows = []
    for name in names:
        fit = sm.OLS(excess_returns[name], design, missing="raise").fit()
        rows.append({
            "Index": name,
            "Alpha (monthly pp)": fit.params["const"],
            "Beta": fit.params["Mkt-RF"],
            "t(alpha)": fit.tvalues["const"],
            "t(beta)": fit.tvalues["Mkt-RF"],
            "R-squared": fit.rsquared,
            "N": int(fit.nobs),
        })
    return pd.DataFrame(rows).set_index("Index"), excess_returns


def plot_capm(sample, results, excess_returns):
    # Shared axes make the slopes and the scale of returns comparable across
    # panels. Every dot is one matched month; the line is its OLS prediction.
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    market = sample["Mkt-RF"]
    grid = np.linspace(market.min(), market.max(), 200)
    for ax, name in zip(axes.flat, results.index):
        row = results.loc[name]
        ax.scatter(market, excess_returns[name], s=18, alpha=0.5, color="steelblue")
        ax.plot(grid, row["Alpha (monthly pp)"] + row["Beta"] * grid,
                color="darkorange", linewidth=2, label="Fitted CAPM")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_title(f"{name}\n"
                     f"alpha={row['Alpha (monthly pp)']:.3f} pp/month; "
                     f"beta={row['Beta']:.3f}\n"
                     f"R-squared={row['R-squared']:.3f}", fontsize=10)
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.supxlabel("Monthly market excess return, Mkt-RF (percentage points)")
    fig.supylabel("Monthly index excess return, index return - RF (percentage points)")
    fig.suptitle(f"CAPM: {sample.index.min()} to {sample.index.max()} | "
                 f"{len(sample)} matched monthly observations per index")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.95))
    path = HERE / "Part1_1_CAPM_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def explain_results(results):
    aggregate = results.loc["Aggregate Index"]
    alpha = aggregate["Alpha (monthly pp)"]
    alpha_t = aggregate["t(alpha)"]
    passes = abs(alpha_t) > 1.96
    print("\nINTERPRETATION")
    print("A t-statistic is an estimate divided by its standard error. It measures")
    print("how far the estimate is from zero in standard-error units. For a two-sided")
    print("5% test, |t| > 1.96 is an approximate guide, not a sharp proof of an effect.")
    print("Borderline values deserve caution. Conventional OLS inference assumes")
    print("constant error variance and uncorrelated errors; monthly fund returns may")
    print("violate these assumptions. Testing eight alphas also invites chance findings.\n")
    print(f"Aggregate Index: beta = {aggregate['Beta']:.3f} "
          f"(t = {aggregate['t(beta)']:.2f}). A 1 percentage-point increase in the")
    print(f"market excess return is associated with a {aggregate['Beta']:.3f} "
          "percentage-point increase in index excess return.")
    print(f"Monthly alpha = {alpha:.3f} percentage points (t = {alpha_t:.2f}); it "
          f"{'exceeds' if passes else 'does not exceed'} the approximate 5% threshold.")
    print("Alpha is the fitted excess return when market excess return is zero.")
    print(f"CAPM explains {aggregate['R-squared']:.1%} of aggregate monthly excess-return")
    print(f"variation, leaving {1 - aggregate['R-squared']:.1%} unexplained by this factor.\n")
    print("Qualitative mutual-fund benchmark (we have no mutual-fund dataset):")
    print("Diversified, equity-oriented mutual funds typically have substantial market")
    print("exposure, often beta nearer one and a higher market-model fit, with little")
    print("reliably positive net alpha. Compare that intuition with the aggregate's")
    print(f"beta of {aggregate['Beta']:.3f}, R-squared of {aggregate['R-squared']:.3f}, "
          f"and alpha t-statistic of {alpha_t:.2f}.")
    if aggregate["Beta"] < 1:
        print("The aggregate has lower market sensitivity than a beta-one equity portfolio.")
    if alpha > 0 and passes:
        print("Its positive alpha is statistically significant under conventional OLS,")
        print("unlike the usual intuition of little reliably positive mutual-fund net alpha.")
    print("This is not an empirical industry comparison or evidence of superior skill; ")
    print("the two CSVs also do not establish comparable fee conventions.\n")

    print("Market exposure, from highest to lowest beta:")
    for name, row in results.sort_values("Beta", ascending=False).iterrows():
        print(f"  {name}: beta {row['Beta']:.3f}")
    print("\nMarket-factor fit, from most to least variation explained:")
    for name, row in results.sort_values("R-squared", ascending=False).iterrows():
        print(f"  {name}: R-squared {row['R-squared']:.3f} "
              f"({row['R-squared']:.1%} explained)")
    strongest = results["R-squared"].idxmax()
    weakest = results["R-squared"].idxmin()
    print(f"\nIn the scatter plots, {strongest} has the clearest linear market relation,")
    print(f"whereas {weakest} has the weakest. Judge scatter relative to each index's")
    print("own variation: a narrow vertical range alone does not imply a high R-squared.")
    print("The fitted slopes show market exposure. Outlying months can affect the fit.")
    positive_alphas = results[(results["Alpha (monthly pp)"] > 0)
                             & (results["t(alpha)"] > 1.96)]
    print("Positive alphas above the approximate threshold: "
          + (", ".join(positive_alphas.index) or "none") + ".")
    for name, row in results.iterrows():
        if abs(abs(row["t(alpha)"]) - 1.96) < 0.1:
            print(f"{name}'s alpha t-statistic is {row['t(alpha)']:.4f}, near 1.96;")
            print("treat this as borderline, not definitive evidence of positive alpha.")
    print("Low R-squared means the single market factor explains little of monthly")
    print("variation; it does not itself prove skill or positive alpha. Other exposures")
    print("and strategy-specific risks may matter. No Question 2 model is estimated.")
    print("Index names follow the short CSV headers; 'FI Arbitrage' has the long")
    print("description 'Relative Value Index', so avoid interpreting it as pure FI arbitrage.")


def main():
    # Print the PDF's evidence and the assumptions needed where inputs are absent.
    import sys
    project_root = next(p for p in Path(__file__).resolve().parents if (p / "Files_Homework").is_dir())
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from homework_assumptions import print_assumptions
    print_assumptions("P1Q1")
    sample, names = load_sample()
    results, excess_returns = estimate_capm(sample, names)
    print(results.to_string(float_format=lambda value: f"{value:.4f}"))
    table_path = HERE / "Part1_1_CAPM_results.csv"
    results.to_csv(table_path)
    plot_path = plot_capm(sample, results, excess_returns)
    explain_results(results)
    print(f"\nResults table saved to: {table_path}")
    print(f"Scatter plots saved to: {plot_path}")


if __name__ == "__main__":
    report = StringIO()
    with redirect_stdout(report):
        main()
    text = report.getvalue()
    print(text, end="")
    (HERE / "Part1_1_results.txt").write_text(text, encoding="utf-8")
