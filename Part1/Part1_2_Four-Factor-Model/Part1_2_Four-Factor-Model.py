"""Part 1, Question 2: four-factor OLS and a CAPM comparison.

Run this file in PyCharm using the project's .venv interpreter, or from the
project root run:
.venv/Scripts/python.exe Part1/Part1_2_Four-Factor-Model/Part1_2_Four-Factor-Model.py

Requires numpy, pandas, statsmodels, and matplotlib. All output files are saved
beside this script. Only the two homework CSVs supply financial data.
"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")  # Save plots even when no graphical window is available.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
# Import the existing Q1 functions without running its main() or its plots.
sys.path.insert(0, str(HERE.parent / "Part1_1_CAPM"))
from Part1_1_CAPM import estimate_capm, load_sample

FACTORS = ["Mkt-RF", "SMB", "HML", "Mom"]
COEFFICIENTS = ["const", *FACTORS]
LABELS = {"const": "Alpha (monthly pp)", "Mkt-RF": "MKT", "SMB": "SMB",
          "HML": "HML", "Mom": "MOM"}


def fit_models(sample, names):
    """Estimate both models on exactly the same dependent-variable observations."""
    capm_table, actual = estimate_capm(sample, names)
    # Q1 computes actual = index return - RF in percentage points. Mkt-RF is
    # already an excess return; SMB, HML and Mom are spread returns. None of
    # these four predictors should have RF subtracted again.
    x_capm = sm.add_constant(sample[["Mkt-RF"]], has_constant="add")
    x_four = sm.add_constant(sample[FACTORS], has_constant="add")
    # Four slopes plus an intercept require five independent design columns
    # and additional observations to estimate the error variance.
    if len(sample) <= x_four.shape[1]:
        raise ValueError("Too few months to estimate four-factor standard errors")
    if np.linalg.matrix_rank(x_four.to_numpy()) != x_four.shape[1]:
        raise ValueError("Four-factor predictors are perfectly collinear")

    estimates, t_stats, comparisons = [], [], []
    fitted = pd.DataFrame(index=sample.index)
    for name in names:
        y = actual[name]
        # OLS includes the explicit constant above. Default fit() uses ordinary
        # (nonrobust) standard errors, matching Q1. Each t-value belongs to its
        # own coefficient: t_j = coefficient_j / standard_error_j.
        capm = sm.OLS(y, x_capm, missing="raise").fit()
        four = sm.OLS(y, x_four, missing="raise").fit()
        if capm.nobs != four.nobs or int(four.nobs) != len(sample):
            raise ValueError(f"The models do not use identical observations for {name}")
        # Confirm that the refitted CAPM reproduces the existing Q1 function.
        np.testing.assert_allclose(
            [capm.params["const"], capm.params["Mkt-RF"], capm.rsquared],
            capm_table.loc[name, ["Alpha (monthly pp)", "Beta", "R-squared"]],
            rtol=1e-10, atol=1e-12,
        )
        estimates.append({LABELS[c]: four.params[c] for c in COEFFICIENTS})
        t_stats.append({LABELS[c]: four.tvalues[c] for c in COEFFICIENTS})
        fitted[name] = four.fittedvalues
        comparisons.append({
            "CAPM R2": capm.rsquared,
            "4F R2": four.rsquared,
            "CAPM adj R2": capm.rsquared_adj,
            "4F adj R2": four.rsquared_adj,
            "Delta adj R2": four.rsquared_adj - capm.rsquared_adj,
            "N": int(four.nobs),
        })
    index = pd.Index(names, name="Index")
    return (pd.DataFrame(estimates, index=index),
            pd.DataFrame(t_stats, index=index),
            pd.DataFrame(comparisons, index=index), actual, fitted, capm_table)


def save_plots(actual, fitted, comparison):
    # A model with four predictors has no single market-return regression line.
    # Instead, x is the fitted excess return using ALL FOUR observed factors;
    # y is actual excess return. Vertical distance from y=x is the residual.
    fig, axes = plt.subplots(2, 4, figsize=(17, 9), layout="constrained")
    for ax, name in zip(axes.flat, actual.columns):
        ax.scatter(fitted[name], actual[name], s=18, alpha=0.55, color="steelblue")
        low = min(actual[name].min(), fitted[name].min())
        high = max(actual[name].max(), fitted[name].max())
        margin = 0.06 * (high - low)
        limits = (low - margin, high + margin)
        ax.plot(limits, limits, "--", color="darkorange", label="Actual = fitted")
        # Equal x/y scales make the reference line genuinely 45 degrees.
        # Each panel has its own limits to show low-volatility styles clearly.
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        row = comparison.loc[name]
        ax.set_title(f"{name}\nR²={row['4F R2']:.3f}; "
                     f"adjusted R²={row['4F adj R2']:.3f}", fontsize=10)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)
    axes.flat[0].legend(fontsize=8)
    fig.supxlabel("Four-factor fitted monthly excess return (percentage points)")
    fig.supylabel("Actual monthly excess return: index return - RF (percentage points)")
    fig.suptitle(f"Four-factor in-sample fit | {actual.index.min()} to "
                 f"{actual.index.max()} | N={len(actual)}\n"
                 "Panel ranges vary; each panel uses equal x and y scales")
    scatter_path = HERE / "Part1_2_four_factor_scatter.png"
    fig.savefig(scatter_path, dpi=180)
    plt.close(fig)

    # Adjusted R² penalizes the three additional predictors. Paired horizontal
    # bars make improvements (or decreases) easy to compare across all styles.
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    positions = np.arange(len(comparison))
    for offset, column, label, color in [
        (-0.19, "CAPM adj R2", "CAPM", "steelblue"),
        (0.19, "4F adj R2", "Four factors", "darkorange"),
    ]:
        bars = ax.barh(positions + offset, comparison[column], height=0.36,
                       label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_yticks(positions, comparison.index)
    ax.invert_yaxis()
    ax.set_xlim(min(0, comparison[["CAPM adj R2", "4F adj R2"]].min().min()) - 0.02, 1)
    ax.set_xlabel("Adjusted R² (same monthly observations in both models)")
    ax.set_title("CAPM versus four-factor model: in-sample explanatory power")
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right")
    comparison_path = HERE / "Part1_2_adjusted_R2_comparison.png"
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)
    return scatter_path, comparison_path


def significance(t_value):
    """Use full-precision t-values; flag near-threshold results explicitly."""
    if abs(abs(t_value) - 1.96) < 0.15:
        side = "above" if abs(t_value) > 1.96 else "below"
        return f"borderline, just {side} the approximate 5% threshold"
    if abs(t_value) > 1.96:
        return "statistically distinguishable from zero using the approximate 5% guide"
    return "not statistically distinguishable from zero using the approximate 5% guide"


def explain(estimates, t_stats, comparison, capm_table):
    print("\nINTERPRETATION")
    print("A t-statistic is a coefficient divided by its standard error. The usual")
    print("two-sided 5% guide is |t| > 1.96. Borderline values are not decisive.")
    print("These are conventional OLS t-statistics; heteroskedasticity or serially")
    print("correlated errors can affect inference. Many simultaneous tests also invite")
    print("chance findings. A significant loading is association, not causation or skill.")
    print("\nFactors: MKT is the equity market return minus RF; SMB is small minus big")
    print("stocks; HML is high minus low book-to-market stocks (value minus growth);")
    print("MOM is past winners minus past losers. Positive loadings indicate exposure")
    print("in these directions; negative loadings suggest the reverse. These are")
    print("return associations, not direct measurements of portfolio holdings.")

    aggregate = estimates.loc["Aggregate Index"]
    aggregate_t = t_stats.loc["Aggregate Index"]
    directions = {"MKT": ("positive market", "negative market"),
                  "SMB": ("small-stock", "large-stock"),
                  "HML": ("value", "growth"),
                  "MOM": ("momentum", "contrarian")}
    print("\nAggregate Index, controlling for the other factors:")
    for factor, (positive, negative) in directions.items():
        coefficient, t_value = aggregate[factor], aggregate_t[factor]
        direction = positive if coefficient >= 0 else negative
        print(f"  {factor}: {coefficient:+.4f} (t={t_value:.3f}), suggesting {direction} exposure;")
        print(f"    {significance(t_value)}.")
    print("Each loading is the associated percentage-point change in index excess")
    print("return for a 1 percentage-point rise in that factor, holding others fixed.")
    alpha_label = "Alpha (monthly pp)"
    print(f"Four-factor alpha: {aggregate[alpha_label]:+.4f} percentage points/month ")
    print(f"(t={aggregate_t[alpha_label]:.3f}); {significance(aggregate_t[alpha_label])}.")
    print("Alpha is the fitted excess return when all four factor returns are zero.")
    old = capm_table.loc["Aggregate Index"]
    print(f"CAPM alpha was {old[alpha_label]:+.4f} pp/month (t={old['t(alpha)']:.3f});")
    print("the change shows how the intercept depends on which exposures are included.")

    print("\nExposure patterns across the seven other styles:")
    others = estimates.drop(index="Aggregate Index")
    for factor in directions:
        high, low = others[factor].idxmax(), others[factor].idxmin()
        print(f"  {factor}: highest {high} ({others.loc[high, factor]:+.3f}); "
              f"lowest {low} ({others.loc[low, factor]:+.3f}).")
        for sign, label in [(1, "Positive"), (-1, "Negative")]:
            selected = [name for name in others.index
                        if estimates.loc[name, factor] * sign > 0
                        and abs(t_stats.loc[name, factor]) > 1.96]
            print(f"    {label} loadings passing |t| > 1.96: "
                  + (", ".join(selected) if selected else "none") + ".")
    for name in others.index:
        for coefficient in estimates.columns:
            value = t_stats.loc[name, coefficient]
            if abs(abs(value) - 1.96) < 0.15:
                print(f"  Caution: {name}, {coefficient}, t={value:.3f} is borderline.")
    print("Keep the observed exposures even if a style's name suggests something else.")
    print(f"For example, Market Neutral still has market beta "
          f"{estimates.loc['Market Neutral', 'MKT']:.3f} "
          f"(t={t_stats.loc['Market Neutral', 'MKT']:.2f}).")
    print(f"Value's HML loading is {estimates.loc['Value', 'HML']:+.3f}, but its "
          f"t={t_stats.loc['Value', 'HML']:.2f} does not pass the 1.96 guide.")
    print("The short label 'FI Arbitrage' actually describes a broader Relative Value")
    print("Index in the source's long description row.")

    print("\nModel comparison, index by index:")
    for name, row in comparison.iterrows():
        gain = row["Delta adj R2"] * 100
        print(f"  {name}: R2 {row['CAPM R2']:.3f} -> {row['4F R2']:.3f}; "
              f"adjusted R2 {row['CAPM adj R2']:.3f} -> {row['4F adj R2']:.3f} "
              f"({gain:+.2f} percentage points).")
    biggest = comparison["Delta adj R2"].idxmax()
    smallest = comparison["Delta adj R2"].idxmin()
    weakest = comparison["4F adj R2"].idxmin()
    strongest = comparison["4F adj R2"].idxmax()
    print(f"Largest adjusted-fit gain: {biggest}; smallest: {smallest}.")
    gains = comparison["Delta adj R2"].sort_values(ascending=False)
    print("The more noticeable gains are in " + ", ".join(gains.index[:4]) + ".")
    print("The gains for " + ", ".join(gains.index[-3:]) + " are comparatively small.")
    print(f"The strongest four-factor fit is {strongest}; the weakest remains {weakest},")
    print(f"where ordinary R2 is only {comparison.loc[weakest, '4F R2']:.1%}.")
    print(f"Market Neutral also remains weakly explained "
          f"(R2={comparison.loc['Market Neutral', '4F R2']:.1%}); FI Arbitrage's "
          f"fit is partial (R2={comparison.loc['FI Arbitrage', '4F R2']:.1%}).")
    print("In the scatter plots, dots nearer the 45-degree line indicate smaller")
    print("residuals. Stronger fits track that line more closely relative to each")
    print("index's own return variation; weak fits leave wide vertical scatter.")
    print("Panel ranges differ, so compare R2 as well as visual distances.")
    print("Ordinary R2 cannot decrease when predictors are added to nested OLS models")
    print("on identical data. Adjusted R2 penalizes extra predictors:")
    print("  adjusted R2 = 1 - (1 - R2) * (n - 1) / (n - k - 1),")
    print("where k is 1 for CAPM and 4 for the four-factor model (excluding intercept).")
    print("Its improvement supports better in-sample explanation after that penalty;")
    print("it does not demonstrate better out-of-sample prediction. Low R2 does not")
    print("prove skill or positive alpha: other exposures and strategy-specific risks")
    print("may remain. No time-varying-beta or autocorrelation models are estimated.")


def main():
    # Print the PDF's evidence and the assumptions needed where inputs are absent.
    import sys
    project_root = next(p for p in Path(__file__).resolve().parents if (p / "Files_Homework").is_dir())
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from homework_assumptions import print_assumptions
    print_assumptions("P1Q2")
    # Reuse Q1's loader: retain short headers, skip descriptions, parse Jan-05
    # and 200501 as monthly dates, and inner-join. '0.47%' becomes 0.47 pp.
    # The loader now checks ALL requested factors for missing/nonfinite values;
    # it fails explicitly rather than silently changing either model's sample.
    sample, names = load_sample(FACTORS, "PART 1, QUESTION 2: FOUR-FACTOR MODEL")
    estimates, t_stats, comparison, actual, fitted, capm_table = fit_models(sample, names)
    print("FOUR-FACTOR COEFFICIENTS (alpha: monthly pp; factor loadings: unitless)")
    print(estimates.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nFOUR-FACTOR t-STATISTICS (each column tests its own coefficient = 0)")
    print(t_stats.rename(columns={"Alpha (monthly pp)": "Alpha"})
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nMODEL FIT: IDENTICAL OBSERVATIONS")
    print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))
    estimates.to_csv(HERE / "Part1_2_coefficients.csv")
    t_stats.rename(columns={"Alpha (monthly pp)": "Alpha"}).to_csv(HERE / "Part1_2_t_statistics.csv")
    comparison.to_csv(HERE / "Part1_2_model_comparison.csv")
    scatter_path, comparison_path = save_plots(actual, fitted, comparison)
    explain(estimates, t_stats, comparison, capm_table)
    print(f"\nActual versus fitted scatter plots: {scatter_path}")
    print(f"Adjusted R2 comparison chart: {comparison_path}")
    print(f"Tables and text report directory: {HERE}")


if __name__ == "__main__":
    # Print the report and save the same text for homework preparation.
    # If execution fails, the exception remains visible rather than reporting
    # invented results or suppressing the error.
    report = StringIO()
    with redirect_stdout(report):
        main()
    text = report.getvalue()
    print(text, end="")
    (HERE / "Part1_2_results.txt").write_text(text, encoding="utf-8")
