"""Part 1, Question 3: CAPM betas in up and down markets.

Run this file in PyCharm with the project's .venv interpreter, or run from
the project root:
.venv/Scripts/python.exe "Part1/Part1_3_Time-Varying Beta/Part1_3_Time-Varying Beta.py"

Requires numpy, pandas, statsmodels, and matplotlib. Financial inputs are
only HW_Hedge Fund.csv and HW_Factors.csv. Outputs go beside this script.
"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a GUI window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
# Reuse the same loader and full-sample CAPM implementation as Questions 1/2.
# Importing this module does not run Question 1's main() or produce its plots.
sys.path.insert(0, str(HERE.parent / "Part1_1_CAPM"))
from Part1_1_CAPM import estimate_capm, load_sample


def define_regimes(sample):
    """Define one common set of regime dates for all eight indices."""
    # The factors are numerical percentage points. Mkt-RF is NOT a raw return:
    # add RF back only to classify months; retain Mkt-RF as the CAPM regressor.
    raw_market = sample["Mkt-RF"] + sample["RF"]
    up = raw_market > 0
    down = raw_market < 0
    zero = raw_market == 0
    if not (up | down | zero).all():
        raise ValueError("Cannot classify one or more market returns")
    # Exactly zero months belong to neither regime. Exclude them from BOTH
    # separate regressions and the pooled interaction regression, but retain
    # them in the full-sample CAPM reference, which matches Question 1.
    print("Regimes use RAW market return = Mkt-RF + RF:")
    print(f"  Up (> 0): {int(up.sum())}; down (< 0): {int(down.sum())}; "
          f"exactly zero: {int(zero.sum())}.")
    print("Zero-return months are excluded from regime and interaction regressions,")
    print("but retained in the full-sample CAPM reference.")
    if zero.any():
        print("Zero-return months: " + ", ".join(sample.index[zero].astype(str)))
    print("All eight indices use identical regime dates. CAPM x = Mkt-RF in both regimes.\n")
    for label, mask in [("up", up), ("down", down)]:
        # Two parameters (intercept and slope) require at least three months
        # to leave positive residual degrees of freedom for standard errors.
        if mask.sum() < 3 or sample.loc[mask, "Mkt-RF"].nunique() < 2:
            raise ValueError(f"Insufficient observations or market variation in {label} regime")
    return raw_market, up, down


def estimate_regime_models(sample, names, up, down):
    # The reused Q1 function subtracts RF once from each index's return.
    # All y values and x values remain in percentage points; betas are unitless.
    full_capm, excess = estimate_capm(sample, names)
    x = sample["Mkt-RF"]
    x_up = sm.add_constant(x.loc[up], has_constant="add")
    x_down = sm.add_constant(x.loc[down], has_constant="add")

    # A separate intercept shift is essential: c allows down-market alpha to
    # differ as well as beta. Omitting Down would force a common intercept.
    nonzero = up | down
    pooled_x = pd.DataFrame({"Mkt-RF": x.loc[nonzero],
                             "Down": down.loc[nonzero].astype(int)})
    pooled_x["Down_x_Mkt-RF"] = pooled_x["Down"] * pooled_x["Mkt-RF"]
    pooled_x = sm.add_constant(pooled_x, has_constant="add")
    if np.linalg.matrix_rank(pooled_x.to_numpy()) != 4:
        raise ValueError("The pooled interaction design is not full rank")

    beta_rows, diagnostic_rows, models = [], [], {}
    for name in names:
        y = excess[name]
        # Each OLS includes an intercept. Conventional OLS standard errors
        # match Q1/Q2. The pooled test assumes a common residual variance;
        # unequal variance or serial dependence can affect its inference.
        up_fit = sm.OLS(y.loc[up], x_up, missing="raise").fit()
        down_fit = sm.OLS(y.loc[down], x_down, missing="raise").fit()
        pooled_fit = sm.OLS(y.loc[nonzero], pooled_x, missing="raise").fit()
        beta_up = up_fit.params["Mkt-RF"]
        beta_down = down_fit.params["Mkt-RF"]
        difference = beta_down - beta_up
        interaction = "Down_x_Mkt-RF"

        # Algebraic check: b = beta_up, b+d = beta_down, a = alpha_up,
        # a+c = alpha_down. This also checks the regimes and samples agree.
        np.testing.assert_allclose(
            [pooled_fit.params["Mkt-RF"],
             pooled_fit.params["Mkt-RF"] + pooled_fit.params[interaction],
             pooled_fit.params["const"],
             pooled_fit.params["const"] + pooled_fit.params["Down"],
             pooled_fit.params[interaction]],
            [beta_up, beta_down, up_fit.params["const"],
             down_fit.params["const"], difference],
            rtol=1e-10, atol=1e-12,
        )
        if int(pooled_fit.nobs) != int(up_fit.nobs + down_fit.nobs):
            raise ValueError(f"Regime and pooled sample sizes disagree for {name}")
        beta_rows.append({
            "Index": name,
            "Full beta": full_capm.loc[name, "Beta"],
            "Up beta": beta_up,
            "Down beta": beta_down,
            "Down - Up": difference,
            "N full": int(full_capm.loc[name, "N"]),
            "N up": int(up_fit.nobs),
            "N down": int(down_fit.nobs),
        })
        diagnostic_rows.append({
            "Index": name,
            "t(beta up)": up_fit.tvalues["Mkt-RF"],
            "t(beta down)": down_fit.tvalues["Mkt-RF"],
            "R2 up": up_fit.rsquared,
            "R2 down": down_fit.rsquared,
            "Pooled d": pooled_fit.params[interaction],
            "t(d)": pooled_fit.tvalues[interaction],
            "p(d)": pooled_fit.pvalues[interaction],
        })
        models[name] = {"up": up_fit, "down": down_fit}
    print("Verified: pooled betas and intercepts match the separate regressions for all indices.\n")
    return (pd.DataFrame(beta_rows).set_index("Index"),
            pd.DataFrame(diagnostic_rows).set_index("Index"), excess, models)


def save_plots(sample, betas, excess, models, up, down):
    # Paired bars use a common beta scale; values above each bar show the size
    # of exposure, while the second results table tests the difference itself.
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    positions = np.arange(len(betas))
    for offset, column, label, color in [
        (-0.19, "Up beta", f"Up market (N={int(up.sum())})", "steelblue"),
        (0.19, "Down beta", f"Down market (N={int(down.sum())})", "darkorange"),
    ]:
        bars = ax.bar(positions + offset, betas[column], width=0.36,
                      label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(positions, betas.index, rotation=20, ha="right")
    ax.set_ylabel("CAPM beta (unitless)")
    ax.set_title(f"Up- versus down-market beta | {sample.index.min()} to {sample.index.max()}")
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.margins(y=0.2)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    beta_path = HERE / "Part1_3_beta_comparison.png"
    fig.savefig(beta_path, dpi=180)
    plt.close(fig)

    # Plot aggregate excess returns against MARKET EXCESS returns, while
    # colors use RAW market returns. Because RF changes over time, the regime
    # boundary is not a fixed vertical line at x=0 on this plot.
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    name = "Aggregate Index"
    for label, mask, color in [("up", up, "steelblue"), ("down", down, "darkorange")]:
        x = sample.loc[mask, "Mkt-RF"]
        fit = models[name][label]
        ax.scatter(x, excess.loc[mask, name], color=color, alpha=0.65, s=28,
                   label=f"{label.title()} market months (N={int(mask.sum())})")
        # Draw each fitted line only over that regime's observed x range.
        grid = np.linspace(x.min(), x.max(), 150)
        ax.plot(grid, fit.params["const"] + fit.params["Mkt-RF"] * grid,
                color=color, linewidth=2.3,
                label=f"{label.title()} fit: alpha={fit.params['const']:.3f} pp, "
                      f"beta={fit.params['Mkt-RF']:.3f}")
    ax.set_xlabel("Monthly market excess return, Mkt-RF (percentage points)")
    ax.set_ylabel("Monthly aggregate excess return, index return - RF (percentage points)")
    ax.set_title(f"Aggregate Index: separate CAPM fits | {sample.index.min()} to {sample.index.max()}\n"
                 "Regime determined by raw market return = Mkt-RF + RF")
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    scatter_path = HERE / "Part1_3_aggregate_scatter.png"
    fig.savefig(scatter_path, dpi=180)
    plt.close(fig)
    return beta_path, scatter_path


def explain(betas, diagnostics):
    print("\nINTERPRETATION")
    aggregate = betas.loc["Aggregate Index"]
    test = diagnostics.loc["Aggregate Index"]
    direction = "higher" if aggregate["Down - Up"] > 0 else "lower"
    significant = test["p(d)"] < 0.05
    print(f"Aggregate beta is {aggregate['Up beta']:.3f} in up markets and "
          f"{aggregate['Down beta']:.3f} in down markets:")
    print(f"down-market exposure is {direction} by {abs(aggregate['Down - Up']):.3f}.")
    print(f"The interaction test gives t={test['t(d)']:.3f}, p={test['p(d)']:.4f}; "
          f"the difference {'is' if significant else 'is not'} statistically")
    print("distinguishable from zero at the conventional two-sided 5% level.")
    print(f"A 1 percentage-point fall in market excess return is associated with a")
    print(f"{aggregate['Down beta']:.3f} percentage-point fall in aggregate excess return")
    print(f"within down months, compared with {aggregate['Up beta']:.3f} using the up-month slope.")
    print("This compares sensitivities, not average returns: each regime has its own intercept.")
    if aggregate["Down - Up"] > 0:
        print("The point estimates suggest slightly less downside protection than the")
        print("up-market beta alone suggests; the economic size of this gap is small.")
    else:
        print("The lower downside sensitivity suggests more protection than its up-market")
        print("beta alone would suggest, although it does not ensure positive returns in downturns.")
    if not significant:
        print("However, the sample does not provide clear statistical evidence of that asymmetry.")

    print("\nOther styles (difference = down beta minus up beta):")
    for name in betas.index.drop("Aggregate Index"):
        row, test = betas.loc[name], diagnostics.loc[name]
        status = "passes the 5% test" if test["p(d)"] < 0.05 else "does not pass the 5% test"
        if abs(abs(test["t(d)"]) - 1.96) < 0.15:
            status += "; near the threshold, interpret cautiously"
        print(f"  {name}: {row['Up beta']:.3f} up, {row['Down beta']:.3f} down; "
              f"difference {row['Down - Up']:+.3f}, p={test['p(d)']:.4f} ({status}).")
    others = betas.drop(index="Aggregate Index")
    biggest = others["Down - Up"].idxmax()
    smallest = others["Down - Up"].idxmin()
    print(f"Largest numerical beta difference: {biggest}; smallest: {smallest}.")
    higher_significant = [name for name in others.index
                          if others.loc[name, "Down - Up"] > 0
                          and diagnostics.loc[name, "p(d)"] < 0.05]
    if higher_significant:
        print("Higher downside exposure with evidence at the 5% level: "
              + ", ".join(higher_significant) + ".")
        print("This weakens their appeal as downside protection relative to their up-market exposure.")
    macro_test = diagnostics.loc["Macro"]
    if betas.loc["Macro", "Down beta"] < 0 and abs(macro_test["t(beta down)"]) < 1.96:
        print("Macro's negative down-market beta suggests potential protection, but that")
        print("beta itself is not distinguishable from zero using the 1.96 guide.")
    print("The full-sample beta need not lie between the regime betas: its single")
    print("line also reflects differences in mean returns between the two regimes.")
    print("Economic size is the beta gap; statistical significance measures evidence")
    print("relative to estimation uncertainty. A small p-value need not imply a large gap.")
    print("A t-statistic is the estimate divided by its standard error. |t| > 1.96")
    print("is an approximate two-sided 5% guide; p(d) uses the finite-sample t distribution.")
    print("Two individually significant regime betas do NOT establish a significant")
    print("difference: the interaction coefficient d directly tests that difference.")
    print("Conventional OLS inference assumes constant error variance and uncorrelated")
    print("errors; violations can affect these tests. Eight tests also invite chance findings.")
    print("Generally, higher down-market beta means more market exposure when protection")
    print("is needed; lower down-market beta may offer more protection. Betas alone do")
    print("not measure total risk, tail losses, or overall investment attractiveness.")
    print("These are historical conditional associations, not causal effects or guarantees")
    print("of future downturn behavior. This is an up/down comparison, not a rolling beta.")


def main():
    # The shared loader retains the eight short headers, skips the description
    # row, parses Jan-05 and 200501 as months, and uses only their inner join.
    # It converts '0.47%' to 0.47 percentage points to match the factors and
    # rejects failed parsing, duplicates, gaps, missing values and infinities.
    sample, names = load_sample(heading="PART 1, QUESTION 3: TIME-VARYING BETA")
    raw_market, up, down = define_regimes(sample)
    betas, diagnostics, excess, models = estimate_regime_models(sample, names, up, down)
    print("TABLE 1: CAPM BETAS AND OBSERVATION COUNTS")
    print(betas.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nTABLE 2: REGIME FIT AND POOLED TEST OF THE BETA DIFFERENCE")
    print(diagnostics.to_string(float_format=lambda value: f"{value:.4f}"))
    print("Pooled d is the coefficient on Down * (Mkt-RF); the pooled model also")
    print("includes an intercept, Mkt-RF, and Down. Its null hypothesis is d = 0.")
    betas.to_csv(HERE / "Part1_3_betas.csv")
    diagnostics.to_csv(HERE / "Part1_3_diagnostics.csv")
    # Save the exact monthly classification so the split can be inspected.
    audit = sample[["Mkt-RF", "RF"]].copy()
    audit["Raw market return (pp)"] = raw_market
    audit["Regime"] = np.where(up, "Up", np.where(down, "Down", "Zero (excluded)"))
    audit.to_csv(HERE / "Part1_3_regime_months.csv")
    beta_path, scatter_path = save_plots(sample, betas, excess, models, up, down)
    explain(betas, diagnostics)
    print(f"\nBeta comparison figure: {beta_path}")
    print(f"Aggregate scatter figure: {scatter_path}")
    print(f"Tables, monthly regime audit, and text report saved in: {HERE}")


if __name__ == "__main__":
    # Preserve a readable copy of the same report printed in the console.
    report = StringIO()
    with redirect_stdout(report):
        main()
    text = report.getvalue()
    print(text, end="")
    (HERE / "Part1_3_results.txt").write_text(text, encoding="utf-8")
