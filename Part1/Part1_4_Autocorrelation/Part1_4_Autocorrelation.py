"""Part 1, Question 4: AR(1) autocorrelation of reported hedge fund returns.

Run with the project's .venv interpreter in PyCharm, or from the project root:
.venv/Scripts/python.exe Part1/Part1_4_Autocorrelation/Part1_4_Autocorrelation.py

Requires numpy, pandas, statsmodels and matplotlib. The only data input is
HW_Hedge Fund.csv. Results and the figure are saved beside this script.
"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")  # Save a figure without needing an interactive window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
DATA_FILE = HERE.parents[1] / "Files_Homework" / "HW_Hedge Fund.csv"
sys.path.insert(0, str(HERE.parent / "Part1_1_CAPM"))
# Reuse Q1's date validation, but NOT its factor-merging loader: Q4 needs the
# full hedge fund history. Importing this function does not read factor data.
from Part1_1_CAPM import parse_months


def format_p(value):
    """Keep tiny p-values visible rather than rounding them to zero."""
    return f"{value:.2e}" if value < 0.0001 else f"{value:.4f}"


def load_returns():
    """Read all hedge fund months; fail explicitly on gaps or invalid values."""
    # Row 0 contains the eight short names. Row 1 contains descriptions and
    # must be skipped, rather than interpreted as an observation.
    returns = pd.read_csv(DATA_FILE, skiprows=[1], dtype=str)
    returns = returns.rename(columns={returns.columns[0]: "Month"})
    names = returns.columns[1:].tolist()
    if len(names) != 8 or "Aggregate Index" not in names:
        raise ValueError(f"Expected eight short index names; found {names}")
    # Parse Jan-05 explicitly as month/year; the shared helper rejects failed
    # parsing and duplicate months and returns calendar-month Period values.
    returns["Month"] = parse_months(returns["Month"], "%b-%y", "hedge fund file")
    returns = returns.set_index("Month").sort_index()
    for name in names:
        raw = returns[name].str.strip()
        if not raw.str.endswith("%", na=False).all():
            raise ValueError(f"Missing value or percent sign in {name}")
        # Keep percent units: '0.47%' becomes 0.47 percentage points, not
        # 0.0047. The intercept has this unit; rho is unitless.
        returns[name] = pd.to_numeric(raw.str.removesuffix("%"), errors="coerce")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("Returns contain missing, nonnumeric or infinite values")
    if len(returns) < 4:
        raise ValueError("At least four monthly returns are needed for AR(1) inference")
    expected = pd.period_range(returns.index.min(), returns.index.max(), freq="M")
    gaps = expected.difference(returns.index)
    if len(gaps):
        raise ValueError(f"Missing calendar months: {gaps.tolist()}. "
                         "Stop rather than construct lags across gaps.")
    return returns


def estimate_ar1(returns):
    # Because dates are sorted and confirmed consecutive, shifting one row is
    # exactly one calendar month. No RF subtraction: the question concerns
    # reported index returns themselves, not CAPM excess returns or residuals.
    lagged = returns.shift(1)
    current = returns.iloc[1:]
    previous = lagged.iloc[1:]
    # Explicit calendar check makes the lag's meaning verifiable.
    if not (returns.index[1:] - 1).equals(returns.index[:-1]):
        raise ValueError("A lag is not the immediately preceding calendar month")
    rows, fits = [], {}
    for name in returns.columns:
        x = sm.add_constant(previous[[name]].rename(columns={name: "Lagged return"}),
                            has_constant="add")
        # OLS includes its own intercept: R_t = a + rho * R_(t-1) + error.
        # fit() uses conventional OLS standard errors, as in the other questions.
        if np.linalg.matrix_rank(x.to_numpy()) != 2 or current[name].nunique() < 2:
            raise ValueError(f"Insufficient return variation for {name}")
        fit = sm.OLS(current[name], x, missing="raise").fit()
        coefficient = "Lagged return"
        # Each quantity below refers specifically to rho, not to the intercept.
        # t and the two-sided p-value test H0: rho = 0; CI uses Student's t.
        low, high = fit.conf_int(alpha=0.05).loc[coefficient]
        p_value = fit.pvalues[coefficient]
        rows.append({
            "Index": name,
            "rho": fit.params[coefficient],
            "t(rho)": fit.tvalues[coefficient],
            "p(rho)": p_value,
            "95% CI low": low,
            "95% CI high": high,
            "R-squared": fit.rsquared,
            "N pairs": int(fit.nobs),
            "Significant 5%": "Yes" if p_value < 0.05 else "No",
        })
        fits[name] = fit
    return pd.DataFrame(rows).set_index("Index"), fits, current, previous


def save_plot(results, fits, current, previous):
    # Each dot is a consecutive calendar-month pair. The fitted line includes
    # the AR(1) intercept and is drawn over the observed lagged-return range.
    fig, axes = plt.subplots(2, 4, figsize=(17, 9), layout="constrained")
    for ax, name in zip(axes.flat, results.index):
        x, y = previous[name], current[name]
        fit, row = fits[name], results.loc[name]
        grid = np.linspace(x.min(), x.max(), 150)
        ax.scatter(x, y, s=18, alpha=0.55, color="steelblue")
        ax.plot(grid, fit.params["const"] + row["rho"] * grid,
                color="darkorange", linewidth=2, label="Fitted AR(1)")
        ax.set_title(f"{name}\nrho={row['rho']:.3f}; p={format_p(row['p(rho)'])}", fontsize=10)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.grid(alpha=0.2)
        # Equal x/y units within each panel avoid visually distorting slopes.
        # Panel ranges vary to keep lower-volatility styles readable.
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        margin = 0.06 * (high - low)
        ax.set_xlim(low - margin, high + margin)
        ax.set_ylim(low - margin, high + margin)
        ax.set_aspect("equal", adjustable="box")
    axes.flat[0].legend(fontsize=8)
    fig.supxlabel("Previous calendar month's reported return (percentage points)")
    fig.supylabel("Current month's reported return (percentage points)")
    fig.suptitle(f"Question 4: AR(1) | current months {current.index.min()} to "
                 f"{current.index.max()} | {len(current)} pairs per index\n"
                 "Panel ranges vary; each panel has equal x/y units")
    path = HERE / "Part1_4_AR1_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def explain(results):
    print("\nQUESTION 4: INTERPRETATION")
    row = results.loc["Aggregate Index"]
    print(f"Aggregate Index: rho={row['rho']:.4f}, t={row['t(rho)']:.3f}, "
          f"p={row['p(rho)']:.4f}, 95% CI [{row['95% CI low']:.4f}, {row['95% CI high']:.4f}].")
    print(f"A 1 percentage-point higher return last month is associated with a")
    print(f"{row['rho']:+.4f} percentage-point change in this month's fitted return.")
    print(f"Its AR(1) explains {row['R-squared']:.1%} of current monthly return variation.")
    significant = results["p(rho)"] < 0.05
    groups = [
        ("Significant positive autocorrelation", significant & (results["rho"] > 0)),
        ("Significant negative autocorrelation", significant & (results["rho"] < 0)),
        ("No clear evidence of either at 5%", ~significant),
    ]
    for label, mask in groups:
        print(f"{label}: " + (", ".join(results.index[mask]) or "none") + ".")
    strongest = results["rho"].idxmax()
    print(f"The largest persistence estimate is {strongest}: rho={results.loc[strongest, 'rho']:.3f},")
    print(f"compared with {row['rho']:.3f} for the aggregate. Statistical significance")
    print("does not mean a strong fit: compare R-squared as well as the p-value.")
    if not significant.loc["Aggregate Index"]:
        print("The aggregate does not show statistically clear AR(1) dependence;")
        print("this does not rule out autocorrelation within particular styles.")
    else:
        print("The aggregate shows statistically detectable AR(1) dependence, but")
        print("the style-level results show whether that finding holds across strategies.")
    for name, item in results.iterrows():
        if abs(abs(item["t(rho)"]) - 1.96) < 0.15:
            print(f"{name} is near the 5% threshold (p={item['p(rho)']:.4f}); interpret cautiously.")
    print("Positive rho indicates persistence; negative rho indicates reversal. This")
    print("is own-return dependence, not a CAPM beta measuring market exposure.")
    print("A t-statistic is rho divided by its standard error. The p-value tests")
    print("H0: rho=0 against either sign; |t| > 1.96 is an approximate 5% guide.")
    print("These are unadjusted, individual tests; eight tests invite chance findings.")
    if (significant & (results["rho"] > 0)).any():
        print("The positive findings are consistent with return smoothing, stale pricing")
        print("or exposure to less liquid assets, but AR(1) alone cannot establish a cause.")
        print("If economic gains/losses are spread across reported months, measured")
        print("monthly volatility can look lower and contemporaneous diversification")
        print("can look better than the underlying economic risk warrants. Annualizing")
        print("volatility by sqrt(12) can also understate risk when serial dependence persists.")
    print("These are index-level reported returns, not evidence about every individual")
    print("hedge fund. Aggregation, changing constituents and omitted exposures can matter.")
    print("Conventional OLS inference assumes constant error variance and uncorrelated")
    print("innovations; remaining dependence or heteroskedasticity can affect inference.")
    print("AR(1) tests only one linear lag: an insignificant result does not prove")
    print("independence. Historical persistence does not guarantee future behavior or")
    print("a profitable trading strategy, especially after costs and liquidity constraints.")


def main():
    returns = load_returns()
    print("PART 1, QUESTION 4: AUTOCORRELATION")
    print(f"Full hedge fund sample: {returns.index.min()} to {returns.index.max()} "
          f"({len(returns)} monthly returns per index).")
    print("Units: percentage points (0.47% is stored as 0.47); no factor file or RF subtraction.")
    print("Checks passed: dates are unique and consecutive; returns are finite and complete.")
    results, fits, current, previous = estimate_ar1(returns)
    print(f"Current-return sample: {current.index.min()} to {current.index.max()}; "
          f"{len(current)} usable month pairs.")
    print(f"The first month, {returns.index.min()}, has no lag and is excluded as a dependent observation.")
    print("Inference: conventional OLS standard errors; two-sided tests of H0: rho=0.\n")
    print(results.to_string(float_format=lambda value: f"{value:.4f}",
                            formatters={"p(rho)": format_p}))
    results.to_csv(HERE / "Part1_4_AR1_results.csv")
    path = save_plot(results, fits, current, previous)
    explain(results)
    print(f"\nScatter figure saved to: {path}")
    print(f"Table and text report saved in: {HERE}")


if __name__ == "__main__":
    # Save the same readable report that appears in the run console.
    report = StringIO()
    with redirect_stdout(report):
        main()
    text = report.getvalue()
    print(text, end="")
    (HERE / "Part1_4_results.txt").write_text(text, encoding="utf-8")
