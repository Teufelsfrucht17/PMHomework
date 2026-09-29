"""Part 5 Question 2: perfect-hindsight market timing, separate from Question 1.

Run with the project .venv interpreter:
.venv/Scripts/python.exe "Part5/Part5_2_Market Timing/Part5_2_Market Timing.py"
Requires numpy, pandas and matplotlib. Only HW_Factors.csv supplies data.
"""

from contextlib import redirect_stdout
from decimal import Decimal
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


def load_data():
    raw = pd.read_csv(SOURCE, dtype=str)
    dates = raw.iloc[:, 0].str.strip()
    if not dates.str.fullmatch(r"\d{6}", na=False).all():
        raise ValueError("Dates must be YYYYMM")
    parsed = pd.to_datetime(dates, format="%Y%m", errors="coerce")
    if parsed.isna().any():
        raise ValueError("Failed date parsing")
    raw.index = pd.PeriodIndex(parsed, freq="M", name="Month")
    if raw.index.has_duplicates:
        raise ValueError("Duplicate factor months")
    raw = raw.sort_index()
    expected = pd.period_range("1927-01", "2026-04", freq="M", name="Month")
    missing = expected.difference(raw.index)
    if len(missing):
        raise ValueError(f"Missing required months: {missing.tolist()}")
    raw = raw.loc[expected]
    pp = raw[["Mkt-RF", "RF"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(pp.to_numpy()).all():
        raise ValueError("Missing, nonnumeric or infinite returns")
    # Rank exact sums of the supplied decimal strings so a mathematical tie
    # is not broken by binary floating-point addition. Earlier date wins ties.
    total_pp_exact = [Decimal(m.strip()) + Decimal(r.strip())
                      for m, r in zip(raw["Mkt-RF"], raw["RF"])]
    order = np.array(sorted(range(len(raw)), key=lambda i: (-total_pp_exact[i], raw.index[i])))
    # Mkt-RF is excess, not total. Reconstruct total market return, then
    # convert percentage points to decimals before any compounding.
    data = pd.DataFrame({"Market total": [float(v) / 100 for v in total_pp_exact],
                         "RF": pp["RF"].to_numpy() / 100,
                         "Mkt-RF": pp["Mkt-RF"].to_numpy() / 100}, index=expected)
    if (data[["Market total", "RF"]] <= -1).any().any():
        raise ValueError("Gross returns must remain positive for the wealth comparison")
    np.testing.assert_allclose(data["Market total"] - data["RF"], data["Mkt-RF"], atol=1e-14)
    return data, order


def terminal_wealth(monthly_returns):
    # Input arrays are always in chronological order, including the first
    # January-1927 return. Both strategies begin just before that return.
    return INITIAL * np.cumprod(1 + monthly_returns)[-1]


def compare_all_k(data, order):
    market = data["Market total"].to_numpy()
    rf = data["RF"].to_numpy()
    n = len(data)
    passive = terminal_wealth(market)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    wealth = np.empty(n + 1)
    for k in range(n + 1):
        # Select months using ONE fixed total-return ranking. np.where retains
        # chronological order; selection never reorders monthly returns.
        selected_returns = np.where(rank <= k, market, rf)
        wealth[k] = terminal_wealth(selected_returns)
    if not np.isfinite(wealth).all():
        raise ValueError("Nonfinite terminal balances")
    np.testing.assert_allclose(wealth[0], terminal_wealth(rf), rtol=1e-13)
    # Identical return arrays and arithmetic mean k=N is exactly passive.
    if wealth[-1] != passive:
        raise AssertionError("k=N does not reproduce passive wealth exactly")
    # Search all k: the curve need not be monotonic as weak months are added.
    crossings = np.flatnonzero(wealth > passive)
    threshold = int(crossings[0]) if len(crossings) else None
    if threshold is not None:
        chosen = set(order[:threshold])
        direct = np.array([market[i] if i in chosen else rf[i] for i in range(n)])
        np.testing.assert_allclose(terminal_wealth(direct), wealth[threshold], rtol=1e-13)
        if not wealth[threshold] > passive or np.any(wealth[:threshold] > passive):
            raise AssertionError("The claimed threshold is not the first strict crossing")
    # Independent multiplicative switch identity checks all strategies. Each
    # equity month replaces a gross RF return by a gross total-market return.
    switches = (1 + market[order]) / (1 + rf[order])
    independent = wealth[0] * np.r_[1, np.cumprod(switches)]
    np.testing.assert_allclose(wealth, independent, rtol=1e-11)
    table = pd.DataFrame({"k": np.arange(n + 1), "Equity months (%)": 100*np.arange(n + 1)/n,
                          "Timing terminal wealth": wealth,
                          "Difference from passive": wealth-passive})
    ranked = data.iloc[order].copy()
    ranked.insert(0, "Rank", np.arange(1, n + 1))
    ranked = ranked.rename(columns={"Market total": "Market total (%)", "RF": "RF (%)", "Mkt-RF": "Mkt-RF (%)"})
    ranked[["Market total (%)", "RF (%)", "Mkt-RF (%)"]] *= 100
    return table, ranked, passive, threshold


def save_plot(table, passive, threshold):
    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
    ax.plot(table["k"], table["Timing terminal wealth"], color="steelblue", label="Top-k total-return months in equity; RF otherwise")
    ax.axhline(passive, color="black", linestyle="--", label="Passive equity")
    ax.set_yscale("log")  # Preserve the full wealth range without clipping.
    if threshold is not None:
        value = table.loc[threshold, "Timing terminal wealth"]
        ax.scatter([threshold], [value], color="darkorange", s=65, zorder=4,
                   label=f"First strict crossing: k={threshold}")
        ax.axvline(threshold, color="darkorange", linestyle=":", alpha=0.7)
    ax.set_xlabel("k: number of historically best total-market-return months held in equity")
    ax.set_ylabel("Terminal wealth (monetary units; logarithmic scale)")
    ax.set_title("Part 5 Q2: perfect-hindsight market timing\n"
                 "January 1927–April 2026 | initial wealth 100,000 | zero transaction costs")
    ax.grid(alpha=0.25, which="major")
    ax.legend(fontsize=9)
    path = HERE / "Part5_2_timing_vs_passive.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    # Print the PDF's evidence and the assumptions needed where inputs are absent.
    import sys
    project_root = next(p for p in Path(__file__).resolve().parents if (p / "Files_Homework").is_dir())
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from homework_assumptions import print_assumptions
    print_assumptions("P5Q2")
    print("PART 5, QUESTION 2: MARKET TIMING")
    data, order = load_data()
    table, ranked, passive, threshold = compare_all_k(data, order)
    n = len(data)
    print(f"Exact sample: {data.index[0]} through {data.index[-1]}, inclusive: {n} monthly observations.")
    print("No missing or duplicate required months; percentage-point inputs converted to decimals.")
    print("Initial wealth: 100,000 monetary units. Passive earns Mkt-RF + RF every month.")
    print("Timing earns that total return in selected months and historical RF otherwise.")
    print("Ranking: market TOTAL return descending; exact ties resolved by earlier date.")
    print("Same ranking for every k. Returns are compounded chronologically; zero transaction costs.")
    print(f"\nPassive equity terminal wealth: {passive:,.2f}")
    print(f"k=0 (always RF): {table.loc[0, 'Timing terminal wealth']:,.2f}")
    print(f"k={n} (always equity): {table.loc[n, 'Timing terminal wealth']:,.2f}; exactly equals passive.")
    print("Equality at k=N does not count as outperformance. Endpoint and switch-identity checks passed.")
    print("\nTEN BEST-RANKED MONTHS (returns in percentage points)")
    print(ranked.head(10).to_string(float_format=lambda v: f"{v:.4f}"))
    if threshold is None:
        print("\nNo k strictly outperforms passive equity under the specified ranking.")
    else:
        print(f"\nFIRST STRICT CROSSING: k={threshold}, {100*threshold/n:.4f}% of all months.")
        print(table.loc[max(0,threshold-1):threshold].to_string(index=False, float_format=lambda v: f"{v:,.6f}"))
        if threshold > 0:
            entering = ranked.iloc[threshold-1]
            print(f"Month entering at threshold: {ranked.index[threshold-1]}; "
                  f"total market {entering['Market total (%)']:.4f}%, RF {entering['RF (%)']:.4f}%, "
                  f"Mkt-RF {entering['Mkt-RF (%)']:.4f}%.")
        else:
            print("Already strictly above passive at k=0; k-1 and an entering equity month do not exist.")
    # Demonstrate that total and excess-return rankings need not be identical.
    excess_order = np.argsort(-data["Mkt-RF"].to_numpy(), kind="stable")
    differences = int(np.sum(order != excess_order))
    print(f"\nTotal-return versus excess-return ranking: {differences} rank positions differ in this sample.")
    print("The one-month incremental return from switching RF to equity is Mkt-RF.")
    print("The terminal-wealth multiplier is (1+market total)/(1+RF). Neither definition")
    print("replaces the assignment's TOTAL-return ranking in the headline calculation.")
    peak = table.loc[table["Timing terminal wealth"].idxmax()]
    print(f"The specified curve peaks at k={int(peak['k'])}, terminal wealth {peak['Timing terminal wealth']:,.2f}.")
    print("Adding months with negative excess returns can reduce terminal wealth, so the")
    print("first crossing was found by searching every k rather than assuming monotonic growth.")
    print("Capturing exceptionally strong months matters through compounding; skipping other")
    print("months also avoids both losses and some gains. RF continues earning a return during")
    print("unselected months, so this is not a strategy that earns zero while out of equities.")
    print("\nDEFENSE CONCLUSION")
    if threshold is not None:
        print(f"With perfect hindsight and the specified ranking, {threshold} selected equity months")
        print(f"({100*threshold/n:.2f}% of {n}) suffice to strictly exceed passive equity's terminal wealth.")
    else:
        print("No selected-month count beats passive under this ranking.")
    print("These best months are identified using the ENTIRE historical sample. This is")
    print("perfect hindsight, not a feasible real-time trading rule or evidence that an")
    print("investor could predict them. RF is a rolling short-term risk-free proxy, not a")
    print("long-duration bond fund. Results assume frictionless switching and no currency conversion.")
    plot_path = save_plot(table, passive, threshold)
    table.to_csv(HERE / "Part5_2_all_k_results.csv", index=False)
    ranked.to_csv(HERE / "Part5_2_ranked_months.csv")
    print(f"\nSaved figure: {plot_path}")
    print(f"All k results: {HERE / 'Part5_2_all_k_results.csv'}")
    print(f"Auditable month ranking: {HERE / 'Part5_2_ranked_months.csv'}")
    print(f"Full report: {HERE / 'Part5_2_results.txt'}")


if __name__ == "__main__":
    report = StringIO()
    with redirect_stdout(report):
        main()
    text = report.getvalue()
    print(text, end="")
    (HERE / "Part5_2_results.txt").write_text(text, encoding="utf-8")
