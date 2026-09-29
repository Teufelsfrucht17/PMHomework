"""
Teil 3 – Portfolio Replication (25 %)

Zehn Portfolios aus den 30 Dow-Jones-Aktien, jeweils mit festen Gewichten und
kostenlosem monatlichem Rebalancing (so verlangt es die Aufgabe). Risikoloser
Zins = 0. Schätzfenster für alle "geschätzten" Portfolios (v–viii, x):
Renditen bis einschließlich Juni 2014 (erste Hälfte der Stichprobe).

Hinweis zur Aufgabenstellung: Dort steht "prices as of Jan 2024". Die Datei
enthält aber die Aktienanzahl per Jan 2004 und der Stichprobenbeginn ist
Jan 2004 – wir gehen deshalb von einem Tippfehler aus und verwenden Jan 2004.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import data
import stats

ESTIMATION_END = "2014-06"     # "halbe Stichprobe"
RISK_AVERSION = 5              # für die Nutzenfunktion in 3.3


def run():
    print("\n=== Teil 3: Portfolio Replication ===")
    d = data.load_prices()
    info, returns, prices_raw = d["info"], d["returns"], d["prices_raw"]

    # Schätzung von erwarteter Rendite und Kovarianz aus der ersten Hälfte
    est = returns.loc[:ESTIMATION_END]
    mu = est.mean()                # erwartete Monatsrendite je Aktie
    cov = est.cov()                # Varianz-Kovarianz-Matrix (monatlich)
    print(f"Schätzfenster: {est.index[0]:%Y-%m} bis {est.index[-1]:%Y-%m} ({len(est)} Monate)")
    print(f"Voller Zeitraum: {returns.index[0]:%Y-%m} bis {returns.index[-1]:%Y-%m} ({len(returns)} Monate)")

    weights = build_portfolios(info, prices_raw, mu, cov)
    _report_weights(weights, info)
    _performance(weights, returns, d["djia"])


# ---------------------------------------------------------------------------
# 3.1 Die zehn Portfolios
# ---------------------------------------------------------------------------
def build_portfolios(info, prices_raw, mu, cov) -> pd.DataFrame:
    """Gibt eine Tabelle zurück: Zeilen = 30 Aktien, Spalten = 10 Portfolios."""
    tickers = mu.index
    n = len(tickers)
    ones = np.ones(n)
    vol = np.sqrt(np.diag(cov))
    cov_inv = np.linalg.inv(cov.values)

    w = {}

    # (i) Marktkapitalisierung Jan 2004: Aktienanzahl 2004 x Kurs Jan 2004
    cap_2004 = info["Shares_2004"] * prices_raw.loc["2004-01"].iloc[0]
    w["(i) Market cap Jan 2004"] = cap_2004 / cap_2004.sum()

    # (ii) Aktienanzahl 2004, aber Kurse aus der Mitte der Stichprobe (Juni 2014)
    cap_2014 = info["Shares_2004"] * prices_raw.loc["2014-06"].iloc[0]
    w["(ii) Market cap (shares 2004, prices Jun 2014)"] = cap_2014 / cap_2014.sum()

    # (iii)/(iv) Gleichgewichtet: erste bzw. zweite Hälfte des Alphabets (nach Firmenname)
    alphabetical = info["Name"].sort_values().index          # Ticker, sortiert nach Name
    first_half, second_half = alphabetical[:15], alphabetical[15:]
    w["(iii) Equal-weight, names 3M-HD"] = pd.Series(1 / 15, index=first_half).reindex(tickers, fill_value=0)
    w["(iv) Equal-weight, names HON-WMT"] = pd.Series(1 / 15, index=second_half).reindex(tickers, fill_value=0)

    # (v) Tangentialportfolio (rf = 0): w ∝ Σ⁻¹ μ  – maximiert die Sharpe Ratio,
    #     Leerverkäufe erlaubt (Gewichte können negativ sein)
    raw = cov_inv @ mu.values
    w["(v) Tangency"] = pd.Series(raw / raw.sum(), index=tickers)

    # (vi) Minimum-Varianz-Portfolio: w ∝ Σ⁻¹ 1
    raw = cov_inv @ ones
    w["(vi) Minimum variance"] = pd.Series(raw / raw.sum(), index=tickers)

    # (vii) Naive Risk Parity: Gewicht ∝ 1 / Volatilität
    raw = 1 / vol
    w["(vii) Inverse volatility"] = pd.Series(raw / raw.sum(), index=tickers)

    # (viii) Most Diversified Portfolio: maximiert die Diversification Ratio
    #        DR = (w'σ) / sqrt(w'Σw), long-only, Summe der Gewichte = 1
    w["(viii) Most diversified"] = _optimize(
        lambda x: -(x @ vol) / np.sqrt(x @ cov.values @ x), n, tickers)

    # (ix) Gleiches Budget je Branche (5 Gruppen x 20 %), innerhalb gleichgewichtet
    group_size = info["Group"].value_counts()
    w["(ix) Equal group budget"] = info["Group"].map(lambda g: 0.2 / group_size[g])

    # (x) Equal Risk Contribution: jede Aktie trägt 1/30 der Portfoliovarianz
    def erc_objective(x):
        portfolio_var = x @ cov.values @ x
        risk_contrib = x * (cov.values @ x) / portfolio_var      # Anteil je Aktie
        return np.sum((risk_contrib - 1 / n) ** 2) * 1e4          # Abweichung vom Ziel 1/n
    w["(x) Equal risk contribution"] = _optimize(erc_objective, n, tickers)

    return pd.DataFrame(w)


def _optimize(objective, n, tickers) -> pd.Series:
    """Numerische Optimierung: long-only (0 ≤ w ≤ 1) und Summe der Gewichte = 1."""
    result = minimize(
        objective, x0=np.full(n, 1 / n), method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    assert result.success, result.message
    return pd.Series(result.x, index=tickers)


def _report_weights(weights, info):
    """Gewichte als Tabelle speichern und als Heatmap zeichnen."""
    print("\n--- 3.1 Gewichte bei Auflegung (in %) ---")
    table = weights.copy()
    table.insert(0, "Name", info["Name"])
    table.insert(1, "Group", info["Group"])
    stats.save_table(table, "part3_weights.csv", print_it=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20,
                           "display.float_format", "{:,.1f}".format):
        print((weights * 100).round(1))
        print("\nSumme der Gewichte:", (weights.sum()).round(3).to_dict())
        print("Kleinste Gewichte (Leerverkäufe?):", weights.min().round(3).to_dict())

    fig, ax = plt.subplots(figsize=(9, 9))
    vmax = np.abs(weights.values).max()
    im = ax.imshow(weights.values * 100, cmap="RdBu_r", vmin=-vmax * 100, vmax=vmax * 100, aspect="auto")
    ax.set_xticks(range(weights.shape[1]), [c.split(" ", 1)[0] for c in weights.columns])
    ax.set_yticks(range(weights.shape[0]), weights.index)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            ax.text(j, i, f"{weights.iloc[i, j] * 100:.0f}", ha="center", va="center", fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Weight (%)", shrink=0.6)
    ax.set_title("Portfolio weights at inception (%)")
    stats.save_figure(fig, "part3_weights_heatmap.png")


# ---------------------------------------------------------------------------
# 3.3 Performance-Kennzahlen (volle Stichprobe)
# ---------------------------------------------------------------------------
def _performance(weights, returns, djia):
    """Monatsrendite jedes Portfolios = Gewichte · Aktienrenditen (monatl. Rebalancing)."""
    print("\n--- 3.3 Performance Jan 2004 – Dez 2024 (rf = 0) ---")
    port = returns @ weights                        # Matrixprodukt: Monate x Portfolios
    port["DJIA index"] = djia
    table = stats.performance_table(port, rf_annual=0.0)
    # Mean-Variance-Nutzen: U = E[r] - 0.5 * A * Var(r)  (auf Jahresbasis)
    table["Utility (A=5)"] = table["Mean return p.a."] - 0.5 * RISK_AVERSION * table["Volatility p.a."] ** 2
    cols = ["Mean return p.a.", "Volatility p.a.", "Skewness", "Min monthly return",
            "Sharpe ratio", "Utility (A=5)", "Max drawdown", "CAGR p.a."]
    stats.save_table(table[cols], "part3_performance.csv")
    stats.save_table(port, "part3_monthly_returns.csv", print_it=False)
    _replication_check(port)

    # Wachstum von 1 $ – zur besseren Lesbarkeit in zwei Gruppen aufgeteilt
    growth = stats.cumulative_growth(port)
    groups = {
        "part3_performance_heuristic.png": ["(i) Market cap Jan 2004", "(ii) Market cap (shares 2004, prices Jun 2014)",
                                             "(iii) Equal-weight, names 3M-HD", "(iv) Equal-weight, names HON-WMT",
                                             "(ix) Equal group budget", "DJIA index"],
        "part3_performance_optimized.png": ["(v) Tangency", "(vi) Minimum variance", "(vii) Inverse volatility",
                                             "(viii) Most diversified", "(x) Equal risk contribution", "DJIA index"],
    }
    for filename, cols in groups.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, col in enumerate(cols):
            style = dict(color="black", linestyle="--") if col == "DJIA index" else dict(color=stats.COLORS[i])
            ax.plot(growth.index, growth[col], label=col, **style)
        ax.axvline(pd.Timestamp("2014-07-01"), color="grey", lw=0.8, linestyle=":")
        ax.text(pd.Timestamp("2014-08-01"), ax.get_ylim()[1] * 0.9, "out-of-sample →", color="grey", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel("Growth of $1 (log scale)")
        ax.set_title("Cumulative performance, monthly rebalanced, Jan 2004 – Dec 2024")
        ax.legend(loc="upper left", fontsize=8)
        stats.save_figure(fig, filename)


def _replication_check(port):
    """
    Wie gut bilden die Portfolios den DJIA nach? Für jedes Portfolio:
    Korrelation, Beta und Tracking Error (Volatilität der Renditedifferenz)
    gegenüber dem Index – über den gemeinsamen Zeitraum (DJIA ab Aug 2004,
    siehe Hinweis zur Datenkorrektur in data.py).
    """
    print("\n--- Replikations-Check gegen den DJIA ---")
    common = port.dropna()                          # nur Monate, in denen auch der DJIA vorliegt
    djia = common["DJIA index"]
    rows = {}
    for name, r in common.drop(columns="DJIA index").items():
        rows[name] = {
            "Correlation with DJIA": r.corr(djia),
            "Beta to DJIA": r.cov(djia) / djia.var(),
            "Tracking error p.a.": (r - djia).std() * np.sqrt(12),
            "Mean return p.a.": r.mean() * 12,
            "DJIA mean return p.a.": djia.mean() * 12,
        }
    table = pd.DataFrame(rows).T
    print(f"Zeitraum: {common.index[0]:%Y-%m} bis {common.index[-1]:%Y-%m} ({len(common)} Monate)")
    stats.save_table(table, "part3_replication_check.csv")


if __name__ == "__main__":
    stats.setup_plot_style()
    run()
