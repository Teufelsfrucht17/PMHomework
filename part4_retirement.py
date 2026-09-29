"""
Teil 4 – Retirement Management (15 %)

Sieben ETF-Portfolios für ein 401k-Konto, feste Gewichte, kostenloses
monatliches Rebalancing. Kennzahlen: annualisierte Rendite, Volatilität,
maximaler Drawdown und Sharpe Ratio (risikoloser Zins = 1 % p.a.).

Wichtig: Die ETFs existieren unterschiedlich lange. Ein Portfolio kann erst ab
dem Monat starten, in dem ALLE seine ETFs Kurse haben. Deshalb werten wir aus:
  (a) jedes Portfolio über seine eigene, maximal verfügbare Historie
  (b) alle Portfolios über den GEMEINSAMEN Zeitraum (fairer Vergleich)
"""

import matplotlib.pyplot as plt
import pandas as pd

import data
import stats

RF_ANNUAL = 0.01

EQUITY = ["SPY", "IWM", "FEZ", "EEM"]          # die vier Aktien-ETFs
BONDS = ["TLT", "AGG", "JNK", "LQD"]           # die vier Anleihen-ETFs
FACTORS = ["MTUM", "SIZE", "VLUE", "USMV"]     # die vier Faktor-ETFs
ALTERNATIVES = ["USRT", "QAI", "IGF", "PSP"]   # REIT, Hedge Funds, Infrastruktur, Private Equity


def _split(total, tickers):
    """Verteilt einen Anteil `total` gleichmäßig auf die angegebenen ETFs."""
    return {t: total / len(tickers) for t in tickers}


# Die sieben Portfolios: Ticker -> Gewicht
PORTFOLIOS = {
    "(i) 60/40 baseline: SPY / TLT": {"SPY": 0.6, "TLT": 0.4},
    "(ii) Global equity (IWM/FEZ/EEM) / TLT": {"IWM": 0.3, "FEZ": 0.15, "EEM": 0.15, "TLT": 0.4},
    "(iii) SPY / bond mix (AGG/JNK/LQD)": {"SPY": 0.6, **_split(0.4, ["AGG", "JNK", "LQD"])},
    "(iv) 4 equity ETFs / 4 bond ETFs": {**_split(0.6, EQUITY), **_split(0.4, BONDS)},
    "(v) 4 factor ETFs / TLT": {**_split(0.6, FACTORS), "TLT": 0.4},
    "(vi) SPY / 4 alternatives": {"SPY": 0.6, **_split(0.4, ALTERNATIVES)},
    "(vii) Equal-weight all 16 ETFs": _split(1.0, EQUITY + BONDS + FACTORS + ALTERNATIVES),
}


def run():
    print("\n=== Teil 4: Retirement Management ===")
    returns = data.returns_from_prices(data.load_etfs())

    # Portfoliorendite je Monat; erst ab dem Monat, in dem alle Bestandteile Daten haben
    port = {}
    for name, weights in PORTFOLIOS.items():
        w = pd.Series(weights)
        assert abs(w.sum() - 1) < 1e-9, name          # Gewichte müssen 100 % ergeben
        r = returns[w.index].dropna()                 # nur Monate mit vollständigen Daten
        port[name] = r @ w
    port = pd.DataFrame(port)
    stats.save_table(port, "part4_monthly_returns.csv", print_it=False)

    # (a) jede Strategie über ihre eigene Historie
    print("\n--- (a) Jedes Portfolio über seine eigene Historie ---")
    own = stats.performance_table(port, RF_ANNUAL)
    stats.save_table(own[["Start", "End", "Months", "Mean return p.a.", "CAGR p.a.",
                          "Volatility p.a.", "Sharpe ratio", "Max drawdown"]],
                     "part4_performance_own_history.csv")

    # (b) gemeinsamer Zeitraum: ab dem spätesten Startmonat (USRT, Dez 2016)
    common = port.dropna()
    print(f"\n--- (b) Gemeinsamer Zeitraum {common.index[0]:%Y-%m} bis {common.index[-1]:%Y-%m} "
          f"({len(common)} Monate) ---")
    table = stats.performance_table(common, RF_ANNUAL)
    table["vs. baseline: CAGR"] = table["CAGR p.a."] - table.loc[table.index[0], "CAGR p.a."]
    table["vs. baseline: Sharpe"] = table["Sharpe ratio"] - table.loc[table.index[0], "Sharpe ratio"]
    stats.save_table(table[["Mean return p.a.", "CAGR p.a.", "Volatility p.a.", "Sharpe ratio",
                            "Max drawdown", "Min monthly return", "vs. baseline: CAGR",
                            "vs. baseline: Sharpe"]], "part4_performance_common_period.csv")

    # Korrelation der Portfolios mit dem Baseline-Portfolio (Erklärung der Unterschiede)
    corr = common.corr().iloc[:, 0].rename("Correlation with baseline")
    stats.save_table(corr.to_frame(), "part4_correlation_with_baseline.csv", print_it=False)

    _plots(common, table)


def _plots(common, table):
    """Wachstum von 1 $ + Drawdowns im gemeinsamen Zeitraum, und Risiko-Rendite-Diagramm."""
    growth = stats.cumulative_growth(common)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3]})
    for i, col in enumerate(growth):
        style = dict(color="black", linewidth=2.2) if i == 0 else dict(color=stats.COLORS[i])
        axes[0].plot(growth.index, growth[col], label=col, **style)
        dd = growth[col] / growth[col].cummax() - 1
        axes[1].plot(dd.index, dd * 100, **style)
    axes[0].set_ylabel("Growth of $1")
    axes[0].set_title(f"ETF retirement portfolios, monthly rebalanced, "
                      f"{common.index[0]:%b %Y} – {common.index[-1]:%b %Y}")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].set_ylabel("Drawdown (%)")
    stats.save_figure(fig, "part4_cumulative_performance.png")

    # Risiko-Rendite-Diagramm: x = Volatilität, y = Rendite; Steigung zur RF = Sharpe
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, name in enumerate(table.index):
        x, y = table.loc[name, "Volatility p.a."] * 100, table.loc[name, "CAGR p.a."] * 100
        ax.scatter(x, y, s=60, color="black" if i == 0 else stats.COLORS[i], zorder=3)
        ax.annotate(name.split(" ", 1)[0], (x, y), xytext=(5, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Annualised volatility (%)")
    ax.set_ylabel("Annualised return, CAGR (%)")
    ax.set_title("Risk vs. return (common period). Labels = portfolio number")
    ax.set_xlim(left=0)
    stats.save_figure(fig, "part4_risk_return.png")


if __name__ == "__main__":
    stats.setup_plot_style()
    run()
