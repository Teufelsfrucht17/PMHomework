"""
Teil 5 – Windfall Investment (15 %)

5.1 Policy-Portfolios: 100.000 € werden einmalig investiert und 30 Jahre OHNE
    Rebalancing gehalten (Buy-and-Hold, die Gewichte "driften"). Elf Mischungen
    Aktien/Anleihen (0/100, 10/90, ..., 100/0): Aktien = Marktfaktor (Mkt-RF + RF),
    Anleihen = risikoloser Zins RF. Der Startzeitpunkt ist unsicher – wir rechnen
    jeden möglichen Startmonat (Ende Jan 1927 bis Ende Apr 1996 = 832 Starts)
    durch und mitteln.

5.2 Market Timing: 100 % Anleihen, außer in den k besten Aktienmonaten der
    gesamten Historie. Ab welchem k schlägt man Buy-and-Hold in Aktien?
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import stats

INITIAL = 100_000
YEARS = 30
HORIZON = YEARS * 12                         # 360 Monate
EQUITY_SHARES = np.arange(0, 1.01, 0.1)      # 0 %, 10 %, ..., 100 % Aktien
RF_ANNUAL = 0.01                             # für die Sharpe Ratio (Aufgabenvorgabe)


def run():
    print("\n=== Teil 5: Windfall Investment ===")
    factors = data.load_factors()
    _policy_portfolios(factors)
    _market_timing(factors)


# ---------------------------------------------------------------------------
# 5.1 Policy-Portfolios (Buy-and-Hold über 30 Jahre)
# ---------------------------------------------------------------------------
def _policy_portfolios(factors):
    mkt = factors["Mkt"].values         # Aktienrendite (inkl. RF)
    rf = factors["RF"].values           # Anleihenrendite
    dates = factors.index
    n_starts = len(mkt) - HORIZON       # Starts, für die 30 Folgejahre vorliegen
    start_dates = dates[:n_starts]
    print(f"Startmonate: {start_dates[0]:%Y-%m} bis {start_dates[-1]:%Y-%m} ({n_starts} Starts)")

    # Für jeden Start s und jede Aktienquote w:
    #   Vermögen_t = INITIAL * [ w * Π(1+mkt) + (1-w) * Π(1+rf) ]   (Produkt über s+1 ... t)
    # Ohne Rebalancing wächst jeder Baustein für sich – daher einfach die Summe.
    # Wir sammeln je Start: Endvermögen, annualisierte Rendite, Volatilität, Sharpe.
    balance = np.zeros((n_starts, len(EQUITY_SHARES)))
    ann_ret = np.zeros_like(balance)
    ann_vol = np.zeros_like(balance)
    for s in range(n_starts):
        window = slice(s + 1, s + 1 + HORIZON)                     # die 360 Monate nach dem Start
        growth_mkt = np.cumprod(1 + mkt[window])                   # Wert von 1 € in Aktien
        growth_rf = np.cumprod(1 + rf[window])                     # Wert von 1 € in Anleihen
        # Matrix: 360 Monate x 11 Portfolios
        value = np.outer(growth_mkt, EQUITY_SHARES) + np.outer(growth_rf, 1 - EQUITY_SHARES)
        monthly = value[1:] / value[:-1] - 1                       # Monatsrenditen des Portfolios
        monthly = np.vstack([value[0] - 1, monthly])               # erste Monatsrendite ergänzen

        balance[s] = INITIAL * value[-1]
        ann_ret[s] = value[-1] ** (1 / YEARS) - 1                  # geometrisch (CAGR)
        ann_vol[s] = monthly.std(axis=0, ddof=1) * np.sqrt(12)
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol

    labels = [f"{int(w * 100)}/{int(round((1 - w) * 100))}" for w in EQUITY_SHARES]   # "Aktien/Anleihen"
    to_df = lambda arr: pd.DataFrame(arr, index=start_dates, columns=labels)
    balance, ann_ret, ann_vol, sharpe = map(to_df, (balance, ann_ret, ann_vol, sharpe))

    # Durchschnitt über alle 832 Startmonate
    summary = pd.DataFrame({
        "Avg. balance after 30y (EUR)": balance.mean(),
        "Median balance (EUR)": balance.median(),
        "Worst balance (EUR)": balance.min(),
        "Best balance (EUR)": balance.max(),
        "Share of starts below 100k": (balance < INITIAL).mean(),
        "Avg. annualised return": ann_ret.mean(),
        "Avg. annualised volatility": ann_vol.mean(),
        "Avg. Sharpe ratio (rf=1%)": sharpe.mean(),
    })
    summary.index.name = "Equity/Bonds"
    print("\n--- 5.1 Durchschnitt über alle Startmonate ---")
    stats.save_table(summary, "part5_policy_portfolios_summary.csv")
    stats.save_table(balance, "part5_policy_balance_by_start.csv", print_it=False)

    # Grafik 1: Endvermögen je Startmonat
    cmap = plt.get_cmap("viridis")
    colors = [cmap(x) for x in np.linspace(0.05, 0.95, len(labels))]
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, c in zip(labels, colors):
        ax.plot(balance.index, balance[label], color=c, label=label)
    ax.axhline(INITIAL, color="grey", linestyle="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("Balance after 30 years (EUR, log scale)")
    ax.set_xlabel("Month in which the windfall is received")
    ax.set_title("EUR 100,000 invested for 30 years without rebalancing (equity/bond policy)")
    ax.legend(title="Equity/Bonds", ncol=4, fontsize=8, loc="upper left")
    stats.save_figure(fig, "part5_balance_by_start.png")

    # Grafik 2: Rendite, Volatilität, Sharpe je Startmonat
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, df, title in zip(axes, [ann_ret * 100, ann_vol * 100, sharpe],
                             ["Annualised return over 30 years (%)",
                              "Annualised volatility over 30 years (%)",
                              "Sharpe ratio over 30 years (rf = 1%)"]):
        for label, c in zip(labels, colors):
            ax.plot(df.index, df[label], color=c, label=label)
        ax.set_title(title)
    axes[0].legend(title="Equity/Bonds", ncol=4, fontsize=8, loc="upper left")
    axes[-1].set_xlabel("Month in which the windfall is received")
    stats.save_figure(fig, "part5_metrics_by_start.png")

    # Grafik 3: Risiko-Rendite-Abwägung über die Portfolios (Durchschnittswerte)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(summary["Avg. annualised volatility"] * 100, summary["Avg. annualised return"] * 100,
            marker="o", color=stats.COLORS[0])
    for label, row in summary.iterrows():
        ax.annotate(label, (row["Avg. annualised volatility"] * 100, row["Avg. annualised return"] * 100),
                    xytext=(5, -3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Average annualised volatility (%)")
    ax.set_ylabel("Average annualised return (%)")
    ax.set_title("Risk-return trade-off of the policy portfolios (averaged over 832 start dates)")
    stats.save_figure(fig, "part5_risk_return_tradeoff.png")


# ---------------------------------------------------------------------------
# 5.2 Market Timing
# ---------------------------------------------------------------------------
def _market_timing(factors):
    mkt, rf = factors["Mkt"], factors["RF"]
    passive = INITIAL * (1 + mkt).prod()               # 100 % Aktien, immer investiert
    print(f"\n--- 5.2 Market Timing ({mkt.index[0]:%Y-%m} bis {mkt.index[-1]:%Y-%m}, {len(mkt)} Monate) ---")
    print(f"Passiv 100 % Aktien: {INITIAL:,.0f} EUR -> {passive:,.0f} EUR")

    # Beste Aktienmonate, absteigend sortiert
    best_months = mkt.sort_values(ascending=False)
    all_bonds = INITIAL * (1 + rf).prod()              # Wert, wenn man nie in Aktien ist

    # Timing mit k besten Monaten: in diesen Monaten Aktienrendite statt RF-Rendite
    #   Wert(k) = Wert(nur Anleihen) * Π_{beste k} (1+mkt)/(1+rf)
    ratio = ((1 + best_months) / (1 + rf[best_months.index])).values
    timing = all_bonds * np.concatenate([[1.0], np.cumprod(ratio)])     # k = 0, 1, 2, ...
    k = np.arange(len(timing))
    table = pd.DataFrame({"Months timed perfectly": k, "Final balance timing (EUR)": timing,
                          "Final balance passive (EUR)": passive,
                          "Timing beats passive": timing > passive}).set_index("Months timed perfectly")
    k_needed = int(table.index[table["Timing beats passive"]][0])
    print(f"Nur Anleihen: {all_bonds:,.0f} EUR")
    print(f"Man muss die besten {k_needed} Monate (von {len(mkt)}) perfekt vorhersagen, "
          f"um Buy-and-Hold zu schlagen.")
    show = table.loc[list(range(0, 11)) + [15, 20, 25, 30, 40, 50, k_needed - 1, k_needed]].sort_index()
    stats.save_table(show.drop_duplicates(), "part5_market_timing.csv")

    best_table = pd.DataFrame({"Market return": best_months.head(k_needed) * 100})
    best_table.index = best_table.index.strftime("%Y-%m")
    stats.save_table(best_table, "part5_best_months.csv", print_it=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    k_max = k_needed + 15
    ax.plot(k[:k_max], timing[:k_max] / 1e6, color=stats.COLORS[0], marker="o", ms=3,
            label="Bonds + k best equity months (perfect timing)")
    ax.axhline(passive / 1e6, color=stats.COLORS[3], linestyle="--", label="Passive 100% equity, buy and hold")
    ax.axvline(k_needed, color="grey", linestyle=":", lw=0.8)
    ax.annotate(f"k = {k_needed}", (k_needed, passive / 1e6), xytext=(-40, 15), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlabel("k = number of best months predicted perfectly (out of %d)" % len(mkt))
    ax.set_ylabel("Final balance (EUR million, log scale)")
    ax.set_title("Market timing vs. buy and hold, Jan 1927 – Apr 2026")
    ax.legend(loc="lower right")
    stats.save_figure(fig, "part5_market_timing.png")


if __name__ == "__main__":
    stats.setup_plot_style()
    run()
