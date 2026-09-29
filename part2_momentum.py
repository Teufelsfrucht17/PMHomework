"""
Teil 2 – International Momentum (25 %)

Strategie (jeden Monat t neu):
  1. Sortiere die 20 Länder nach ihrer kumulierten Rendite über die 11 Monate
     t-12 bis t-2 (der letzte Monat t-1 wird übersprungen).
  2. Die 3 besten Länder ("Winners") bekommen zusammen 80 % (je 4/15),
     die 3 schlechtesten ("Losers") bekommen 0 %,
     die restlichen 14 Länder teilen sich 20 % (je 1/70).
  3. Halte das Portfolio im Monat t, danach neu sortieren (keine Kosten).

Ausgewertet werden:
  * Kumulierte Rendite (Grafik), Ø annualisierte Rendite, Volatilität,
    maximaler Drawdown, Sharpe Ratio
  * Vier-Faktor-Regression: Kann das Modell die Momentum-Rendite erklären?
Als Vergleich dient das gleichgewichtete Portfolio aller 20 Länder (je 1/20) –
das ist das "neutrale" Portfolio ohne Momentum-Tilt.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import stats

N_TOP = 3               # Anzahl Gewinner-/Verlierer-Länder
W_TOP = 0.80 / N_TOP    # = 4/15 Gewicht je Gewinnerland
W_BOTTOM = 0.0          # Gewicht je Verliererland
LOOKBACK = 11           # Monate im Sortierfenster (t-12 ... t-2)
SKIP = 1                # übersprungener Monat (t-1)


def run():
    print("\n=== Teil 2: International Momentum ===")
    world = data.load_world()
    factors = data.load_factors()

    weights = momentum_weights(world)
    # Portfolio-Rendite im Monat t = Summe(Gewicht_t * Länderrendite_t).
    # Die Gewichte stehen zum Monatsanfang fest (nur Vergangenheitsdaten!).
    strategy = (weights * world).sum(axis=1)[weights.notna().all(axis=1)]
    strategy.name = "Momentum strategy"

    # Vergleichsportfolios im gleichen Zeitraum
    ew = world.loc[strategy.index].mean(axis=1).rename("Equal-weight 20 countries")
    winners = _group_return(world, weights, W_TOP).rename("Winners (top 3)")
    losers = _group_return(world, weights, W_BOTTOM).rename("Losers (bottom 3)")
    wml = (winners - losers).rename("Winners minus Losers")

    fac = factors.loc[strategy.index]
    portfolios = pd.concat([strategy, ew, winners, losers, wml], axis=1)
    print(f"Zeitraum: {strategy.index[0]:%Y-%m} bis {strategy.index[-1]:%Y-%m} ({len(strategy)} Monate)")

    _performance(portfolios, fac)
    _cumulative_plot(strategy, ew, fac)
    _factor_regression(portfolios, fac)
    _winner_frequency(weights)


def momentum_weights(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet für jeden Monat t die Portfoliogewichte je Land.

    Signal_t = kumulierte Rendite von t-12 bis t-2
             = rollierendes 11-Monats-Produkt von (1+r), um 2 Monate verschoben.
    """
    cum_11m = (1 + returns).rolling(LOOKBACK).apply(np.prod, raw=True) - 1
    signal = cum_11m.shift(SKIP + 1)          # Fenster endet in t-2

    # Rang 1 = bestes Land ... Rang 20 = schlechtestes Land
    rank = signal.rank(axis=1, ascending=False)
    n = returns.shape[1]
    w_middle = (1 - N_TOP * W_TOP - N_TOP * W_BOTTOM) / (n - 2 * N_TOP)   # = 1/70

    weights = pd.DataFrame(w_middle, index=returns.index, columns=returns.columns)
    weights[rank <= N_TOP] = W_TOP
    weights[rank > n - N_TOP] = W_BOTTOM
    weights[signal.isna()] = np.nan           # erste 12 Monate: noch kein Signal
    return weights


def _group_return(world, weights, group_weight):
    """Gleichgewichtete Rendite aller Länder, die gerade das Gewicht `group_weight` haben."""
    mask = (weights == group_weight)
    return world[mask].mean(axis=1)[weights.notna().all(axis=1)]


def _performance(portfolios, fac):
    """Kennzahlen der Strategie im Vergleich zu Benchmark und Faktoren."""
    print("\n--- Performance ---")
    rf_annual = fac["RF"].mean() * 12
    # Faktoren zum Vergleich: Markt (inkl. RF) und Momentum-Faktor (long-short, ohne RF)
    compare = pd.concat([portfolios, fac[["Mkt", "Mom"]].rename(
        columns={"Mkt": "US market (Fama-French)", "Mom": "US momentum factor (Mom)"})], axis=1)
    table = stats.performance_table(compare, rf_annual)
    # Für Long-Short-Reihen (Winners minus Losers, Mom) ist der Sharpe ohne RF-Abzug korrekt,
    # weil sie selbst schon eine Differenz zweier Renditen sind.
    for name in ["Winners minus Losers", "US momentum factor (Mom)"]:
        table.loc[name, "Sharpe ratio"] = table.loc[name, "Mean return p.a."] / table.loc[name, "Volatility p.a."]
    print(f"Durchschnittlicher risikoloser Zins im Zeitraum: {rf_annual:.2%} p.a.")
    stats.save_table(table, "part2_performance.csv")


def _cumulative_plot(strategy, ew, fac):
    """Wachstum von 1 $ – Strategie vs. gleichgewichtete Benchmark vs. US-Markt."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    growth = stats.cumulative_growth(pd.concat(
        [strategy, ew, fac["Mkt"].rename("US market")], axis=1))
    for col in growth:
        axes[0].plot(growth.index, growth[col], label=col)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Growth of $1 (log scale)")
    axes[0].set_title("International momentum strategy: cumulative performance")
    axes[0].legend(loc="upper left")

    # Drawdown der Strategie darunter
    wealth = stats.cumulative_growth(strategy)
    dd = wealth / wealth.cummax() - 1
    axes[1].fill_between(dd.index, dd * 100, 0, color=stats.COLORS[3], alpha=0.5)
    axes[1].set_ylabel("Drawdown (%)")
    stats.save_figure(fig, "part2_cumulative_return.png")


def _factor_regression(portfolios, fac):
    """
    Vier-Faktor-Regression. Für Long-only-Portfolios wird die ÜBERSCHUSSRENDITE
    (minus RF) erklärt, für die Long-Short-Reihe "Winners minus Losers" die
    Rendite selbst (RF kürzt sich dort heraus).
    Ein signifikantes Alpha = Rendite, die die 4 Faktoren NICHT erklären können.
    """
    print("\n--- Vier-Faktor-Regression ---")
    X = fac[["Mkt-RF", "SMB", "HML", "Mom"]]
    y = portfolios.sub(fac["RF"], axis=0)
    y["Winners minus Losers"] = portfolios["Winners minus Losers"]
    y["Strategy minus Equal-weight"] = portfolios["Momentum strategy"] - portfolios["Equal-weight 20 countries"]
    table = stats.regression_table(y, X)
    table["alpha p.a."] = table["alpha"] * 12
    stats.save_table(table, "part2_four_factor.csv")

    # Zusätzlich CAPM als Referenz, um den Beitrag der weiteren Faktoren zu sehen
    capm = stats.regression_table(y, fac[["Mkt-RF"]])
    capm["alpha p.a."] = capm["alpha"] * 12
    stats.save_table(capm, "part2_capm.csv", print_it=False)


def _winner_frequency(weights):
    """Wie oft war jedes Land unter den Gewinnern bzw. Verlierern? (Balkengrafik)"""
    w = weights.dropna()
    freq = pd.DataFrame({
        "Winner (top 3)": (w == W_TOP).mean() * 100,
        "Loser (bottom 3)": (w == W_BOTTOM).mean() * 100,
    }).sort_values("Winner (top 3)", ascending=False)
    stats.save_table(freq, "part2_winner_loser_frequency.csv", print_it=False)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    idx = np.arange(len(freq))
    ax.bar(idx - 0.2, freq["Winner (top 3)"], width=0.4, label="Winner (top 3)", color=stats.COLORS[2])
    ax.bar(idx + 0.2, freq["Loser (bottom 3)"], width=0.4, label="Loser (bottom 3)", color=stats.COLORS[3])
    ax.set_xticks(idx, freq.index, rotation=45, ha="right")
    ax.set_ylabel("% of months")
    ax.set_title("How often is each country in the winner / loser group?")
    ax.legend()
    stats.save_figure(fig, "part2_winner_loser_frequency.png")


if __name__ == "__main__":
    stats.setup_plot_style()
    run()
