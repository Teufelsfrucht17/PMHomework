"""
Teil 1 – Hedge Funds (20 %)

Fragen:
  1. CAPM:           Beta, Alpha, t-Werte und R² je Hedge-Fund-Index + Streudiagramm
  2. Vier-Faktor:    dasselbe mit Markt, SMB, HML und Momentum
  3. Zeitvariables Beta: Beta in steigenden vs. fallenden Märkten
  4. Autokorrelation: AR(1)-Regression je Index

Alle Regressionen verwenden ÜBERSCHUSSRENDITEN (Rendite minus risikoloser Zins),
weil CAPM und Faktormodelle Überschussrenditen erklären.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import stats


def run():
    print("\n=== Teil 1: Hedge Funds ===")

    # ---- Daten laden und auf den gemeinsamen Zeitraum bringen --------------
    hf = data.load_hedge_funds()
    factors = data.load_factors()
    common = hf.index.intersection(factors.index)      # Jan 2005 – Apr 2026
    hf, factors = hf.loc[common], factors.loc[common]
    excess = hf.sub(factors["RF"], axis=0)             # Überschussrendite je Index
    print(f"Gemeinsamer Zeitraum: {common[0]:%Y-%m} bis {common[-1]:%Y-%m} ({len(common)} Monate)")

    capm_table = _capm(excess, factors)
    ff4_table = _four_factor(excess, factors)
    _compare_r2(capm_table, ff4_table)
    _time_varying_beta(excess, factors)
    _autocorrelation(hf)


# ---------------------------------------------------------------------------
# 1.1 CAPM
# ---------------------------------------------------------------------------
def _capm(excess, factors):
    """Regression: (r_HF - rf) = alpha + beta * (r_Mkt - rf)"""
    print("\n--- 1.1 CAPM ---")
    table = stats.regression_table(excess, factors[["Mkt-RF"]])
    table["alpha p.a."] = table["alpha"] * 12          # Monats-Alpha auf Jahr hochgerechnet
    stats.save_table(table, "part1_capm.csv")

    # Streudiagramm: Überschussrendite des Index gegen die Marktüberschussrendite,
    # dazu die geschätzte CAPM-Gerade. Je enger die Punkte an der Geraden liegen,
    # desto besser erklärt das CAPM die Renditen (hohes R²).
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True, sharey=True)
    x = factors["Mkt-RF"] * 100
    for ax, name in zip(axes.flat, excess.columns):
        y = excess[name] * 100
        ax.scatter(x, y, s=10, alpha=0.6, color=stats.COLORS[0])
        a, b = table.loc[name, "alpha"] * 100, table.loc[name, "beta_Mkt-RF"]
        xs = np.linspace(x.min(), x.max(), 2)
        ax.plot(xs, a + b * xs, color=stats.COLORS[3])
        ax.set_title(f"{name}\nβ={b:.2f}, R²={table.loc[name, 'R2']:.2f}")
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
    fig.supxlabel("Market excess return (% per month)")
    fig.supylabel("Hedge fund excess return (% per month)")
    stats.save_figure(fig, "part1_capm_scatter.png")
    return table


# ---------------------------------------------------------------------------
# 1.2 Vier-Faktor-Modell (Carhart)
# ---------------------------------------------------------------------------
def _four_factor(excess, factors):
    """Regression: (r_HF - rf) = alpha + b1*Mkt-RF + b2*SMB + b3*HML + b4*Mom"""
    print("\n--- 1.2 Vier-Faktor-Modell ---")
    X = factors[["Mkt-RF", "SMB", "HML", "Mom"]]
    table = stats.regression_table(excess, X)
    table["alpha p.a."] = table["alpha"] * 12
    stats.save_table(table, "part1_four_factor.csv")

    # Bei vier Faktoren kann man nicht mehr "Rendite gegen Faktor" zeichnen.
    # Stattdessen: tatsächliche Rendite gegen die vom Modell vorhergesagte Rendite.
    # Ein perfektes Modell hätte alle Punkte auf der 45°-Linie.
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, excess.columns):
        fitted = stats.ols(excess[name], X)["fitted"] * 100
        actual = excess[name].loc[fitted.index] * 100
        ax.scatter(fitted, actual, s=10, alpha=0.6, color=stats.COLORS[0])
        lim = [min(actual.min(), fitted.min()), max(actual.max(), fitted.max())]
        ax.plot(lim, lim, color=stats.COLORS[3], label="45° line")
        ax.set_title(f"{name}\nR²={table.loc[name, 'R2']:.2f}")
    axes[0, 0].legend(loc="upper left")
    fig.supxlabel("Fitted excess return from 4-factor model (% per month)")
    fig.supylabel("Actual hedge fund excess return (% per month)")
    stats.save_figure(fig, "part1_four_factor_scatter.png")
    return table


def _compare_r2(capm, ff4):
    """Gegenüberstellung: Wie viel Erklärungskraft bringt das 4-Faktor-Modell zusätzlich?"""
    comp = pd.DataFrame({
        "R2 CAPM": capm["R2"],
        "R2 4-factor": ff4["R2"],
        "adj. R2 CAPM": capm["adj. R2"],
        "adj. R2 4-factor": ff4["adj. R2"],
        "alpha p.a. CAPM": capm["alpha p.a."],
        "alpha p.a. 4-factor": ff4["alpha p.a."],
    })
    comp["R2 gain"] = comp["R2 4-factor"] - comp["R2 CAPM"]
    print("\n--- Vergleich CAPM vs. Vier-Faktor-Modell ---")
    stats.save_table(comp, "part1_capm_vs_four_factor.csv")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    idx = np.arange(len(comp))
    ax.bar(idx - 0.2, comp["adj. R2 CAPM"], width=0.4, label="CAPM", color=stats.COLORS[0])
    ax.bar(idx + 0.2, comp["adj. R2 4-factor"], width=0.4, label="4-factor", color=stats.COLORS[1])
    ax.set_xticks(idx, comp.index, rotation=25, ha="right")
    ax.set_ylabel("adjusted R²")
    ax.set_title("Explanatory power: CAPM vs. four-factor model")
    ax.legend()
    stats.save_figure(fig, "part1_r2_comparison.png")


# ---------------------------------------------------------------------------
# 1.3 Zeitvariables Beta (Auf- vs. Abwärtsmärkte)
# ---------------------------------------------------------------------------
def _time_varying_beta(excess, factors):
    """
    Aufwärtsmarkt = Marktüberschussrendite > 0, Abwärtsmarkt = < 0.

    Wir schätzen EINE Regression mit einem Zusatzterm:
        r = alpha + beta_down * Mkt + (beta_up - beta_down) * Mkt * D_up
    D_up ist 1 in Aufwärtsmonaten, sonst 0. Der Koeffizient auf dem Zusatzterm
    ist direkt die DIFFERENZ der beiden Betas – sein t-Wert sagt uns, ob der
    Unterschied statistisch signifikant ist.
    """
    print("\n--- 1.3 Beta in Auf- und Abwärtsmärkten ---")
    mkt = factors["Mkt-RF"]
    up = (mkt > 0).astype(float)
    X = pd.DataFrame({"Mkt-RF": mkt, "Mkt-RF x Up": mkt * up})
    raw = stats.regression_table(excess, X)

    table = pd.DataFrame({
        "beta down-market": raw["beta_Mkt-RF"],
        "beta up-market": raw["beta_Mkt-RF"] + raw["beta_Mkt-RF x Up"],
        "difference (up - down)": raw["beta_Mkt-RF x Up"],
        "t(difference)": raw["t(beta_Mkt-RF x Up)"],
        "p(difference)": raw["p(beta_Mkt-RF x Up)"],
    })
    print(f"Aufwärtsmonate: {int(up.sum())}, Abwärtsmonate: {int((1 - up).sum())}")
    stats.save_table(table, "part1_up_down_beta.csv")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    idx = np.arange(len(table))
    ax.bar(idx - 0.2, table["beta down-market"], width=0.4, label="Down market (Mkt-RF < 0)", color=stats.COLORS[3])
    ax.bar(idx + 0.2, table["beta up-market"], width=0.4, label="Up market (Mkt-RF > 0)", color=stats.COLORS[2])
    ax.set_xticks(idx, table.index, rotation=25, ha="right")
    ax.set_ylabel("CAPM beta")
    ax.set_title("Market beta of hedge fund indices in down vs. up markets")
    ax.legend()
    stats.save_figure(fig, "part1_up_down_beta.png")


# ---------------------------------------------------------------------------
# 1.4 Autokorrelation (AR(1))
# ---------------------------------------------------------------------------
def _autocorrelation(hf):
    """
    AR(1): r_t = c + phi * r_{t-1} + Fehler.
    phi > 0 und signifikant bedeutet: auf einen guten Monat folgt tendenziell
    wieder ein guter Monat (Renditen sind "geglättet" / illiquide Positionen).
    """
    print("\n--- 1.4 Autokorrelation (AR(1)) ---")
    rows = {}
    for name, r in hf.items():
        res = stats.ols(r, r.shift(1).to_frame("lag1"))     # Rendite auf Vormonatsrendite
        rows[name] = {
            "AR(1) coefficient": res["beta_lag1"],
            "t-stat": res["t(beta_lag1)"],
            "p-value": res["p(beta_lag1)"],
            "significant at 5%": res["p(beta_lag1)"] < 0.05,
            "R2": res["R2"],
        }
    table = pd.DataFrame(rows).T
    stats.save_table(table, "part1_autocorrelation.csv")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    colors = [stats.COLORS[3] if s else stats.COLORS[0] for s in table["significant at 5%"]]
    ax.bar(table.index, table["AR(1) coefficient"].astype(float), color=colors)
    ax.set_xticks(range(len(table)), table.index, rotation=25, ha="right")
    ax.set_ylabel("AR(1) coefficient")
    ax.set_title("First-order autocorrelation of monthly returns (red = significant at 5%)")
    stats.save_figure(fig, "part1_autocorrelation.png")


if __name__ == "__main__":
    stats.setup_plot_style()
    run()
