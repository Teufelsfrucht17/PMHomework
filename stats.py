"""
stats.py – Gemeinsame Werkzeuge für alle Aufgabenteile.

Enthält
  * Performance-Kennzahlen (annualisierte Rendite, Volatilität, Sharpe Ratio,
    maximaler Drawdown, Schiefe, ...)
  * eine kleine Hülle um die lineare Regression (OLS) aus statsmodels
  * Hilfsfunktionen für Grafiken (einheitliches Aussehen, Farben, Speichern)

Konvention: Alle Renditen sind monatlich und als Dezimalzahl (0.01 = 1 %).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

MONTHS_PER_YEAR = 12

# Ausgabeordner (werden in main.py angelegt)
OUT_DIR = Path(__file__).parent / "output"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

# Farbpalette, die auch für Menschen mit Farbfehlsichtigkeit unterscheidbar ist
# (Okabe-Ito). Wird in fester Reihenfolge vergeben.
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
          "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


# ---------------------------------------------------------------------------
# Performance-Kennzahlen
# ---------------------------------------------------------------------------
def cumulative_growth(returns: pd.Series | pd.DataFrame):
    """Wert von 1 Geldeinheit im Zeitverlauf: (1+r1)*(1+r2)*... """
    return (1 + returns).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximaler Drawdown = größter prozentualer Verlust vom bisherigen
    Höchststand bis zum darauffolgenden Tiefpunkt (negative Zahl).
    """
    wealth = cumulative_growth(returns)
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1
    return drawdown.min()


def performance_table(returns: pd.DataFrame, rf_annual: float = 0.0) -> pd.DataFrame:
    """
    Kennzahlen-Tabelle für mehrere Renditereihen (eine Zeile pro Spalte).

    rf_annual: risikoloser Zins pro Jahr für die Sharpe Ratio (z. B. 0.01 = 1 %).
    Annualisierung: Mittelwert * 12 (arithmetisch), CAGR (geometrisch),
                    Volatilität * sqrt(12).
    """
    rows = {}
    for name, r in returns.items():
        r = r.dropna()
        n_years = len(r) / MONTHS_PER_YEAR
        mean_annual = r.mean() * MONTHS_PER_YEAR
        vol_annual = r.std() * np.sqrt(MONTHS_PER_YEAR)
        rows[name] = {
            "Start": r.index[0].strftime("%Y-%m"),
            "End": r.index[-1].strftime("%Y-%m"),
            "Months": len(r),
            "Mean return p.a.": mean_annual,
            "CAGR p.a.": (1 + r).prod() ** (1 / n_years) - 1,
            "Volatility p.a.": vol_annual,
            "Sharpe ratio": (mean_annual - rf_annual) / vol_annual,
            "Max drawdown": max_drawdown(r),
            "Skewness": r.skew(),
            "Min monthly return": r.min(),
            "Max monthly return": r.max(),
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def ols(y: pd.Series, X: pd.DataFrame) -> dict:
    """
    Lineare Regression  y = alpha + b1*X1 + b2*X2 + ... + Fehler.

    Rückgabe: Dictionary mit den Koeffizienten ("alpha", "beta_<Name>"), den
    dazugehörigen t-Statistiken ("t(...)"), p-Werten, R² und adj. R².
    Fehlende Werte werden vorher entfernt, damit y und X exakt zusammenpassen.
    """
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    model = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit()

    out = {}
    for name in model.params.index:
        label = "alpha" if name == "const" else f"beta_{name}"
        out[label] = model.params[name]
        out[f"t({label})"] = model.tvalues[name]
        out[f"p({label})"] = model.pvalues[name]
    out["R2"] = model.rsquared
    out["adj. R2"] = model.rsquared_adj
    out["N"] = int(model.nobs)
    out["fitted"] = model.fittedvalues     # für Grafiken (tatsächlich vs. Modell)
    return out


def regression_table(returns: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Führt ols() für jede Spalte von `returns` aus und stapelt die Ergebnisse."""
    results = {}
    for name, y in returns.items():
        res = ols(y, X)
        res.pop("fitted")
        results[name] = res
    return pd.DataFrame(results).T


# ---------------------------------------------------------------------------
# Grafiken & Tabellen speichern
# ---------------------------------------------------------------------------
def setup_plot_style():
    """Einheitliches, ruhiges Aussehen für alle Grafiken."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "axes.prop_cycle": plt.cycler(color=COLORS),
        "legend.frameon": False,
        "font.size": 9,
    })


def save_figure(fig, filename: str):
    """Speichert eine Grafik als PNG in output/figures und schließt sie."""
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"   Grafik gespeichert: output/figures/{filename}")


def save_table(df: pd.DataFrame, filename: str, print_it: bool = True):
    """Speichert eine Tabelle als CSV in output/tables und zeigt sie an."""
    df.to_csv(TAB_DIR / filename)
    if print_it:
        with pd.option_context("display.width", 200, "display.max_columns", 30,
                               "display.float_format", "{:,.4f}".format):
            print(df)
    print(f"   Tabelle gespeichert: output/tables/{filename}")


def percent(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Hilfsfunktion: ausgewählte Spalten von Dezimal in Prozent umrechnen (für die Anzeige)."""
    df = df.copy()
    df[columns] = df[columns].astype(float) * 100
    return df
