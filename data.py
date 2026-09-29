"""
data.py – Einlesen und Aufbereiten der fünf Datensätze.

Alle Funktionen liefern "saubere" pandas-DataFrames zurück:
  * Zeilenindex  = Monat (als Datum, jeweils der Monatsanfang)
  * Spalten      = Wertpapiere / Faktoren / Länder
  * Renditen     = IMMER als Dezimalzahl (0.05 = 5 %), egal wie sie in der
                   Rohdatei stehen (Prozent, "3.2%", Preise ...).

Damit müssen sich die Aufgabenteile nicht mehr um Einheiten kümmern – das ist
genau die Stolperfalle, vor der die Aufgabenstellung warnt.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Ordner, in dem die CSV-Dateien liegen (relativ zu dieser Datei)
DATA_DIR = Path(__file__).parent / "Files_Homework"


def _yyyymm_to_date(index) -> pd.DatetimeIndex:
    """Wandelt einen Index wie 200401 (Jahr+Monat als Zahl) in ein Datum um."""
    return pd.to_datetime(index.astype(str), format="%Y%m")


# ---------------------------------------------------------------------------
# 1) Fama-French-Faktoren + risikoloser Zins
# ---------------------------------------------------------------------------
def load_factors() -> pd.DataFrame:
    """
    Monatliche Faktorrenditen Jan 1927 – Apr 2026.
    Spalten: Mkt-RF, SMB, HML, Mom, RF   (alle in Dezimal, nicht Prozent).
    Zusätzlich: 'Mkt' = Marktrendite inkl. Zins (= Mkt-RF + RF).
    """
    df = pd.read_csv(DATA_DIR / "HW_Factors.csv", index_col=0)
    df.index = _yyyymm_to_date(df.index)
    df = df / 100                       # Prozent -> Dezimal
    df["Mkt"] = df["Mkt-RF"] + df["RF"]  # Gesamtrendite des Marktes
    return df


# ---------------------------------------------------------------------------
# 2) Hedge-Fund-Indizes (HFRI)
# ---------------------------------------------------------------------------
def load_hedge_funds() -> pd.DataFrame:
    """
    Monatliche Renditen von 8 Hedge-Fund-Indizes, Jan 2005 – Mai 2026.
    Die Datei hat zwei Kopfzeilen (Kurzname + langer Name) und die Renditen
    stehen als Text mit Prozentzeichen ("1.24%"). Wir behalten die Kurznamen.
    """
    df = pd.read_csv(DATA_DIR / "HW_Hedge Fund.csv", index_col=0, skiprows=[1])
    df.index = pd.to_datetime(df.index, format="%b-%y")   # "Jan-05" -> 2005-01-01
    df.index.name = "Date"
    # "1.24%" -> 1.24 -> 0.0124
    df = df.apply(lambda col: col.str.rstrip("%").astype(float)) / 100
    return df


# ---------------------------------------------------------------------------
# 3) Länderindizes (20 Industrieländer, in USD)
# ---------------------------------------------------------------------------
def load_world() -> pd.DataFrame:
    """Monatliche Aktienrenditen von 20 Ländern, Jan 1991 – Dez 2025, Dezimal."""
    df = pd.read_csv(DATA_DIR / "HW_World.csv", index_col=0)
    df.index = _yyyymm_to_date(df.index)
    return df / 100


# ---------------------------------------------------------------------------
# 4) Dow-Jones-Aktien: Kurse, Aktienanzahl, Branchen, DJIA-Index
# ---------------------------------------------------------------------------
def load_prices() -> dict:
    """
    Liest HW_Prices.csv, die einen ungewöhnlichen Aufbau hat:
        Zeile 0: Firmenname        Zeile 1: Branche ("Major Group")
        Zeile 2: Aktien Jan 2004   Zeile 3: Aktien Dez 2024
        Zeile 4: Ticker            ab Zeile 5: Monatskurse (Jan 2004 – Dez 2024)
    Letzte Spalte = DJIA-Index, danach eine leere Spalte (wird verworfen).

    Rückgabe: Dictionary mit
        'info'        – Tabelle je Aktie: Name, Branche, Aktienanzahl 2004/2024
        'prices'      – Monatskurse der 30 Aktien (splitbereinigt, s.u.)
        'prices_raw'  – Monatskurse wie in der Datei (für Marktkapitalisierung)
        'returns'     – Monatsrenditen der 30 Aktien (Dezimal)
        'djia'        – Monatsrenditen des DJIA-Index
    """
    raw = pd.read_csv(DATA_DIR / "HW_Prices.csv", header=None)
    raw = raw.iloc[:, :32]                     # leere letzte Spalte abschneiden

    tickers = raw.iloc[4, 1:].tolist()         # Zeile 4 enthält die Ticker
    info = pd.DataFrame(
        {
            "Name": raw.iloc[0, 1:31].values,
            "Group": raw.iloc[1, 1:31].values,
            "Shares_2004": raw.iloc[2, 1:31].astype(float).values,
            "Shares_2024": raw.iloc[3, 1:31].astype(float).values,
        },
        index=tickers[:30],
    )

    prices = raw.iloc[5:, 1:].astype(float)
    prices.columns = tickers
    prices.index = _yyyymm_to_date(raw.iloc[5:, 0])
    prices.index.name = "Date"

    djia = prices.pop("DJIA")                  # Index separat behandeln
    prices_raw = prices.copy()

    # --- DJIA-Spalte ist um 6 Monate verschoben -------------------------------
    # Plausibilitätsprüfung: Die DJIA-Werte passen nicht zu ihren Datumszeilen.
    # Zeile "200808" enthält 7062.93 = echter DJIA-Schlusskurs vom Feb 2009,
    # Zeile "200401" enthält 10140 = Schlusskurs Jul 2004, Zeile "202412"
    # enthält 44095 = Schlusskurs Jun 2025. Die Korrelation mit den 30 Aktien
    # ist ohne Korrektur -0.13, mit 6 Monaten Verschiebung +0.97.
    # Wir rücken den Index deshalb um 6 Monate nach vorne: Der echte DJIA ist
    # damit von Jul 2004 bis Dez 2024 verfügbar (Jan-Jun 2004 fehlen).
    djia.index = djia.index + pd.DateOffset(months=6)
    djia = djia.loc[:prices.index[-1]]

    # --- Splitbereinigung -------------------------------------------------
    # Drei Aktiensplits (2:1) sind in den Rohdaten NICHT bereinigt und würden
    # sonst als "-50 % Monatsrendite" erscheinen. Wir teilen alle Kurse VOR dem
    # Split durch 2, damit die Renditen korrekt sind. Die Rohkurse bleiben für
    # die Marktkapitalisierung erhalten (dort passen Kurs und Aktienzahl zusammen).
    #   AAPL: Split Feb 2005 | UNH: Split Mai 2005 | CAT: Split Jul 2005
    for ticker, split_month in [("AAPL", "2005-02"), ("UNH", "2005-05"), ("CAT", "2005-07")]:
        prices.loc[prices.index < split_month, ticker] /= 2

    return {
        "info": info,
        "prices": prices,
        "prices_raw": prices_raw,
        "returns": prices.pct_change().dropna(),
        "djia": djia.pct_change().dropna().rename("DJIA"),
    }


# ---------------------------------------------------------------------------
# 5) ETF-Kurse
# ---------------------------------------------------------------------------
def load_etfs() -> pd.DataFrame:
    """
    Monatskurse von 16 ETFs, Jan 2000 – Jun 2026. Nicht jeder ETF existiert
    über den ganzen Zeitraum (NaN davor). Als Spaltennamen verwenden wir die
    Ticker in Klammern, z. B. "SPDR S&P 500 ETF Trust (SPY)" -> "SPY".
    """
    df = pd.read_csv(DATA_DIR / "HW_ETFs.csv", index_col=0)
    df.index = _yyyymm_to_date(df.index)
    df.index.name = "Date"
    df.columns = df.columns.str.extract(r"\((\w+)\)")[0].rename(None)   # Ticker herausziehen
    return df


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Monatsrendite = Kurs heute / Kurs Vormonat - 1 (fehlende Kurse bleiben NaN)."""
    return prices.pct_change(fill_method=None)
