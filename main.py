"""
Portfolio Management – Homework 2026
=====================================
Startpunkt des Projekts. Führt alle fünf Aufgabenteile nacheinander aus und
schreibt sämtliche Ergebnisse nach output/:
    output/tables/   – alle Ergebnistabellen als CSV
    output/figures/  – alle Grafiken als PNG

Ausführen:   python main.py
Einzelne Teile lassen sich auch separat starten, z. B.  python part1_hedge_funds.py

Projektaufbau:
    data.py                – Einlesen und Bereinigen der fünf CSV-Dateien
    stats.py               – gemeinsame Kennzahlen, Regressionen, Grafik-Hilfen
    part1_hedge_funds.py   – Teil 1: Hedge Funds (CAPM, 4-Faktor, Up/Down-Beta, AR(1))
    part2_momentum.py      – Teil 2: Internationale Momentum-Strategie
    part3_replication.py   – Teil 3: Zehn Portfolios aus den Dow-Jones-Aktien
    part4_retirement.py    – Teil 4: ETF-Portfolios für die Altersvorsorge
    part5_windfall.py      – Teil 5: Windfall (Policy-Portfolios, Market Timing)
"""

import part1_hedge_funds
import part2_momentum
import part3_replication
import part4_retirement
import part5_windfall
import stats


def main():
    stats.FIG_DIR.mkdir(parents=True, exist_ok=True)
    stats.TAB_DIR.mkdir(parents=True, exist_ok=True)
    stats.setup_plot_style()

    part1_hedge_funds.run()
    part2_momentum.run()
    part3_replication.run()
    part4_retirement.run()
    part5_windfall.run()

    print("\nFertig. Alle Tabellen liegen in output/tables, alle Grafiken in output/figures.")


if __name__ == "__main__":
    main()
