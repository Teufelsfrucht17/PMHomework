# Portfolio Management – Homework 2026

Python-Lösung der fünf Aufgabenteile (Hedge Funds, International Momentum,
Portfolio Replication, Retirement Management, Windfall Investment).

## Schnellstart

```bash
pip install -r requirements.txt
python main.py
```

`main.py` führt alle Teile nacheinander aus (Laufzeit ca. 30 Sekunden) und legt
alle Ergebnisse in `output/` ab. Jeder Teil kann auch einzeln gestartet werden,
z. B. `python part2_momentum.py`.

## Projektstruktur

| Datei / Ordner | Inhalt |
|---|---|
| `main.py` | Startpunkt – ruft alle fünf Teile auf |
| `data.py` | Einlesen und Bereinigen der fünf CSV-Dateien (Einheiten, Datumsformate, Splits) |
| `stats.py` | Gemeinsame Werkzeuge: Kennzahlen (Rendite, Vola, Sharpe, Drawdown), Regression, Grafik-Hilfen |
| `part1_hedge_funds.py` | Teil 1: CAPM, Vier-Faktor-Modell, Up/Down-Beta, AR(1) |
| `part2_momentum.py` | Teil 2: Momentum-Strategie über 20 Länder + Faktor-Regression |
| `part3_replication.py` | Teil 3: Zehn Portfolios aus den 30 Dow-Jones-Aktien vs. DJIA |
| `part4_retirement.py` | Teil 4: Sieben ETF-Portfolios für ein 401k-Konto |
| `part5_windfall.py` | Teil 5: Buy-and-Hold-Policy-Portfolios über 30 Jahre + Market Timing |
| `Files_Homework/` | Rohdaten (CSV) und Aufgabenstellung (PDF) |
| `output/tables/` | Alle Ergebnistabellen als CSV (`partX_...csv`) |
| `output/figures/` | Alle Grafiken als PNG (`partX_...png`) |
| `output/run_log.txt` | Konsolenausgabe des letzten kompletten Laufs |
| `REPORT.md` | Ausformulierte Ergebnisse und Interpretation (ausführlich, Englisch) |
| `PLAUSIBILITY.md` | Plausibilitätsprüfung aller Daten, Ergebnisse und Grafiken (Deutsch) |
| `paper/` | Kurzes Paper zu allen fünf Teilen (`Homework_Paper.md` + `Homework_Paper.pdf`) |

## Konventionen im Code

* Alle Renditen werden in `data.py` in **Dezimalzahlen** umgerechnet (0.05 = 5 %),
  egal ob sie in der Rohdatei als Prozent, als Text ("1.2%") oder als Kurs vorliegen.
* Alle Regressionen (CAPM, Vier-Faktor) verwenden **Überschussrenditen**
  (Rendite minus risikoloser Zins RF aus `HW_Factors.csv`).
* Annualisierung: Mittelwert × 12, Volatilität × √12; zusätzlich wird die
  geometrische Rendite (CAGR) ausgewiesen.
* Jede Datei beginnt mit einem Kommentarblock, der erklärt, was sie tut; jede
  Funktion ist für Nicht-Programmierer kommentiert.

## Getroffene Annahmen (siehe auch REPORT.md)

* Teil 3: "prices as of Jan **2024**" in der Aufgabe wird als Tippfehler für Jan **2004** gelesen
  (Stichprobenbeginn; die Datei enthält "Shares Jan 2004").
* Teil 3: Drei nicht bereinigte Aktiensplits (AAPL Feb 2005, UNH Mai 2005, CAT Jul 2005)
  werden in `data.py` korrigiert. Der HPQ-Kurssturz im Nov 2015 (Abspaltung von HPE)
  bleibt unverändert und wird im Bericht erwähnt.
* Teil 3: Die DJIA-Spalte der Kursdatei ist um 6 Monate verschoben (Zeile "200808" enthält den
  echten Schlusskurs vom Feb 2009). `data.py` korrigiert das; der DJIA ist damit ab Jul 2004 verfügbar.
* Teil 3/4: Alle Kursreihen sind ohne Dividenden/Kupons (Preisrenditen). Das benachteiligt
  Anleihen-ETFs und REITs im Vergleich; siehe PLAUSIBILITY.md.
* Teil 4: Jedes Portfolio startet, sobald alle seine ETFs Kurse haben; der Vergleich
  erfolgt über den gemeinsamen Zeitraum ab Dez 2016 (Start des REIT-ETF USRT).
