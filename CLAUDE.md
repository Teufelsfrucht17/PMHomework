# Projektkontext: Portfolio Management Homework 2026

Diese Datei gibt jeder neuen Claude-Sitzung den nötigen Kontext. Bitte aktuell halten.

## Worum es geht

Hausarbeit im Kurs **Portfolio Management, Fall 2026** (Prof. Dr. Paula Cocoma, Frankfurt School).
Gruppenarbeit mit 1–3 Personen, zählt 40 Punkte der Endnote.

- **Abgabe:** nur die PDF wird bewertet, Upload in Canvas vor der Deadline. Code darf als Zusatzmaterial mit.
- **Verteidigung:** 13. Oktober 2026. Mindestens ein Gruppenmitglied muss anwesend sein und Fragen beantworten können, sonst 20 % Abzug für die ganze Gruppe.
- **TA:** Maria Chiara Leone.
- **Aufgabenstellung:** `Files_Homework/PM___Homework2026.pdf`

## Wünsche des Nutzers (immer beachten)

- Antworten auf **Deutsch**.
- Code **ausführlich kommentieren, verständlich für Nicht-Programmierer** (Kommentare auf Deutsch).
- Code **einfach halten**, keine unnötigen Teile, klare und leicht verständliche Struktur.
- Das Paper selbst ist auf **Englisch** (Kurssprache).

## Aufgaben und Gewichtung

| Teil | Thema | Gewicht | Daten |
|---|---|---|---|
| 1 | Hedge Funds: CAPM, 4-Faktor, Up/Down-Beta, AR(1) | 20 % | HW_Hedge Fund.csv, HW_Factors.csv |
| 2 | International Momentum (20 Länder) | 25 % | HW_World.csv, HW_Factors.csv |
| 3 | Portfolio Replication: 10 Portfolios aus DJIA-Aktien | 25 % | HW_Prices.csv |
| 4 | Retirement Management: 7 ETF-Portfolios | 15 % | HW_ETFs.csv |
| 5 | Windfall: 30-J-Policy-Portfolios + Market Timing | 15 % | HW_Factors.csv |

## Projektstruktur

```
main.py                 Startpunkt, führt alle 5 Teile aus (python main.py, ca. 30 s)
data.py                 Einlesen + Bereinigen aller CSVs, Renditen immer als Dezimalzahl
stats.py                Kennzahlen, OLS-Regression, Grafik-/Tabellen-Helfer
part1_hedge_funds.py … part5_windfall.py   je ein Aufgabenteil, jeweils mit run()
output/tables/          alle Ergebnistabellen (CSV)
output/figures/         alle Grafiken (PNG)
output/run_log.txt      Konsolenausgabe des letzten Laufs
REPORT.md               ausführliche Ergebnisse + Interpretation (Englisch)
PLAUSIBILITY.md         Plausibilitätsprüfung aller Daten/Ergebnisse (Deutsch)
paper/Homework_Paper.md Quelle des Abgabe-Papers (Englisch), daraus wird die PDF gebaut
paper/Homework_Paper.pdf aktuelles Paper, 8 Seiten
```

Umgebung: Anaconda-Python 3.12 unter `/opt/anaconda3/bin/python3` mit pandas, numpy, matplotlib, scipy, statsmodels.

PDF neu bauen:

```bash
cd paper && pandoc Homework_Paper.md -o Homework_Paper.pdf --pdf-engine=xelatex --columns=120
```

## Konventionen im Code

- Renditen monatlich, als Dezimalzahl (0.05 = 5 %). Umrechnung passiert nur in `data.py`.
- Regressionen immer mit **Überschussrenditen** (Rendite minus RF aus der Faktordatei).
- Annualisierung: Mittelwert × 12, Volatilität × √12; zusätzlich CAGR.
- Sharpe: Teil 3 mit rf = 0, Teil 4 und 5 mit rf = 1 % (Vorgabe der Aufgabe), Teil 2 mit Ø T-Bill-Zins.
- Grafiken mit farbenblind-tauglicher Okabe-Ito-Palette aus `stats.COLORS`.
- Jedes Modul beginnt mit einem Erklärblock, Dateinamen der Ausgaben beginnen mit `partX_`.

## Datenprobleme, die bereits behoben sind (nicht nochmal "reparieren")

1. **DJIA-Spalte in HW_Prices.csv ist um 6 Monate verschoben.** Zeile "200808" enthält 7062.93, den echten Schlusskurs vom Feb 2009. Korrelation mit den 30 Aktien ohne Korrektur −0.13, mit Korrektur +0.97. `data.py` verschiebt den Index um 6 Monate; DJIA-Renditen gibt es damit ab Aug 2004.
2. **Drei unbereinigte 2:1-Splits:** AAPL Feb 2005, UNH Mai 2005, CAT Jul 2005. Kurse vor dem Split werden halbiert. Für die Marktkapitalisierung bleiben die Rohkurse (`prices_raw`).
3. **HPQ Nov 2015 −53 %** ist die HPE-Abspaltung und bleibt bewusst unverändert (dokumentiert).
4. **Alle Kurse ohne Dividenden/Kupons.** Nicht behebbar; Anleihen-ETFs werden um ca. 3–4 pp p.a. unterschätzt. Im Paper als Einschränkung genannt.

## Getroffene Annahmen

- Teil 3: "prices as of Jan **2024**" in der Aufgabe wird als Tippfehler für Jan **2004** gelesen.
- Teil 3: Alphabet-Portfolios sortiert nach Firmenname (3M … Home Depot vs. Honeywell … Walmart).
- Teil 3: Tangency und Min-Var in geschlossener Form (Leerverkäufe erlaubt); MDP und ERC long-only per SLSQP.
- Teil 4: Portfolio startet, sobald alle ETFs Kurse haben; fairer Vergleich über gemeinsamen Zeitraum Dez 2016 – Jun 2026.
- Teil 5: 832 Startmonate (Jan 1927 – Apr 1996), Buy-and-Hold ohne Rebalancing, Sharpe mit konstantem rf = 1 %.

## Kernergebnisse (für schnelle Orientierung)

- **Teil 1:** HF-Composite β = 0.30, α = 1.8 % p.a. (t = 2.5), R² 0.66. 4-Faktor bringt nur +1–5 pp R². Event-Driven und FI Arbitrage haben signifikant höheres Beta in fallenden Märkten. AR(1) signifikant bei 5 von 8 Indizes, FI Arb φ = 0.43.
- **Teil 2:** Momentum 12.2 % p.a. vs. 10.3 % Benchmark, Sharpe 0.56 vs 0.46. Profit lädt stark auf US-MOM (β 0.38, t 8.3), Alpha danach insignifikant.
- **Teil 3:** Cap-weight (i) repliziert DJIA (Korr. 0.95, TE 4.7 %). Min-Var: niedrigste Vola 10.5 %, Drawdown −15 %. Tangency 31.8 % p.a. ist Look-ahead (Sharpe in-sample 2.4, out-of-sample 1.1, 5× Brutto-Exposure).
- **Teil 4:** Gemeinsamer Zeitraum: (vi) SPY + Alternatives am besten (10.4 % CAGR, Sharpe 0.76), (ii) international am schlechtesten (4.3 %). Treiber: US-Large-Cap-Dekade und TLT-Crash 2022.
- **Teil 5:** 100/0 im Schnitt €2.46 Mio nach 30 J., schlechtester Start Aug 1929 immer noch €868k. Market Timing: man muss die **59 besten von 1192 Monaten** perfekt treffen, um Buy-and-Hold zu schlagen.

## Stand und offene Punkte

- [x] Alle 5 Teile implementiert, laufen fehlerfrei
- [x] Plausibilitätsprüfung (PLAUSIBILITY.md)
- [x] Paper als PDF (8 Seiten)
- [ ] Gruppennamen und Matrikelnummern im Paper-Kopf eintragen (`author:` in `paper/Homework_Paper.md`)
- [ ] Tabellen im PDF gegen Seitenumbruch sichern (optional)
- [ ] Git-Commit (bisher ist nichts committet; `.idea/` und `__pycache__/` sind in `.gitignore`)
- [ ] Vorbereitung auf die Verteidigung am 13.10.2026
