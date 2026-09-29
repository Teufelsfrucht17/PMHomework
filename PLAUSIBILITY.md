# Plausibilitätsprüfung – Daten, Ergebnisse und Grafiken

Ziel: Jedes Ergebnis wurde gegen (a) bekannte Referenzwerte aus Literatur/Markt,
(b) eine unabhängige zweite Rechnung im Code und (c) die Grafiken abgeglichen.
Alle Prüfungen sind reproduzierbar (Befehle im Anhang). Legende: ✅ plausibel,
⚠️ plausibel mit Einschränkung, ❌ Fehler gefunden und behoben.

---

## 1. Rohdaten

| Prüfung | Erwartung | Ergebnis | Urteil |
|---|---|---|---|
| Faktoren Jan 1927 – Apr 2026, Mkt-Rendite (Mkt-RF + RF) | US-Markt langfristig ca. 10 % p.a., Vol ca. 18–20 % | CAGR 10,3 %, Vol 18,4 %, Ø RF 3,2 % | ✅ |
| Marktfaktor AR(1) | liquider Index ≈ 0 | φ = 0,003 | ✅ (Referenz für Teil 1.4) |
| Alle Datensätze zeitlich zueinander ausgerichtet? Kreuzkorrelation mit Mkt-Faktor bei Lag −6…+6 | Maximum bei Lag 0 | DJ-Aktien 0,94, HF 0,81, World 0,82, SPY 0,99, IWM 0,91, EEM 0,75, MTUM 0,89, USRT 0,79 – alle Maximum bei Lag 0 | ✅ |
| **DJIA-Spalte in HW_Prices.csv** | Korrelation mit den 30 DJ-Aktien ≈ 0,95 | **−0,13 bei Lag 0, +0,97 bei Lag 6.** Zeile „200808“ = 7062,93 (echter Schluss Feb 2009), Zeile „200401“ = 10140 (Schluss Jul 2004), Zeile „202412“ = 44095 (Schluss Jun 2025) | ❌ **behoben**: Index um 6 Monate verschoben (`data.py`), DJIA ab Jul 2004 nutzbar |
| Aktiensplits in HW_Prices.csv | keine Monatsrendite < −40 % ohne Grund | AAPL Feb 2005 −42 %, UNH Mai 2005 −49 %, CAT Jul 2005 −43 % = bekannte 2:1-Splits | ❌ **behoben** (Kurse vor Split halbiert); ⚠️ HPQ Nov 2015 −53 % = HPE-Spin-off, unverändert gelassen und dokumentiert |
| Aktienzahl × Kurs Jan 2004 = Marktkapitalisierung | MSFT ≈ 300 Mrd, AAPL ≈ 8 Mrd (2004) | MSFT 10,1 % Gewicht, AAPL 0,3 % | ✅ (Aktienzahlen in Tsd., Kurse unbereinigt – passt zusammen) |
| ETF-Kursrenditen | SPY seit 2000 inkl. Dividenden ≈ 7,5 % p.a.; TLT/AGG ≈ 3–4 % p.a. | SPY 6,6 %, TLT 0,2 %, **AGG −0,2 %** | ⚠️ **Preisrenditen ohne Ausschüttungen.** Aktien-ETFs um ≈ 1,5–2 pp, Anleihen-ETFs um ≈ 3–4 pp unterschätzt. Aufgabe gibt nur Kurse vor; im Paper als Einschränkung genannt |
| Hedge-Fund-Datei: zwei Kopfzeilen, Prozent-Strings | 257 Monate Jan 2005 – Mai 2026 | 257 × 8, keine NaN | ✅ |
| World: 20 Länder, USD, Prozent | 420 Monate, keine NaN | ✅ | ✅ |

---

## 2. Teil 1 – Hedge Funds

| Prüfung | Referenz | Ergebnis | Urteil |
|---|---|---|---|
| Beta HFRI Composite | Literatur: 0,3–0,4 (Fung/Hsieh, Asness et al.) | 0,30 | ✅ |
| R² CAPM Composite | ca. 0,6–0,7 | 0,66 | ✅ |
| Alpha Composite | HFRI-Alphas nach 2005 klein positiv, 1–3 % p.a. | 1,8 % p.a., t = 2,5 | ✅ |
| Macro-Beta nahe 0, Equity-Styles 0,5–0,6 | Stilprofile | Macro 0,06, L/S 0,51, Value 0,57 | ✅ |
| Vier-Faktor: Growth lädt negativ auf HML, Value/Event-Driven positiv auf SMB | ökonomische Logik | Growth β_HML −0,09 (t −2,4), Value β_SMB 0,11, ED β_SMB 0,13 | ✅ |
| R²-Gewinn 4-Faktor vs CAPM | klein (Faktoren erklären HF schlecht) | +0,01 bis +0,05 | ✅ |
| Up/Down-Beta: Interaktionsregression vs. zwei getrennte Regressionen | gleiche Größenordnung | Composite 0,32/0,29 vs 0,30/0,28; ED 0,42/0,28 vs 0,46/0,31; FI Arb 0,22/0,12 vs 0,23/0,13 | ✅ (Unterschiede nur wegen gemeinsamer vs. getrennter Konstante) |
| AR(1) FI Arbitrage/Event-Driven hoch, Macro ≈ 0 | Getmansky/Lo/Makarov 2004: illiquide Stile 0,2–0,5 | FI Arb 0,43, ED 0,22, Macro 0,01 | ✅ |
| Grafiken | Streudiagramme: enge Wolke bei Value/L-S, formlos bei Macro; 45°-Plots 4-Faktor konsistent mit R² | geprüft | ✅ |

---

## 3. Teil 2 – International Momentum

| Prüfung | Erwartung | Ergebnis | Urteil |
|---|---|---|---|
| Gewichte je Monat | Summe = 1; genau 3 × 4/15, 3 × 0, 14 × 1/70 | Summe 1,000000 in allen 408 Monaten; Zähler [3], [3], [14] | ✅ |
| Signal-Timing (kein Look-ahead) | Gewichte Jan 1992 aus Renditen Jan–Nov 1991 | Handrechnung Top 3: HongKong, Australia, Singapore = Code | ✅ |
| Erster Monat | 12 Monate nach Datenstart (Jan 1991) | Jan 1992 | ✅ |
| Beta der Long-only-Portfolios zum US-Markt | ≈ 0,9 | 0,92 | ✅ |
| Momentum-Profit lädt auf MOM | Asness/Moskowitz/Pedersen 2013: Länder-Momentum korreliert mit US-Momentum | β_MOM 0,38 (t 8,3) für W−L | ✅ |
| W−L Sharpe vs. US-MOM | Länder-Momentum schwächer als Aktien-Momentum | 0,23 vs 0,28 | ✅ |
| Max Drawdown Strategie ≈ Benchmark | long-only, 100 % Aktien | −60,6 % vs −60,3 % (2008/09) | ✅ |
| Grafik: Japan häufigster Loser, kleine volatile Märkte häufig Winner | Japan 1990er/2000er-Bärenmarkt | Japan 32 % Loser-Monate | ✅ |

---

## 4. Teil 3 – Portfolio Replication

| Prüfung | Erwartung | Ergebnis | Urteil |
|---|---|---|---|
| Gewichtssummen aller 10 Portfolios | = 1 | 1,000 | ✅ |
| ERC: Risikobeiträge | alle = 1/30 = 0,0333 | min 0,0333, max 0,0333 | ✅ |
| MDP: Diversification Ratio höher als alle anderen | Definition | MDP 2,24 > ERC 1,91 > MinVar 1,89 > EW 1,74 > InvVol 1,73 | ✅ |
| Tangency/MinVar mit Leerverkäufen | Closed form erlaubt negative Gewichte | Tangency 300 % long / 200 % short; MinVar DD −14 % | ✅ erwartbar; ⚠️ nicht umsetzbar, im Text erklärt |
| Portfolio (i) repliziert DJIA | Korrelation > 0,9 | **nach DJIA-Korrektur** 0,95, TE 4,7 % p.a.; Renditen 8,0 % vs 8,1 % | ✅ (vorher −0,14 → Datenfehler entdeckt) |
| DJIA Preisindex 2004–2024 | Jul 2004 10 140 → Dez 2024 ≈ 42 500; CAGR ohne Dividenden ≈ 7 % | 7,3 % p.a. | ✅ |
| Tangency: in-sample vs out-of-sample | in-sample stark überhöht | Sharpe 2,40 (bis Jun 2014) vs 1,12 (ab Jul 2014); Rendite 40 % vs 24 % p.a. | ✅ Look-ahead sichtbar; oos immer noch hoch wegen AAPL/CRM (32 %/27 % Gewicht) |
| MinVar niedrigste Vola & kleinster Drawdown | Low-Vol-Effekt | 10,5 %, −15 % | ✅ |
| Alphabet-Portfolios | erste 15 Namen enthalten AAPL, BAC, HPQ, GS | Liste geprüft; Sortierung nach Firmenname (3M zuerst) | ✅ (Sortierung nach Ticker wäre eine mögliche Alternative) |
| Skewness negativ, min. Monatsrendite in 2008/2020 | Krisenmonate | alle außer Tangency/MDP negativ | ✅ |
| Grafik Heatmap | Vorzeichen/Skalierung stimmen mit Tabelle überein | geprüft | ✅ |
| Grafik Performance | DJIA und (i) laufen nach Korrektur parallel, Tiefpunkt Feb 2009 | geprüft | ✅ |

---

## 5. Teil 4 – Retirement Management

| Prüfung | Erwartung | Ergebnis | Urteil |
|---|---|---|---|
| Gewichtssummen der 7 Portfolios | = 1 (assert im Code) | ok | ✅ |
| Startmonate | erster Monat mit allen Renditen; USRT ab Nov 2016 → Dez 2016 | Baseline Aug 2002, (vi)/(vii) Dez 2016 | ✅ |
| Korrelation aller Portfolios mit Baseline | hoch (alle ≈ 60/40) | 0,89–0,97 | ✅ |
| Baseline 60/40 SPY/TLT ab 2002 | Literatur (Total Return) ca. 7–8 % p.a., Vol ≈ 10 % | 6,1 % CAGR, Vol 9,9 % | ⚠️ ca. 1,5–2 pp unter Total-Return-Werten, weil Preisrenditen |
| Anleihen-ETF-Renditen | AGG TR ≈ 3 % p.a. | AGG −0,2 % (Preis) | ⚠️ systematische Benachteiligung aller Anleihen-Sleeves (ii, iii, iv, v gegenüber vi) |
| TLT 2022 | bekannt −31 % TR | −33 % Preis, DD −51 % ab 2020 | ✅ |
| Ranking (vi) > (iii) > (i) > (vii) > (v) > (iv) > (ii) | US-Large-Cap-Dominanz 2017–2026, TLT-Crash 2022 | konsistent mit Marktgeschehen | ✅, aber ⚠️ (vi) ist verstecktes Aktien-Beta |
| Grafik Risk/Return: alle Punkte zwischen 11 % und 13 % Vola | ähnliche Aktienquote | geprüft | ✅ |

---

## 6. Teil 5 – Windfall Investment

| Prüfung | Erwartung | Ergebnis | Urteil |
|---|---|---|---|
| Anzahl Startmonate | Jan 1927 – Apr 1996 = 832 | 832 | ✅ |
| Buy-and-Hold ≠ Rebalancing | B&H mit Aktienanteil-Drift höher | 60/40: B&H Ø 1,63 Mio vs Rebalancing 1,23 Mio; Endaktienquote Median 91 % | ✅ (Hinweis der Aufgabe bestätigt) |
| Ranking der Portfolios über Startdaten konstant | mehr Aktien → höherer 30-J-Endwert | in allen 832 Starts | ✅ (Grafik: Linien kreuzen sich nie) |
| Schlechtester 100/0-Start | direkt vor 1929-Crash | Aug 1929 (868 k, 7,5 % p.a.) | ✅ |
| Kein Startmonat mit Verlust nach 30 J. | nominal | 0 % der Starts < 100 k | ✅ |
| Sharpe 0/100 | rf = 1 % konstant vorgegeben, tatsächl. RF Ø 3,2 % | 4,06 (Artefakt) | ⚠️ im Text erklärt, nicht interpretierbar |
| Passiv 100 % Aktien 1927–2026 | 100 k → ca. 1–2 Mrd (10–11 % p.a. über 99 J.) | 1,68 Mrd | ✅ |
| Beste Monate | Apr 1933 +38,9 %, Aug 1932 +37,2 %, Jul 1932 +33,6 % | identisch mit CRSP-Daten | ✅ |
| Anzahl nötiger Monate | Größenordnung „einige Dutzend“ (Best-Days-Studien) | 59 von 1192 (5 %); 24 davon vor 1946 | ✅ |
| Grafik: Volatilität für frühe Starts höher | 1930er-Volatilität | 100/0 bei 24 % für Starts 1927, 15–16 % danach | ✅ |

---

## 7. Zusammenfassung der Befunde

1. **Ein echter Datenfehler**: DJIA-Spalte 6 Monate verschoben. Ohne Korrektur wären Korrelation und Tracking Error in Teil 3 unbrauchbar (Korrelation −0,13). Behoben.
2. **Drei unbereinigte Splits** (AAPL, UNH, CAT 2005). Behoben.
3. **Preisrenditen ohne Ausschüttungen** in Teil 3 und 4. Nicht behebbar mit den gegebenen Daten; alle Aussagen zu Anleihen-ETFs und REITs sind entsprechend vorsichtig formuliert.
4. **Look-ahead** in Teil 3 (Portfolios ii, v–viii, x) ist beabsichtigt durch die Aufgabenstellung und wird im Text als solcher diskutiert.
5. Alle übrigen Ergebnisse liegen in den aus Literatur und Marktgeschichte bekannten Größenordnungen; Zweitrechnungen (getrennte Regressionen, Handrechnung des Momentum-Signals, Risikobeiträge, Diversification Ratio) stimmen mit dem Hauptcode überein.

## Anhang – Reproduktion

```bash
python main.py                      # alle Ergebnisse neu erzeugen
python -c "import data; d=data.load_prices(); print(d['returns'].mean(axis=1).corr(d['djia']))"   # DJIA-Check (0.97)
```
