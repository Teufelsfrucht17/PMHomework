# Portfolio Management – Homework 2026: Results and Interpretation

All numbers below are produced by `python main.py`. Full tables are in `output/tables/`,
figures in `output/figures/`. Returns are monthly; "p.a." means annualised
(mean × 12, volatility × √12). Excess returns are computed with the 1‑month T‑bill
rate (`RF`) from `HW_Factors.csv`.

---

## Part 1 – Hedge Funds

**Data.** HFRI 400 indices Jan 2005 – May 2026 (percent strings converted to decimals),
Fama‑French factors to Apr 2026. Common sample: **Jan 2005 – Apr 2026, 256 months**.
All regressions use excess returns `r_HF − RF`.

### 1.1 CAPM

`r_HF − RF = α + β (Mkt − RF) + ε`  (figure: `part1_capm_scatter.png`)

| Index | α p.a. | t(α) | β | t(β) | R² |
|---|---:|---:|---:|---:|---:|
| Aggregate Index | 1.8 % | 2.48 | 0.30 | 22.4 | 0.66 |
| Long/Short | 1.3 % | 1.37 | 0.51 | 28.7 | 0.76 |
| Market Neutral | 1.8 % | 2.91 | 0.09 | 8.1 | 0.21 |
| Growth | 1.8 % | 1.37 | 0.53 | 21.3 | 0.64 |
| Value | 0.3 % | 0.36 | 0.57 | 33.6 | 0.82 |
| Event‑Driven | 1.6 % | 1.58 | 0.35 | 19.2 | 0.59 |
| Macro | 2.3 % | 1.95 | 0.06 | 2.6 | 0.03 |
| FI Arbitrage | 2.4 % | 3.45 | 0.17 | 13.2 | 0.41 |

**Interpretation.**
* The aggregate hedge‑fund index has a market beta of only **0.30** but a positive,
  statistically significant alpha of about **1.8 % p.a.** (t = 2.5). This is the
  mirror image of the mutual‑fund industry: actively managed equity mutual funds
  typically have betas close to 1 and, net of fees, alphas that are zero or
  slightly negative (Jensen 1968, Carhart 1997, Fama‑French 2010). Hedge funds
  therefore deliver something mutual funds mostly do not: low market exposure
  plus a (small) positive alpha, i.e. returns that look like "diversifying skill".
* CAPM works well for the equity‑oriented styles (Long/Short, Growth, Value:
  R² 0.64–0.82) – they are essentially "diluted" equity portfolios. It works
  poorly for Market Neutral (R² 0.21), FI Arbitrage (0.41) and especially Macro
  (R² 0.03, beta 0.06): their return drivers are not the equity market. The
  scatter plots show this directly: tight clouds around the regression line for
  the equity styles, a shapeless cloud for Macro.
* Note the positive alphas of FI Arbitrage and Market Neutral: their returns are
  smooth and their strategies are "short volatility" in nature; part of this
  alpha is compensation for tail/liquidity risk that a linear CAPM cannot see
  (see 1.4).

### 1.2 Four‑factor model (Carhart)

`r_HF − RF = α + β_M (Mkt−RF) + β_SMB SMB + β_HML HML + β_MOM MOM + ε`
(figure: `part1_four_factor_scatter.png`, `part1_r2_comparison.png`)

| Index | α p.a. | t(α) | β_Mkt | β_SMB | β_HML | β_MOM | R² | ΔR² vs CAPM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggregate Index | 1.9 % | 2.57 | 0.30 | **0.06** (2.4) | 0.01 | 0.02 | 0.67 | +0.01 |
| Long/Short | 1.5 % | 1.56 | 0.50 | **0.11** (3.2) | −0.03 | 0.01 | 0.77 | +0.01 |
| Market Neutral | 1.6 % | 2.61 | 0.10 | 0.00 | **0.03** (2.0) | **0.05** (3.6) | 0.25 | +0.04 |
| Growth | 1.9 % | 1.40 | 0.53 | 0.08 | **−0.09** (−2.4) | 0.01 | 0.65 | +0.01 |
| Value | 0.7 % | 0.83 | 0.53 | **0.11** (3.7) | 0.04 | −0.02 | 0.83 | +0.01 |
| Event‑Driven | 2.0 % | 2.13 | 0.32 | **0.13** (3.9) | **0.08** (3.3) | −0.01 | 0.64 | +0.05 |
| Macro | 1.9 % | 1.62 | 0.08 | −0.03 | 0.05 | **0.07** (3.0) | 0.06 | +0.04 |
| FI Arbitrage | 2.7 % | 3.96 | 0.15 | 0.03 | 0.00 | **−0.05** (−3.2) | 0.44 | +0.03 |

(t‑statistics in parentheses for significant loadings.)

**Interpretation.**
* **Aggregate index:** the market is by far the dominant factor (β ≈ 0.30). The
  only additional significant exposure is a small positive **SMB** loading
  (hedge funds hold, on average, smaller stocks than the market). HML and MOM are
  not significant. Alpha stays at ≈ 1.9 % p.a. and significant – the extra
  factors do *not* explain hedge‑fund alpha away.
* **Style loadings make economic sense:** Growth funds load negatively on HML
  (growth = anti‑value), Value and Event‑Driven load positively on HML/SMB,
  Market Neutral and Macro have a positive momentum tilt (trend following),
  FI Arbitrage is negatively exposed to momentum (a carry/"short vol" profile).
* **Is the four‑factor model better?** Only marginally. R² rises by 1–5
  percentage points; the biggest gains are for Event‑Driven, Market Neutral and
  Macro, but for the latter R² remains at 0.06. The equity factors were built
  for long‑only equity portfolios; hedge‑fund returns also depend on non‑linear
  (option‑like) payoffs, credit, rates, FX and commodities, which neither model
  captures (Fung & Hsieh 2004 add trend‑following and bond factors for that
  reason). So: a better model, but not a good model for non‑equity styles.

### 1.3 Time‑varying beta: up vs. down markets

Up market = `Mkt−RF > 0` (164 months), down market = `< 0` (92 months). We
estimate one regression with an interaction term so that the t‑statistic tests
the *difference* directly (figure: `part1_up_down_beta.png`).

| Index | β down | β up | Difference (up − down) | t |
|---|---:|---:|---:|---:|
| Aggregate Index | 0.32 | 0.29 | −0.03 | −0.6 |
| Long/Short | 0.53 | 0.49 | −0.04 | −0.7 |
| Market Neutral | 0.07 | 0.11 | +0.04 | 1.0 |
| Growth | 0.55 | 0.52 | −0.03 | −0.4 |
| Value | 0.57 | 0.56 | 0.00 | 0.0 |
| Event‑Driven | 0.42 | 0.28 | **−0.14** | **−2.5** |
| Macro | 0.01 | 0.11 | +0.10 | 1.5 |
| FI Arbitrage | 0.22 | 0.12 | **−0.09** | **−2.3** |

**Interpretation.** For the aggregate index and the equity styles the up‑ and
down‑market betas are statistically indistinguishable – there is no evidence of
market‑timing ability, but also no evidence of the "bad" asymmetry. For
**Event‑Driven** and **FI Arbitrage** the down‑market beta is significantly
*higher* than the up‑market beta (0.42 vs 0.28 and 0.22 vs 0.12). These
strategies behave like a written put option: they participate little when
markets rise but lose disproportionately in sell‑offs (merger spreads widen,
credit/liquidity dries up – 2008, March 2020). This makes them *less*
attractive than the unconditional beta suggests: the diversification benefit
disappears exactly when it is needed most, and a mean‑variance investor would
over‑allocate to them if they only looked at the full‑sample beta. Only Macro
shows the desirable (but insignificant) pattern of a higher beta in up markets.

### 1.4 Autocorrelation: AR(1)

`r_t = c + φ r_{t−1} + ε`, on raw monthly returns (figure: `part1_autocorrelation.png`).

| Index | φ | t | p‑value |
|---|---:|---:|---:|
| Aggregate Index | 0.16 | 2.49 | 0.013 |
| Long/Short | 0.12 | 1.85 | 0.065 |
| Market Neutral | 0.13 | 2.03 | 0.043 |
| Growth | 0.15 | 2.41 | 0.017 |
| Value | 0.07 | 1.10 | 0.271 |
| Event‑Driven | 0.22 | 3.58 | <0.001 |
| Macro | 0.01 | 0.15 | 0.877 |
| FI Arbitrage | **0.43** | **7.56** | <0.001 |

**Interpretation.** Five of eight indices show significantly positive
first‑order autocorrelation, most strongly FI Arbitrage (φ = 0.43) and
Event‑Driven (0.22). For liquid, efficiently priced assets monthly returns should
be close to unpredictable (the S&P 500 has φ ≈ 0). Positive autocorrelation in
hedge‑fund indices is the classic signature of **illiquid holdings and return
smoothing** (Getmansky, Lo & Makarov 2004): positions that are marked
to model or to stale prices spread a shock over several months. Consequences:
(i) reported volatility, and thus Sharpe ratios and the alphas in 1.1–1.2, are
**overstated** (true risk is higher); (ii) betas estimated on contemporaneous
returns are **understated** (lagged market betas would add exposure); (iii)
the "smooth" strategies are exactly the ones with the bad down‑market beta in
1.3, so liquidity risk is the common thread. Macro (liquid futures/FX) shows no
autocorrelation, consistent with this explanation.

---

## Part 2 – International Momentum

**Strategy.** Each month t the 20 countries are ranked on their cumulative
return over months t−12 … t−2 (11 months, month t−1 skipped). Top 3 get
4/15 each (80 % in total), bottom 3 get 0 %, the remaining 14 get 1/70 each
(20 %). Held for month t, then re‑sorted; no transaction costs. First month with
a full signal: **Jan 1992**; sample **Jan 1992 – Dec 2025, 408 months**. The
natural benchmark is the equal‑weighted portfolio of all 20 countries (the same
universe without the momentum tilt). We also report the pure long‑short spread
"Winners minus Losers".

### Performance (figure: `part2_cumulative_return.png`)

| | Mean p.a. | CAGR | Vol p.a. | Sharpe¹ | Max DD | Skew | Worst month |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Momentum strategy** | **12.2 %** | 11.2 % | 17.6 % | **0.56** | −60.6 % | −0.47 | −23.6 % |
| Equal‑weight 20 countries | 10.3 % | 9.2 % | 16.9 % | 0.46 | −60.3 % | −0.61 | −23.9 % |
| Winners (top 3) | 12.8 % | 11.7 % | 18.2 % | 0.57 | −61.0 % | −0.38 | −23.5 % |
| Losers (bottom 3) | 9.3 % | 7.3 % | 20.8 % | 0.33 | −65.3 % | −0.10 | −23.5 % |
| Winners minus Losers | 3.5 % | 2.4 % | 15.1 % | 0.23 | −43.3 % | 0.07 | −17.2 % |
| US market (Fama‑French) | 11.4 % | 10.8 % | 15.1 % | 0.60 | −50.3 % | −0.62 | −17.1 % |
| US momentum factor (MOM) | 4.6 % | 3.2 % | 16.2 % | 0.28 | −57.8 % | −1.47 | −34.3 % |

¹ Sharpe with the average T‑bill rate over the period (2.46 % p.a.); for the
long‑short series no rate is subtracted.

**Findings.**
* The momentum tilt adds about **1.9 percentage points p.a.** over the
  equal‑weighted benchmark at almost the same volatility; the Sharpe ratio rises
  from 0.46 to 0.56. $1 grows to ≈ $37 versus ≈ $20 for the benchmark. The
  winners beat the losers by 3.5 % p.a., so international country momentum
  exists – but it is weaker than the classic US stock‑level momentum premium
  and, in this sample, weaker than the US momentum factor's long‑run average.
* Because the strategy is long‑only with 80 % in three countries, it inherits
  the full market risk of developed‑market equities: a 60 % drawdown in
  2008–09 and a 38 % drawdown in 2000–03, identical to the benchmark. The tilt
  changes the *relative* performance, not the *absolute* risk.
* The winner/loser frequency table (`part2_winner_loser_frequency.png`) shows
  that small, volatile markets (Finland, New Zealand, Norway, Austria, Sweden)
  are most often in the top 3, Japan is by far the most frequent loser (32 % of
  months). Momentum sorts on past returns, so high‑volatility countries are
  over‑represented in both extremes.

### Can the four‑factor model explain the momentum profit? (`part2_four_factor.csv`)

| Dependent variable (excess return) | α p.a. | t(α) | β_Mkt | β_SMB | β_HML | β_MOM | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Momentum strategy | 1.0 % | 0.50 | 0.92 | 0.02 | 0.10 | 0.06 (1.7) | 0.59 |
| Equal‑weight 20 countries | −0.5 % | −0.32 | 0.92 | 0.04 | 0.14 (3.3) | −0.05 | 0.70 |
| Winners minus Losers | 2.0 % | 0.82 | −0.02 | −0.03 | 0.00 | **0.38** (8.3) | 0.17 |
| Strategy minus Equal‑weight | 1.5 % | 1.43 | 0.00 | −0.02 | −0.04 | **0.11** (5.7) | 0.10 |

* Both long‑only portfolios are, unsurprisingly, mostly US‑market exposure
  (β ≈ 0.92, R² 0.6–0.7). Under the CAPM the strategy's alpha is 1.8 % p.a.
  (t = 0.9), the active return (strategy − benchmark) has an alpha of 2.2 % p.a.
  with t = 2.05 – borderline significant.
* Adding the factors, the **momentum profit loads strongly and significantly on
  the US momentum factor** (Winners − Losers: β_MOM = 0.38, t = 8.3; active
  return: β_MOM = 0.11, t = 5.7) and on nothing else. The alpha of the active
  return shrinks from 2.2 % to 1.5 % p.a. and becomes insignificant (t = 1.4).
* **Conclusion:** the international country momentum profit is not an
  independent anomaly – it is largely the same phenomenon as US stock momentum,
  which comoves across markets (Asness, Moskowitz & Pedersen 2013, "Value and
  Momentum Everywhere"). The four‑factor model explains the *sign and a
  significant share* of the profit; the remaining alpha is positive but not
  statistically different from zero. The low R² of the long‑short regressions
  (0.10–0.17) shows, however, that country momentum is far from a perfect
  substitute for the US factor: most of its month‑to‑month variation is
  country‑specific. Compared to the factors themselves the strategy has a
  higher raw Sharpe ratio (0.56) than MOM (0.28) or HML, but only because it is
  a long‑only equity portfolio carrying the market premium; its *pure*
  momentum component has a Sharpe of 0.23, below MOM's 0.28 in the same period.

---

## Part 3 – Portfolio Replication (Dow Jones stocks)

**Data.** 30 DJIA stocks and the DJIA index, monthly prices Jan 2004 – Dec 2024
(returns Feb 2004 – Dec 2024, 251 months). Three unadjusted 2:1 stock splits
(AAPL Feb 2005, UNH May 2005, CAT Jul 2005) were corrected in `data.py`;
without the correction they would show up as −50 % months. The HPQ price drop of
−53 % in Nov 2015 reflects the spin‑off of HPE (value moved to new shares, not
lost) and is left as in the data. **The DJIA column in the file is shifted by six months** (e.g. the row labelled
Aug 2008 contains 7,062.93, the actual DJIA close of Feb 2009; the correlation
with the 30 stocks is −0.13 unadjusted and +0.97 after shifting). `data.py`
moves the index forward by six months, so the true DJIA is available from
Aug 2004 (returns) to Dec 2024. Estimation window for portfolios v–viii and x:
returns up to **June 2014** (125 months). Risk‑free rate = 0. All portfolios
hold fixed weights with costless monthly rebalancing.

**Assumption.** The task says "prices as of Jan 2024"; the file provides shares
as of Jan 2004 and the sample starts in Jan 2004, so we read this as Jan 2004.

### 3.1 Weights at inception (`part3_weights.csv`, `part3_weights_heatmap.png`)

Key features (full 30×10 table in the CSV):

| Portfolio | Construction | Largest positions | Notes |
|---|---|---|---|
| (i) Market cap Jan 2004 | shares 2004 × price Jan 2004 | MSFT 10 %, PFE 9.5 %, WMT 7.9 %, INTC 6.7 % | AAPL only 0.3 %, CRM 0.1 % |
| (ii) Market cap, prices Jun 2014 | shares 2004 × price Jun 2014 | MSFT 9.4 %, WMT 6.8 %, IBM 6.5 %, JNJ 6.5 % | AAPL 4.3 % (uses future prices) |
| (iii) EW first 15 names (3M … Home Depot) | 1/15 each | | includes AAPL, BAC, HPQ |
| (iv) EW last 15 names (Honeywell … Walmart) | 1/15 each | | includes MSFT, CRM, UNH |
| (v) Tangency | w ∝ Σ⁻¹μ | JNJ 42 %, DIS 38 %, AAPL 32 %, MCD 32 % | short DD −40 %, INTC −30 %, KO −26 %, PG −24 %; gross exposure ≈ 5× |
| (vi) Minimum variance | w ∝ Σ⁻¹1 | WMT 19 %, IBM 19 %, MMM 14 % | shorts DD −14 %, HON −14 % |
| (vii) Inverse volatility | w ∝ 1/σ | JNJ 5.5 %, PG 5.1 %, MCD 5.0 % | all between 1.6 % and 5.5 % |
| (viii) Most diversified (long‑only) | max (w'σ)/√(w'Σw) | WMT 20 %, CRM 14 %, AAPL 9 % | 15 stocks at zero weight |
| (ix) Equal group budget | 20 % per industry, EW inside | Finance and Energy stocks 5 % each | Manufacturing (10 stocks) only 2 % each |
| (x) Equal risk contribution (long‑only) | RC_i = 1/30 of variance | WMT 6.9 %, CRM 5.8 %, PG 5.0 % | between 1.3 % (BAC) and 6.9 % |

### 3.2 Which portfolios are comparable?

* **Investable at the time vs. look‑ahead:** (i), (iii), (iv), (ix) and the
  DJIA use *only* information available on 1 Jan 2004 (or none at all). (ii)
  uses June‑2014 prices and (v)–(viii), (x) use return moments estimated on
  Feb 2004 – Jun 2014. Their performance in the first half of the sample is
  **in‑sample** and biased upwards; only Jul 2014 – Dec 2024 is a fair
  out‑of‑sample test. This matters enormously for (v).
* **Long‑only vs. long‑short:** (v) and (vi) contain short positions; (v) has
  ≈ 300 % long / 200 % short (gross exposure 5×). They are not comparable to the long‑only,
  fully‑invested portfolios in terms of leverage, implementability and risk.
* **Estimation‑free vs. estimated:** (i)–(iv), (ix) need no statistical
  estimates; (vii), (viii), (x) need only volatilities/correlations (robust);
  (v) needs expected returns – the most error‑prone input (Michaud 1989,
  DeMiguel et al. 2009). Among the estimated portfolios, (vi), (vii), (viii)
  and (x) form a natural "risk‑based" family and are comparable with each other.
* **Same universe, different weighting:** (i), (vi)–(x) and the DJIA all hold
  all 30 stocks; (iii)/(iv) each hold only half the universe and are really
  comparable only to each other (and their average is the EW‑30 portfolio).
  The DJIA is price‑weighted, which none of our portfolios replicate exactly;
  the closest in spirit are (i) and (ii).

### 3.3 Performance Feb 2004 – Dec 2024 (`part3_performance.csv`, figures `part3_performance_*.png`)

| Portfolio | Mean p.a. | Vol p.a. | Skew | Worst month | Sharpe | Utility (A = 5) | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| (i) Market cap Jan 2004 | 8.0 % | 13.8 % | −0.24 | −13.0 % | 0.58 | 3.2 % | −45 % |
| (ii) Market cap, prices 2014 | 9.8 % | 13.5 % | −0.24 | −12.2 % | 0.73 | 5.3 % | −41 % |
| (iii) EW first 15 names | 11.3 % | 18.4 % | −0.10 | −16.5 % | 0.62 | 2.9 % | −55 % |
| (iv) EW last 15 names | 9.9 % | 11.9 % | −0.17 | −9.2 % | 0.83 | 6.3 % | −37 % |
| (v) Tangency | **31.8 %** | 19.0 % | +0.09 | −11.4 % | **1.67** | **22.7 %** | −26 % |
| (vi) Minimum variance | 9.8 % | **10.5 %** | −0.16 | −8.5 % | 0.94 | 7.0 % | **−15 %** |
| (vii) Inverse volatility | 9.6 % | 13.5 % | −0.21 | −12.2 % | 0.71 | 5.0 % | −42 % |
| (viii) Most diversified | 13.9 % | 11.7 % | +0.25 | −8.5 % | 1.19 | 10.5 % | −29 % |
| (ix) Equal group budget | 10.9 % | 15.3 % | −0.13 | −13.6 % | 0.71 | 5.1 % | −49 % |
| (x) Equal risk contribution | 10.4 % | 12.6 % | −0.07 | −10.1 % | 0.83 | 6.5 % | −39 % |
| **DJIA index** (Aug 2004 – Dec 2024) | 8.1 % | 14.5 % | −0.40 | −14.1 % | 0.56 | 2.9 % | −49 % |

Utility = mean − ½·5·σ² (annualised). Skewness is on monthly returns.

**Observations.**
* The **cap‑weighted portfolio (i) tracks the DJIA closely** (8.0 % vs 8.1 %
  p.a., correlation 0.95, tracking error 4.7 % p.a.; see
  `part3_replication_check.csv`). Inverse‑volatility (vii, correlation 0.98,
  TE 3.2 %) and the equal‑group‑budget portfolio (ix, TE 3.9 %) replicate the
  price‑weighted index even more closely, because the DJIA itself is close to
  an equal‑weighted portfolio of its 30 members. The tangency portfolio has a
  correlation of only 0.37 with the index – it is a different investment. Its weakness is the 2004 weighting: it holds 10 % MSFT and
  9.5 % PFE but only 0.3 % AAPL and 0.1 % CRM, and monthly rebalancing keeps
  selling the winners. Portfolio (ii), which "knows" the 2014 prices, already
  gains 1.8 % p.a. from that.
* **Risk‑based portfolios beat cap weighting out of sample as well:** minimum
  variance (vi) has the lowest volatility (10.5 %) and by far the smallest
  drawdown (−15 % in 2008, versus −49 % for the DJIA), inverse‑vol (vii) and
  ERC (x) sit between EW and min‑var. This is the well‑documented
  "low‑volatility effect" plus the benefit of avoiding the 2008 concentration in
  financials (BAC, AXP, GS, JPM get small weights in all risk‑based portfolios).
* **The tangency portfolio (v) looks spectacular (31.8 % p.a., Sharpe 1.67) but
  is an illusion.** Its first 10 years are in‑sample: it went long the stocks
  that had performed best (AAPL, DIS, MCD, JNJ) and short the losers (DD, INTC,
  KO, PG, BAC) *of exactly that period*. The figure shows the return
  concentrated before mid‑2014; afterwards it still does well (AAPL, CRM
  continued), but with 5× gross leverage and a −40 % short in a single stock it
  is not a portfolio anyone would have held. The most diversified portfolio
  (viii) also benefits from hindsight (its 20 % WMT / 14 % CRM / 9 % AAPL tilt)
  but only through the covariance matrix.
* **Alphabet portfolios** show how much single‑stock luck matters: the first 15
  names (with AAPL, but also BAC, HPQ, GS, DD, BA) are more volatile and had a
  55 % drawdown; the last 15 (MSFT, CRM, UNH, NKE, HD…) delivered a much better
  Sharpe ratio (0.83 vs 0.62). Neither is a sensible design.
* The **equal group budget (ix)** effectively over‑weights Finance and
  Energy/Transport (5 % per stock) relative to Manufacturing (2 %); it is an
  equal‑weight portfolio with a sector bet, and in this sample the sector bet
  cost volatility (15.3 %) without extra return.
* **Skewness** is negative for almost all portfolios (crash months of 2008
  and March 2020), least negative for the risk‑based and most diversified
  portfolios.

**Which portfolio would I choose on 1 January 2004?** Only portfolios that use
no future information are admissible: (i), (iii), (iv), (ix) or the DJIA. Among
those, the choice based on ex‑ante reasoning (not on the realised table) would
be a **diversified, estimation‑free portfolio over all 30 stocks**, i.e. the
cap‑weighted portfolio (i)/DJIA or – given the evidence on the 1/N and
low‑volatility effects available already in 2004 – an equal‑weighted or
inverse‑volatility portfolio using *pre‑2004* data. With a risk aversion of 5,
the utility column ranks (iv) > (ix) > (i) ≈ (iii) ≈ DJIA among the admissible
ones, but (iv) is an alphabetical accident. If one insisted on picking from the
table: (i) as the honest replication of the index, or (ix)/EW as the "naive
diversification" alternative. The high numbers of (v), (viii) and (vi) are not
available to a January‑2004 investor and should not drive the decision –
they illustrate look‑ahead bias, not investment skill.

---

## Part 4 – Retirement Management (ETF portfolios)

**Data.** Monthly ETF prices Jan 2000 – Jun 2026; returns from price changes.
The ETFs have different inception dates (SPY 2000, TLT/LQD 2002, AGG 2003,
EEM 2003, FEZ 2004, PSP 2006, IGF 2007, JNK 2008, QAI 2009, USMV 2011,
MTUM/SIZE/VLUE 2013, USRT 2016). A portfolio can only start when all its ETFs
exist, so we report (a) each portfolio over its own longest history and (b) all
seven over the **common period Dec 2016 – Jun 2026 (115 months)**, which is the
only fair comparison. Fixed weights, monthly rebalancing, rf = 1 % for the Sharpe
ratio. (`part4_performance_own_history.csv`, `part4_performance_common_period.csv`)

### (b) Common period Dec 2016 – Jun 2026

| Portfolio | Mean p.a. | CAGR | Vol p.a. | Sharpe (rf 1 %) | Max DD | Corr. with (i) |
|---|---:|---:|---:|---:|---:|---:|
| (i) 60/40 SPY / TLT (baseline) | 7.4 % | 6.9 % | 11.7 % | 0.55 | −27.1 % | 1.00 |
| (ii) IWM/FEZ/EEM (30/15/15) / TLT | 5.0 % | 4.3 % | 12.7 % | 0.32 | −32.2 % | 0.93 |
| (iii) SPY / AGG‑JNK‑LQD | 8.1 % | 7.8 % | 11.4 % | 0.63 | −22.5 % | 0.94 |
| (iv) 4 equity / 4 bond ETFs | 6.0 % | 5.5 % | 11.9 % | 0.43 | −26.8 % | 0.91 |
| (v) MTUM/SIZE/VLUE/USMV / TLT | 6.6 % | 6.1 % | 11.6 % | 0.48 | −29.1 % | 0.97 |
| (vi) SPY / USRT‑QAI‑IGF‑PSP | **10.8 %** | **10.4 %** | 12.9 % | **0.76** | **−22.0 %** | 0.89 |
| (vii) Equal‑weight 16 ETFs | 7.0 % | 6.6 % | 11.3 % | 0.53 | −23.4 % | 0.90 |

### (a) Own history (for information)

| Portfolio | Start | Months | CAGR | Vol | Sharpe | Max DD |
|---|---|---:|---:|---:|---:|---:|
| (i) 60/40 SPY / TLT | 2002‑08 | 287 | 6.1 % | 9.9 % | 0.54 | −31 % |
| (ii) global equity / TLT | 2004‑08 | 263 | 4.9 % | 11.8 % | 0.38 | −35 % |
| (iii) SPY / bond mix | 2008‑02 | 221 | 5.6 % | 11.5 % | 0.45 | −36 % |
| (iv) 4 equity / 4 bond | 2008‑02 | 221 | 3.5 % | 12.6 % | 0.26 | −38 % |
| (v) factor ETFs / TLT | 2013‑05 | 158 | 6.0 % | 10.6 % | 0.51 | −29 % |

(figures: `part4_cumulative_performance.png`, `part4_risk_return.png`)

**Discussion – why do the portfolios differ from the 60/40 baseline?**
1. **Equity choice dominates.** All portfolios are ≈ 0.9 correlated with the
   baseline; what separates them is *which* equity they hold. In 2017–2026 the
   S&P 500 (large‑cap US, tech‑heavy) was the best equity market in the world.
   Replacing it with small caps (IWM), Europe (FEZ) and emerging markets (EEM) –
   portfolio (ii) – cost **2.6 % p.a.** and produced the deepest drawdown
   (−32 %): small caps and EM suffered more in 2020 and 2022 and did not
   participate in the AI‑driven mega‑cap rally. Portfolio (iv), which dilutes
   SPY to 15 %, loses for the same reason. Geographic diversification is sound
   ex ante, but it was penalised ex post in this specific decade.
2. **Long‑term Treasuries were a poor "safe" asset in this period.** TLT has
   duration ≈ 17 years; it lost 33 % in 2022 alone and 51 % from its 2020 peak,
   and it fell *together with* equities. Replacing TLT with a mix of aggregate bonds,
   high yield and investment‑grade credit (iii) shortened duration and added
   credit spread carry: higher return (+0.8 % p.a.), lower volatility, smaller
   drawdown and a better Sharpe ratio (0.63 vs 0.55). Before 2022 the opposite
   held (TLT was the best crash hedge in 2008 and 2020), which is why the
   baseline looks better over its own long history.
3. **Factor ETFs (v)** did not beat plain SPY: the four factors partly cancel
   (Value and Size lagged badly 2017–2020, Momentum and Min‑Vol did fine) and the
   equal mix ends up as a slightly worse S&P 500 (−0.9 % p.a., corr. 0.97).
   Factor premia are long‑horizon phenomena; a 10‑year window is short.
4. **Alternatives (vi)** delivered the best result (10.4 % CAGR, Sharpe 0.76,
   smallest drawdown) – but not because REITs, hedge‑fund replication, or
   infrastructure were great: it is because they *replaced TLT* in the 40 %
   sleeve. Listed private equity (PSP) and infrastructure (IGF) are equity‑like
   (the portfolio is effectively ≈ 90 % equity risk), so (vi) simply took more
   equity risk in a strong equity decade and avoided the 2022 bond crash. Its
   volatility (12.9 %) is the highest of the seven; for a client near retirement
   this is not a like‑for‑like improvement.
5. **1/N over 16 ETFs (vii)** is roughly a 50/25/25 equity/bond/alternative
   mix; it lands close to the baseline in return with the lowest volatility of
   all (11.3 %) – naive diversification "works" in the sense of DeMiguel et al.
   (2009), without any estimation.

**Recommendation for the client.** Keep a 60/40 structure, but (a) diversify
the bond sleeve across duration and credit (portfolio iii) rather than holding
only 20+‑year Treasuries, and (b) be aware that the outperformance of
alternatives in (vi) is mainly hidden equity beta. Geographic equity
diversification (ii, iv) should not be abandoned on the basis of one decade in
which the US happened to win. The 1/N portfolio (vii) is a reasonable
low‑maintenance default.

---

## Part 5 – Windfall Investment

### 5.1 Policy portfolios: €100,000, 30‑year buy‑and‑hold, no rebalancing

Equity = Fama‑French market return (Mkt−RF + RF), bonds = 1‑month T‑bill.
Windfall received at the end of any month from **Jan 1927 to Apr 1996 (832 start
dates)**, each followed by exactly 360 months of data (to Apr 2026). Without
rebalancing the equity share drifts, so each path is computed as
`w·Π(1+r_mkt) + (1−w)·Π(1+r_f)` and the monthly portfolio returns are taken from
that path. Sharpe ratio uses a constant rf = 1 % as prescribed.
(`part5_policy_portfolios_summary.csv`, figures `part5_balance_by_start.png`,
`part5_metrics_by_start.png`, `part5_risk_return_tradeoff.png`)

| Equity/Bonds | Avg. balance after 30 y | Median | Worst | Best | Avg. return p.a. | Avg. vol p.a. | Avg. Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0/100 | €387 k | €315 k | €132 k | €714 k | 4.1 % | 0.7 % | 4.06 |
| 10/90 | €594 k | €535 k | €206 k | €1.20 m | 5.9 % | 4.0 % | 1.38 |
| 20/80 | €802 k | €766 k | €280 k | €1.70 m | 7.0 % | 6.3 % | 1.01 |
| 30/70 | €1.01 m | €966 k | €353 k | €2.20 m | 7.8 % | 8.0 % | 0.88 |
| 40/60 | €1.22 m | €1.14 m | €427 k | €2.71 m | 8.5 % | 9.4 % | 0.81 |
| 50/50 | €1.42 m | €1.29 m | €501 k | €3.21 m | 9.1 % | 10.6 % | 0.77 |
| 60/40 | €1.63 m | €1.46 m | €574 k | €3.71 m | 9.6 % | 11.7 % | 0.74 |
| 70/30 | €1.84 m | €1.65 m | €648 k | €4.22 m | 10.0 % | 12.8 % | 0.71 |
| 80/20 | €2.05 m | €1.83 m | €721 k | €4.72 m | 10.4 % | 13.8 % | 0.69 |
| 90/10 | €2.25 m | €2.00 m | €795 k | €5.22 m | 10.7 % | 14.8 % | 0.67 |
| 100/0 | €2.46 m | €2.18 m | €868 k | €5.77 m | 11.1 % | 15.8 % | 0.65 |

**Trade‑offs.**
* **Return rises almost linearly with the equity share, risk rises faster at
  the low end and slower at the high end.** Going from 0 % to 10 % equity adds
  1.8 % p.a. return for 3.3 pp of volatility; going from 90 % to 100 % adds only
  0.3 % p.a. for 1 pp of volatility. The average Sharpe ratio therefore falls
  monotonically with the equity share (the 0/100 value of 4 is an artefact of
  measuring the actual T‑bill return against a constant 1 % rate; it is not a
  meaningful risk‑adjusted return).
* **Over 30 years the equity risk is mostly upside risk.** No start date in
  832 lost money in nominal terms, even for 100 % equity: the worst 30‑year
  outcome for 100/0 (a windfall received in Aug 1929, just before the crash) is
  still €868 k, i.e. 7.5 % p.a., higher than the *best* outcome of the pure
  bond portfolio (€714 k). The balance plot shows that the ranking of the
  policies never changes across start dates – more equity always ended higher
  after 30 years – but the dispersion is huge: 100/0 ranges from €0.9 m to
  €5.8 m depending on the start month.
* **Buy‑and‑hold drift** makes the "60/40" label misleading: after 30 years of
  equity outperformance a 60/40 portfolio ends at ≈ 90 % equity (82–97 %
  across start dates), so its
  realised volatility (11.7 %) is higher than that of a rebalanced 60/40, and
  risk is back‑loaded to the end of the horizon.
* **Time variation.** Annualised returns for all equity‑containing policies
  peak for windfalls received in 1932–33, 1942 and 1970–75 (buying after
  crashes) and trough for 1929, 1937, 1946 and 1965–68 (buying at peaks).
  Bond‑only outcomes depend entirely on the interest‑rate regime: dreadful for
  starts in the 1930s–40s (near‑zero rates, then inflation), good for starts in
  the 1960s–70s. The bond and equity "good start dates" do not coincide, which
  is the diversification argument for holding some of each.

**Choice today.** For a genuine 30‑year horizon with no intermediate
withdrawals, the historical evidence points to a **high equity share
(80–100 %)**: the long‑horizon downside of equities is smaller than the
long‑horizon downside of cash (real‑value erosion) and the expected terminal
wealth is 5–6× that of bonds. A pragmatic answer is **80/20 or 90/10**: it keeps
≥ 95 % of the equity upside (€2.05–2.25 m vs €2.46 m on average) while cutting
the average volatility by 1–2 pp and giving a buffer that avoids forced selling
in a crash, which matters if the "no withdrawals" assumption does not hold in
practice. Today's starting valuations are high, which historically has been
associated with the lower part of the return range, but not with losses over
30 years. A very risk‑averse investor could choose 60/40; anything below 40 %
equity sacrifices too much terminal wealth for a horizon this long.

### 5.2 Market timing (`part5_market_timing.csv`, figure `part5_market_timing.png`)

Sample Jan 1927 – Apr 2026, 1192 months. Passive 100 % equity turns €100,000
into **€1.68 billion** (11.1 % p.a.). Being 100 % in T‑bills throughout gives
€2.5 m. Switching into equities *only* in the k best months of the entire
history (perfect foresight, no costs):

| k best months | Final balance | k best months | Final balance |
|---:|---:|---:|---:|
| 1 (Apr 1933, +38.9 %) | €3.4 m | 25 | €101 m |
| 2 | €4.7 m | 30 | €162 m |
| 5 | €9.5 m | 40 | €388 m |
| 10 | €19.1 m | 50 | €858 m |
| 20 | €60.5 m | **59** | **€1.68 bn > passive** |

**An investor would have to predict perfectly the 59 best months out of 1192
(≈ 5 % of all months, or about one month every 20 months) to beat buy‑and‑hold.**
Even 50 perfectly timed months leave the timer with half of the passive
investor's wealth. The equity premium is not earned in a few "special" months
that a skilled forecaster could pick; it accrues steadily and the biggest months
cluster in the depths of crises (1932–33, 1938, 1974–75, 2009, 2020) – exactly
when a timer is least likely to be invested. Combined with the transaction
costs and taxes of switching, the evidence strongly favours the passive policy
of 5.1 over any attempt to time the market.

---

## Data notes and assumptions (summary)

| Item | Handling |
|---|---|
| Units | All returns converted to decimals in `data.py`; factors and world returns are in percent in the files, hedge‑fund returns are text with "%" |
| Hedge funds vs. factors | Common sample Jan 2005 – Apr 2026 (factor file ends Apr 2026, HF file May 2026) |
| Excess returns | `r − RF` with the monthly T‑bill rate from `HW_Factors.csv` |
| Part 3 "Jan 2024" | Treated as Jan 2004 (file has "Shares Jan 2004"; sample starts Jan 2004) |
| Part 3 splits | AAPL (Feb 2005), UNH (May 2005), CAT (Jul 2005) 2:1 splits corrected for returns; raw prices kept for market caps; HPQ/HPE spin‑off Nov 2015 left unadjusted |
| Part 3 DJIA | Index column shifted by 6 months in the file; corrected in `data.py` (true DJIA Jul 2004 – Dec 2024) |
| Part 3 / 4 dividends | Stock, ETF and DJIA series are price‑only; returns exclude dividends and coupons, which understates bond ETFs (AGG, LQD, TLT) and REITs most |
| Part 3 optimisations | Tangency and min‑var closed form (shorts allowed); MDP and ERC long‑only via numerical optimisation (SLSQP) |
| Part 4 | Portfolio starts when all constituents have returns; comparison on common window Dec 2016 – Jun 2026 |
| Part 5 | 832 start months (Jan 1927 – Apr 1996), 360‑month horizon, buy‑and‑hold; Sharpe with constant rf = 1 % as prescribed |
