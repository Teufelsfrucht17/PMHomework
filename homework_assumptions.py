"""Source-based qualifications printed by each homework task.

Reviewed against all six pages of PM___Homework2026.pdf and its linked
clarification document on 2026-09-29. These notes disclose choices; they do
not claim that incomplete data have been repaired or that proxies are exact.
"""

CLARIFICATION_URL = "https://docs.google.com/document/d/1x6AjdyrmyYMxpA_6SyAKfxTtCe-Fe-Qv7Hl7ttluny8/edit?usp=sharing"

PRICE_POLICY = (
    "PROFESSOR Q4/Q5 CLARIFICATIONS supplied by the user: keep the original "
    "dates and column alignment in HW_Prices.csv and HW_ETFs.csv. Do not "
    "shift DJIA, Salesforce (CRM), or PSP to compensate for suspected offsets. "
    "Calculate simple price returns directly as P_t/P_(t-1)-1. Additional "
    "dividend, coupon, distribution or total-return data are not required. "
    "No external price corrections or added distributions are applied. This "
    "is the required assignment data policy, not an unresolved input requirement."
)
SHARES = (
    "RESOLVED BY PROFESSOR CLARIFICATION supplied by the user: Part 3.1(i)'s "
    "'Jan 2024' is a typo and means January 2004. Use the supplied Jan-2004 "
    "shares and Jan-2004 prices for (i); use the same Jan-2004 shares with "
    "June-2014 prices for (ii). December-2024 shares are not used as proxies. "
    "This resolves the share-date conflict. Q4/Q5 additionally require the "
    "supplied prices and original alignment; historical universe availability "
    "is not documented."
)

NOTES = {
    "P1Q1": [
        "PDF pp.1-2 states that market returns are already excess returns and warns "
        "about units. Both regressand and regressor use percentage points: subtract "
        "RF only from the index return. This is resolved by the PDF, not an arbitrary assumption.",
        "PDF p.1's factor end date conflicts with the supplied file, which reaches "
        "Apr-2026 (also required by Part 5, p.6). Use the actual matched months; "
        "do not extend the regression to unmatched hedge-fund months.",
        "The PDF specifies no standard-error correction: conventional OLS inference "
        "is the disclosed baseline, not a guarantee against heteroskedasticity or "
        "serial dependence. Mutual-fund comparison is qualitative because no mutual-fund data are supplied.",
    ],
    "P1Q2": [
        "PDF p.2 requests the four-factor regression. We use Mkt-RF, SMB, HML and "
        "Mom from the supplied file, with RF subtracted only from hedge-fund returns, "
        "and the same observations as CAPM. Conventional OLS standard errors are an "
        "explicit inference convention, not specified by the PDF.",
        "A four-predictor model has no single bivariate fitted line against the "
        "market. Actual versus fitted excess returns implements the requested scatter "
        "plot without hiding the other factors. Adjusted R-squared adds a complexity "
        "penalty; fit remains in-sample evidence, not a prediction test.",
    ],
    "P1Q3": [
        "PDF p.2 says to use the market RETURN to identify up/down markets; p.1 "
        "says Mkt-RF is excess. Therefore the classification uses Mkt-RF+RF, while "
        "CAPM still uses Mkt-RF. Zero raw-return months belong to neither regime.",
        "Separate intercepts avoid imposing equal conditional alphas. The pooled "
        "interaction test supplements the requested beta comparison and tests the "
        "difference directly. Conventional OLS inference is an assumption; significance "
        "of each beta is not significance of their difference.",
    ],
    "P1Q4": [
        "PDF p.2 asks for AR(1) of hedge-fund returns, not CAPM residuals or excess "
        "returns. Thus RF and factor matching are unnecessary; use the full hedge-fund "
        "file and exact previous-calendar-month lags. The first return has no lag.",
        "Conventional OLS standard errors are a disclosed baseline. Persistence "
        "in reported index returns does not identify its cause or establish smoothing "
        "for each individual fund; the PDF supplies no fund-level pricing evidence.",
    ],
    "P2": [
        "PDF p.3 explicitly sets nonnegative weights in step 3 but calls the same "
        "portfolio long/short in step 4. Follow the explicit long-only formula: "
        "3*(4/15)+14*(1/70)=1; no short position is justified by those weights.",
        "The 11-month compounded signal is t-12 through t-2; t-1 is skipped. "
        "Ties are resolved alphabetically because the PDF supplies no tie rule.",
        "Average annualized return is implemented as 12 times the monthly mean; "
        "CAGR is separately labeled because the PDF does not choose an averaging "
        "convention. Sharpe uses matched RF with sample excess-return SD; annual "
        "volatility and Sharpe use sqrt(12), without a serial-correlation adjustment.",
        "PDF p.1 explicitly says country returns are USD-based and need no exchange "
        "rate adjustment. The computation preserves that common currency.",
    ],
    "P3Q1": [
        SHARES,
        "DATE CONFLICT: PDF p.1 describes Jun-2004 to Dec-2022 prices, but Part 3 "
        "Q3 p.4 specifies Jan-2004 to Dec-2024 and the CSV actually supplies that "
        "price range. Use the actual dates and the specific task's June-2014 "
        "estimation cutoff. There are 125 returns, Feb-2004 through Jun-2014; "
        "the named cutoff governs rather than mechanically halving the row count.",
        "Long-only/full investment are the user's selected baseline. The PDF does "
        "NOT impose short-sale/leverage bounds, so these particular bounds were "
        "not uniquely forced by it. Some constraint specification was needed; "
        "unrestricted optimizations could produce different answers. RF=0 and "
        "costless monthly rebalancing ARE explicit in Part 3 p.3.",
        "Alphabetical higher means last 15 company names and lower means first "
        "15, following the user's convention; labels alone in the PDF are ambiguous. "
        "Item (x)'s '30 stocks groups' is interpreted as 30 STOCK risk budgets, "
        "because it specifies 1/30 each; five industry-group budgets are separately requested in (ix).",
        PRICE_POLICY,
    ],
    "P3Q2": [
        SHARES,
        "PDF p.4 asks which portfolios are comparable; it does not assert a common "
        "feasible inception date. A July-2014 comparison of (v)-(viii),(x) uses "
        "only their pre-July estimates, using the supplied price data and universe. Alphabetical "
        "and group portfolios additionally require classifications known by then. "
        "No historical constituent/classification record is supplied.",
        "Monthly rebalance in PDF p.3 means restoring the Q1 target vectors. "
        "Rolling re-estimation is not specified and would define a different strategy. "
        "Both cap portfolios can join the July-2014 comparison under the corrected Jan-2004 share date, subject to historical universe availability.",
        PRICE_POLICY,
    ],
    "P3Q3": [
        "PDF p.4 requests Jan-2004 to Dec-2024 performance, but the first price is "
        "Jan-2004. Without Dec-2003 prices, a Jan-2004 return is unobservable. The "
        "correct available return sample is Feb-2004 to Dec-2024 (251 months); "
        "an invented January zero would bias statistics. Month labels are treated "
        "as month-end observations, not known prices on January 1.",
        "PDF p.3 specifies costless monthly rebalance and RF=0; p.4 gives gamma=5. "
        "We restore fixed Q1 targets monthly, using monthly mean and sample variance "
        "in U=mean-5*variance/2 with decimal returns. ddof=1 and bias-corrected "
        "sample skewness are explicitly chosen conventions; the PDF does not specify them.",
        SHARES,
        "Full-sample ranking is the requested DESCRIPTIVE calculation, not a claim "
        "of January-2004 investability. June-2014 estimates and P2 prices were "
        "unavailable then. P1 uses January-2004 month-end inputs, not January-1 "
        "execution prices. Alphabetical/group definitions could be preset, "
        "conditional on contemporaneous membership/classifications. The PDF supplies "
        "no pre-2004 estimates or execution prices to select a unique ex-ante optimum.",
        PRICE_POLICY,
    ],
    "P4": [
        "PDF p.2 explicitly warns that ETF availability differs; pp.4-5 specify "
        "the seven weights. A common return sample is needed for a fair comparison "
        "and follows the user's instruction. USRT's Nov-2016 first price implies "
        "Dec-2016 first return; pre-inception prices cannot be filled to invent returns.",
        "PDF p.5's 33.3% bond-sleeve weights are rounded wording. Use exactly 1/3 "
        "of the 40% sleeve for each bond ETF, so total allocation is exactly 100%. "
        "Regional percentages apply within the 60% equity sleeve; all-sixteen equal "
        "weight need not be 60/40. Monthly rebalancing is explicit.",
        "The stated 1% rate is treated as annual effective and converted by "
        "(1.01)^(1/12)-1. The PDF does not specify its monthly conversion or arithmetic "
        "versus geometric annual return; both arithmetic annualization and CAGR are "
        "reported. Sample SD and sqrt(12) annualization are stated conventions.",
        PRICE_POLICY,
    ],
    "P5Q1": [
        "PDF pp.5-6 explicitly says month-END windfall, next 30 years and NO "
        "rebalancing. Thus Jan-1927 starts with Feb-1927's return, and Apr-1996 "
        "ends Apr-2026: exactly 832 starts with 360 subsequent months each.",
        "PDF p.1 identifies the market series as excess; add RF before compounding "
        "equity. The bond proxy earns historical monthly RF, while the separate "
        "constant 1% benchmark in p.6 is used for Sharpe. It is not a long-duration "
        "bond return. The file's Apr-2026 end supports the task despite p.1's stale May-2025 description.",
        "PDF p.6 does not define average annualized return precisely: report "
        "12*monthly arithmetic mean and separate CAGR. Use sample SD, sqrt(12) "
        "annualization and effective monthly 1% conversion, then average each "
        "path's measures equally across the 832 starts, not a Sharpe of averaged inputs.",
        "The euro starting amount has no supplied EUR/USD series. Scaling USD-based "
        "returns by 100,000 is a hypothetical monetary-unit scenario; no currency "
        "conversion or purchasing-power adjustment can be claimed.",
    ],
    "P5Q2": [
        "PDF p.6 specifies Jan-1927 through Apr-2026, so all 1192 monthly returns "
        "are used, including Jan-1927 (unlike Q1's end-month windfall windows). "
        "PDF p.1 establishes Mkt-RF as excess: passive and selected equity months "
        "earn Mkt-RF+RF; unselected months earn the supplied RF.",
        "The user's explicit interpretation of 'best months' is highest TOTAL "
        "market return, with earlier dates breaking ties. This is distinct from "
        "ranking incremental excess returns or market/RF gross-return ratios; "
        "neither alternative silently replaces the specified ranking.",
        "Perfect knowledge of historical best months and zero transaction costs "
        "are the question's hypothetical premise, not evidence of forecasting "
        "ability. Strict outperformance excludes equality at the all-equity endpoint.",
    ],
}


def print_assumptions(task):
    print("\nPDF CROSS-CHECK: REQUIREMENTS, NECESSARY CHOICES AND UNRESOLVED INPUTS")
    print("Source: PM___Homework2026.pdf; page numbers refer to its six printed pages.")
    print("Its linked clarification document was checked on 2026-09-29; it addresses")
    print("submission format only. The subsequent professor correction supplied by the user")
    print("resolves Part 3.1: January 2024 means January 2004; see the Part 3 notes.")
    if task in {"P3Q1", "P3Q2", "P3Q3", "P4"}:
        print("Further professor Q4/Q5 answers require original column alignment and supplied-price returns.")
    for number, note in enumerate(NOTES[task], 1):
        print(f"{number}. {note}")
    print("These qualifications are part of the answer, not claims that missing information was repaired.")


if __name__ == "__main__":
    for task in NOTES:
        print(f"\n=== {task} ===")
        print_assumptions(task)
