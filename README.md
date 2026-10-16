# S&P 500 Long-Run Returns, Inflation & Valuation (1871–2026)

A Python analysis of 150+ years of monthly S&P 500 data that asks three questions:

1. **What did the market really return** once dividends and inflation are counted?
2. **How deep were the biggest crashes**, and how long did recovery take?
3. **Does a high valuation (CAPE ratio) predict weaker returns** over the next 10 years?

## Data

Robert Shiller's monthly S&P 500 data set, as published at
[github.com/datasets/s-and-p-500](https://github.com/datasets/s-and-p-500): price, dividends,
earnings, CPI, long-term interest rate and CAPE (PE10) for every month since January 1871.

**Cleaning:** the source fills months it hasn't updated yet with `0` instead of leaving them
blank. Since a price, CPI or dividend of zero is impossible, the script converts those 420
placeholder zeros to missing values. Return math uses only months where price, dividend and
CPI all exist (Jan 1871 – Jun 2023).

## Key findings

- Price-only growth: 4.6% per year
- Total return (dividends reinvested): 9.2% per year
- Real total return (after inflation): 6.9% per year
- Share of total return from dividends: about half
- Best decade (real): 1950s, 16.7% per year. Worst: 2000s, -3.2% per year
- CAPE vs. next-10-year real return: correlation of -0.52
- Cheapest 20% of CAPE readings: next 10 years averaged 10.9% per year, never negative
- Priciest 20% of CAPE readings: next 10 years averaged 3.0% per year, negative 30% of the time

**Biggest crashes** (inflation-adjusted, dividends reinvested, monthly averages):

- Great Depression: fell 77% (Sep 1929 to Jun 1932), 7 years 2 months to a new high
- 1970s stagflation: fell 50% (Jan 1973 to Dec 1974), 12 years to a new high
- Dot-com bust: fell 45% (Aug 2000 to Feb 2003), 12 years 9 months to a new high
- 2008 financial crisis: fell 50% (Oct 2007 to Mar 2009), 5 years 5 months to a new high
- COVID-19: fell 19% (Jan 2020 to Mar 2020), 7 months to a new high

Monthly averages smooth out single-day lows, so these falls are smaller than the
daily-close figures often quoted in the news.

## Charts

![Growth of $1](charts/1_growth_of_1_dollar.png)
![Drawdowns](charts/2_drawdowns.png)
![CAPE vs returns](charts/3_cape_vs_future_returns.png)
![Returns by CAPE quintile](charts/4_returns_by_cape_quintile.png)

## Method

- **Monthly total return** = (price this month + one month of dividends) ÷ price last month − 1.
  Dividends in the source are annual rates, so one month earns 1/12.
- **Real return** removes that month's CPI inflation: (1 + total return) ÷ (1 + inflation) − 1.
- **Annualized return** compounds the monthly returns and converts to a yearly rate.
- **Drawdown** = how far the inflation-adjusted index sits below its earlier peak.
- **Forward 10-year return** = the annualized real return from each month to 120 months later,
  compared against that month's CAPE with a correlation and a linear trend line, then grouped
  into five equal-size CAPE buckets (quintiles).

**Limitations:** correlation is not causation, and CAPE explains only about a quarter of the
variation in 10-year returns (R² = 0.27), so it is a long-run signal, not a timing tool.

## Run it

```bash
pip install pandas numpy matplotlib
python sp500_analysis.py
```

Outputs `results.txt` (all numbers) and four charts in `charts/`.
