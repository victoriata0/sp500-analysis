"""
S&P 500 Long-Run Returns, Inflation, and Valuation Analysis (1871-2026)

Data: Robert Shiller's monthly S&P 500 data set, published at
https://github.com/datasets/s-and-p-500 (data/data.csv)

Questions:
  1. How much did the S&P 500 really return once dividends and inflation are counted?
  2. How deep were the biggest crashes, and how long did recovery take?
  3. Does a high CAPE (price vs. 10-year average earnings) predict weaker
     returns over the next 10 years?

Run:  python sp500_analysis.py
Outputs: charts/*.png and results.txt
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data" / "sp500.csv"
CHARTS = Path(__file__).parent / "charts"
CHARTS.mkdir(exist_ok=True)

lines = []  # collected findings, written to results.txt


def report(text=""):
    print(text)
    lines.append(text)


# 1. Load and clean
df = pd.read_csv(DATA, parse_dates=["Date"]).set_index("Date")
raw_rows = len(df)

# The source fills months it has not updated yet with 0 instead of leaving them
# blank. A price, CPI, or dividend of 0 is impossible, so treat 0 as missing.
cols = ["Dividend", "Earnings", "Consumer Price Index", "Long Interest Rate",
        "Real Price", "Real Dividend", "Real Earnings", "PE10"]
zeros_replaced = int((df[cols] == 0).sum().sum())
df[cols] = df[cols].replace(0, np.nan)

# Keep only months where price, dividend and CPI all exist, so total and
# inflation-adjusted returns are computed on complete data.
clean = df.dropna(subset=["SP500", "Dividend", "Consumer Price Index"]).copy()

report("S&P 500 LONG-RUN ANALYSIS")
report(f"Rows loaded: {raw_rows:,} monthly observations "
       f"({df.index.min():%b %Y} - {df.index.max():%b %Y})")
report(f"Placeholder zeros converted to missing: {zeros_replaced}")
report(f"Complete months used for return math: {len(clean):,} "
       f"({clean.index.min():%b %Y} - {clean.index.max():%b %Y})")
report()


# 2. Total return and inflation-adjusted (real) return
# Dividend is an annual rate, so one month earns 1/12 of it.
price, div, cpi = clean["SP500"], clean["Dividend"], clean["Consumer Price Index"]
clean["nominal_tr"] = (price + div / 12) / price.shift(1) - 1   # monthly total return
clean["inflation"] = cpi / cpi.shift(1) - 1
clean["real_tr"] = (1 + clean["nominal_tr"]) / (1 + clean["inflation"]) - 1
clean = clean.dropna(subset=["real_tr"])

# Growth of $1 invested at the start, dividends reinvested
clean["nominal_index"] = (1 + clean["nominal_tr"]).cumprod()
clean["real_index"] = (1 + clean["real_tr"]).cumprod()
clean["price_only_index"] = price.loc[clean.index] / price.loc[clean.index[0]]

years = len(clean) / 12


def annualized(series):
    return (1 + series).prod() ** (12 / len(series)) - 1


nom_cagr = annualized(clean["nominal_tr"])
real_cagr = annualized(clean["real_tr"])
infl_cagr = annualized(clean["inflation"])
price_cagr = clean["price_only_index"].iloc[-1] ** (1 / years) - 1

report("1) RETURNS")
report(f"   Price-only growth:            {price_cagr:6.2%} per year")
report(f"   Total return (w/ dividends):  {nom_cagr:6.2%} per year")
report(f"   Average inflation:            {infl_cagr:6.2%} per year")
report(f"   Real total return:            {real_cagr:6.2%} per year")
report(f"   $1 invested in {clean.index[0]:%Y} grew to "
       f"${clean['nominal_index'].iloc[-1]:,.0f} nominal, "
       f"a {clean['real_index'].iloc[-1]:,.0f}x gain in purchasing power after inflation")
report(f"   Dividends' share of total return: "
       f"{1 - price_cagr / nom_cagr:.0%}")

# Real total return by decade
decade = (clean.index.year // 10) * 10
by_decade = clean.groupby(decade)["real_tr"].apply(annualized)
months_per_decade = clean.groupby(decade).size()
by_decade = by_decade[months_per_decade >= 60]   # skip partial decades < 5 yrs
report("   Real total return by decade:")
for d, r in by_decade.items():
    report(f"     {d}s  {r:7.2%}")
best, worst = by_decade.idxmax(), by_decade.idxmin()
report(f"   Best decade: {best}s ({by_decade[best]:.1%}/yr); "
       f"worst: {worst}s ({by_decade[worst]:.1%}/yr)")
report()


# 3. Drawdowns: how far below its previous peak was the market?
idx = clean["real_index"]
peak = idx.cummax()
clean["drawdown"] = idx / peak - 1


def crash_stats(start, end):
    """Peak-to-bottom fall within one crash window, measured from that window's
    own peak, plus how many months it took to climb back above that peak."""
    window = idx.loc[start:end]
    trough_date = window.idxmin()
    before = window.loc[:trough_date]
    peak_date = before.idxmax()
    peak_value = before.max()
    fall = window.min() / peak_value - 1
    after = idx.loc[trough_date:]
    recovered = after[after >= peak_value]
    rec_date = recovered.index[0] if len(recovered) else None
    months = ((rec_date.year - peak_date.year) * 12 + rec_date.month - peak_date.month
              if rec_date is not None else None)
    return fall, peak_date, trough_date, rec_date, months


report("2) BIGGEST CRASHES (inflation-adjusted, dividends reinvested)")
crashes = {
    "Great Depression": ("1929-01", "1935-12"),
    "1970s stagflation": ("1973-01", "1982-12"),
    "Dot-com bust": ("2000-01", "2003-12"),
    "2008 financial crisis": ("2007-06", "2009-12"),
    "COVID-19": ("2020-01", "2020-12"),
}
for name, (s, e) in crashes.items():
    dd, pk, tr, rec, m = crash_stats(s, e)
    rec_txt = f"{m // 12} yrs {m % 12} mo to new high" if m is not None else "not yet recovered"
    report(f"   {name:<22} {dd:7.1%}  (peak {pk:%b %Y}, bottom {tr:%b %Y}; {rec_txt})")
report()


# 4. Valuation: does a high CAPE predict lower 10-year returns?
# Forward 10-year annualized real return from each month
clean["fwd10_real"] = (idx.shift(-120) / idx) ** (1 / 10) - 1
val = clean.dropna(subset=["PE10", "fwd10_real"])
corr = val["PE10"].corr(val["fwd10_real"])
slope, intercept = np.polyfit(val["PE10"], val["fwd10_real"], 1)
r2 = corr ** 2

val = val.assign(cape_q=pd.qcut(val["PE10"], 5,
                                labels=["Cheapest 20%", "2nd", "Middle", "4th", "Priciest 20%"]))
quintiles = val.groupby("cape_q", observed=True).agg(
    cape_low=("PE10", "min"), cape_high=("PE10", "max"),
    avg_fwd10=("fwd10_real", "mean"), pct_negative=("fwd10_real", lambda s: (s < 0).mean()))

latest_cape = df["PE10"].dropna()
report("3) VALUATION (CAPE) vs. NEXT-10-YEAR REAL RETURN")
report(f"   Months with a full 10 years of future data: {len(val):,} "
       f"({val.index.min():%Y} - {val.index.max():%Y})")
report(f"   Correlation: {corr:.2f}   R-squared: {r2:.2f}")
report(f"   Each +1 point of CAPE ~ {slope:.2%} lower annual real return over next 10 yrs")
report("   By CAPE quintile:")
for q, row in quintiles.iterrows():
    report(f"     {q:<13} CAPE {row.cape_low:5.1f}-{row.cape_high:5.1f}:  "
           f"avg {row.avg_fwd10:6.2%}/yr, negative {row.pct_negative:5.1%} of the time")
report(f"   Latest CAPE in data: {latest_cape.iloc[-1]:.1f} "
       f"({latest_cape.index[-1]:%b %Y}); historical median {df['PE10'].median():.1f}")
report()


# 5. Charts
# Chart 1: growth of $1, nominal vs. inflation-adjusted (log scale)
plt.figure(figsize=(10, 5))
plt.plot(clean.index, clean["nominal_index"], label="Nominal")
plt.plot(clean.index, clean["real_index"], label="Inflation-adjusted")
plt.yscale("log")
plt.title("Growth of $1 in the S&P 500, dividends reinvested")
plt.xlabel("Year")
plt.ylabel("Value of $1 (log scale)")
plt.legend()
plt.savefig(CHARTS / "1_growth_of_1_dollar.png")
plt.close()

# Chart 2: drawdown from previous peak
plt.figure(figsize=(10, 5))
plt.plot(clean.index, clean["drawdown"] * 100)
plt.title("How far below its last peak the market fell (inflation-adjusted)")
plt.xlabel("Year")
plt.ylabel("Drawdown (%)")
plt.savefig(CHARTS / "2_drawdowns.png")
plt.close()

# Chart 3: CAPE vs. next-10-year real return, with a trend line
plt.figure(figsize=(8, 6))
plt.scatter(val["PE10"], val["fwd10_real"] * 100, s=5, alpha=0.4)
xs = np.linspace(val["PE10"].min(), val["PE10"].max(), 100)
plt.plot(xs, (slope * xs + intercept) * 100, color="red", label=f"Trend (r = {corr:.2f})")
plt.title("CAPE vs. next-10-year real return")
plt.xlabel("CAPE ratio")
plt.ylabel("Annualized real return over next 10 years (%)")
plt.legend()
plt.savefig(CHARTS / "3_cape_vs_future_returns.png")
plt.close()

# Chart 4: average next-10-year return by CAPE quintile
plt.figure(figsize=(8, 5))
plt.bar(quintiles.index.astype(str), quintiles["avg_fwd10"] * 100)
plt.title("Average next-10-year real return, by starting CAPE")
plt.xlabel("CAPE quintile")
plt.ylabel("Annualized real return (%)")
plt.savefig(CHARTS / "4_returns_by_cape_quintile.png")
plt.close()

(Path(__file__).parent / "results.txt").write_text("\n".join(lines) + "\n")
print(f"\nSaved {len(list(CHARTS.glob('*.png')))} charts to {CHARTS}/ and results.txt")
