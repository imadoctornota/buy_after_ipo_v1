# Buy the IPO or wait for the dip?

Code for the analysis behind my video on what happened after the IPOs of companies that later became some of the largest stocks in the market.

The question here is narrower than "should you buy IPOs?" I started with companies in the current S&P 500 and Nasdaq-100 and looked backward. That means the sample is survivor-biased on purpose: these are companies we already know became large enough to make a major index.

What I wanted to know was whether an investor who had somehow picked one of those future winners needed to buy immediately, or whether a lower price usually appeared later.

## What the script tests

`ipo_analysis.py` pulls the current S&P 500 and Nasdaq-100 membership, downloads daily Yahoo Finance prices, applies the listing exclusions in `data/listing_overrides.csv`, and then measures returns from the first trading-day close.

The main comparisons are:

- how often the stock traded below its first-day close within 1, 3, 6, and 12 months;
- whether simply waiting a fixed amount of time actually produced a lower entry price;
- raw returns and returns relative to the S&P 500 from one day through ten years;
- the same "did it dip?" test starting from an ordinary trading day two years later;
- one-year results by IPO decade;
- whether the first-day open-to-close move predicted later returns.

The S&P 500 index (`^GSPC`) is the main benchmark rather than SPY because SPY only begins in 1993.

For the horizon tests, the script reports medians, bootstrap 95% confidence intervals, exact two-sided sign tests, and Benjamini-Hochberg adjusted q-values. The first-day-move correlations use Spearman correlation with the same multiple-testing correction.

## Sample construction

This part matters more than usual.

A first Yahoo Finance date is not automatically an IPO date. Some current index members are spin-offs, direct listings, relistings, later share classes, or products of mergers and corporate reorganizations. Yahoo also has a legacy boundary where a number of old price histories simply begin in 1962.

The script therefore does two things before the return analysis:

1. drops histories that begin at the legacy 1962 boundary;
2. applies the explicit exclusions in `data/listing_overrides.csv`.

The video used a hand-reviewed set of **346 traditional IPOs**. Because index membership changes and some listing classifications involve judgment, every run writes `outputs/cohort_audit.csv`. I would look at that file before treating a rerun months or years later as an exact reproduction of the video sample.

The rounded figures quoted in the video are also saved in `data/video_summary.csv` so the published claims are separate from whatever a future live refresh produces.

## Run it

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python ipo_analysis.py
```

The first run takes a while because it downloads full price histories for several hundred securities. Those files are cached locally. Use `--refresh` if you want to redownload them.

```bash
python ipo_analysis.py --refresh
```

Results are written to `outputs/`, including the cohort audit, company-level returns, dip frequencies, the two-years-later placebo, decade results, first-day-move correlations, inference tables, and the figures used to inspect the results.

## The result I found most interesting

Roughly 90% of the eventual winners traded below their first-day closing price within a year. That sounds like a strong argument for waiting.

But it was not that simple. If the rule was just "wait six months and buy then," a lower price was available only about 35% of the time. By one year, the median stock in the video sample was up roughly 16%, about 9% ahead of the S&P 500 over the same dates.

The extra check changed how I interpreted the first result: when I repeated the dip test from an ordinary trading day two years after the IPO, the same stocks dipped below the new starting price at about the same rate. In this survivor sample, the apparent "post-IPO dip" looked much more like ordinary stock volatility than a special IPO effect.

The effect was also smaller among more recent IPOs. For companies that went public from 2010 onward, the one-year median advantage over the S&P 500 was only about 3% in the video analysis.

## Caveats

This is not a test of the average IPO. Failed companies, delistings, acquisitions, and firms that never made the S&P 500 or Nasdaq-100 are missing by design. The analysis also ignores taxes, transaction costs, and the fact that an investor in real time does not know which IPO will become a future index winner.

That survivor bias is the main reason I would not interpret these results as an IPO trading strategy. A separate analysis of the full IPO universe is needed for that question.

Data: Yahoo Finance daily prices; current index membership from public Wikipedia tables. Historical analysis only, not financial advice.
