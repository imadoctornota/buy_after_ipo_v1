"""Analysis used for the IPO video.

The question is deliberately narrower than "are IPOs good investments?". The
sample starts with companies that are in the S&P 500 or Nasdaq-100 today and
looks backward to their public-market debut. That makes this a survivor sample
by construction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from statsmodels.stats.multitest import multipletests
import yfinance as yf


SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
BENCHMARK = "^GSPC"
LEGACY_YAHOO_BOUNDARY = pd.Timestamp("1962-01-05")
VIDEO_COHORT_N = 346

HORIZONS = {
    "1d": pd.DateOffset(days=1),
    "1w": pd.DateOffset(weeks=1),
    "1m": pd.DateOffset(months=1),
    "3m": pd.DateOffset(months=3),
    "6m": pd.DateOffset(months=6),
    "1y": pd.DateOffset(years=1),
    "2y": pd.DateOffset(years=2),
    "3y": pd.DateOffset(years=3),
    "5y": pd.DateOffset(years=5),
    "10y": pd.DateOffset(years=10),
}

DIP_WINDOWS = {
    "1m": pd.DateOffset(months=1),
    "3m": pd.DateOffset(months=3),
    "6m": pd.DateOffset(months=6),
    "1y": pd.DateOffset(years=1),
}

CORR_HORIZONS = ["1w", "1m", "3m", "6m", "1y", "3y", "5y"]


@dataclass
class Listing:
    ticker: str
    company: str
    index_sp500: bool
    index_nasdaq100: bool
    first_date: pd.Timestamp
    first_open: float
    first_close: float
    include: bool
    reason: str


def clean_ticker(x: str) -> str:
    return str(x).strip().replace(".", "-")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(str(v) for v in col if str(v) != "nan").strip() for col in df.columns]
    return df


def fetch_index_members() -> pd.DataFrame:
    sp_tables = [_flatten_columns(x.copy()) for x in pd.read_html(SP500_URL)]
    sp = next(x for x in sp_tables if "Symbol" in x.columns and "Security" in x.columns)
    sp = sp[["Symbol", "Security"]].rename(columns={"Symbol": "ticker", "Security": "company"})
    sp["ticker"] = sp["ticker"].map(clean_ticker)
    sp["index_sp500"] = True

    nq_tables = [_flatten_columns(x.copy()) for x in pd.read_html(NASDAQ100_URL)]
    nq = None
    for x in nq_tables:
        ticker_col = next((c for c in x.columns if re.fullmatch(r"(?i)(ticker|ticker symbol|symbol)", c.strip())), None)
        company_col = next((c for c in x.columns if re.search(r"(?i)(company|security)", c)), None)
        if ticker_col and company_col and 80 <= len(x) <= 120:
            nq = x[[ticker_col, company_col]].rename(columns={ticker_col: "ticker", company_col: "company"})
            break
    if nq is None:
        raise RuntimeError("Could not find the Nasdaq-100 constituent table on Wikipedia.")

    nq["ticker"] = nq["ticker"].map(clean_ticker)
    nq["index_nasdaq100"] = True

    out = sp.merge(nq, on="ticker", how="outer", suffixes=("_sp", "_nq"))
    out["company"] = out.get("company_sp").combine_first(out.get("company_nq"))
    out["index_sp500"] = out["index_sp500"].fillna(False).astype(bool)
    out["index_nasdaq100"] = out["index_nasdaq100"].fillna(False).astype(bool)
    return out[["ticker", "company", "index_sp500", "index_nasdaq100"]].sort_values("ticker").reset_index(drop=True)


def load_overrides(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "include", "ipo_date", "reason"])
    d = pd.read_csv(path, dtype={"ticker": str})
    d["ticker"] = d["ticker"].map(clean_ticker)
    d["include"] = pd.to_numeric(d["include"], errors="coerce")
    d["ipo_date"] = pd.to_datetime(d["ipo_date"], errors="coerce")
    return d.set_index("ticker", drop=False)


def download_history(ticker: str, cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker.replace('^', 'INDEX_')}.csv"

    if cache_path.exists() and not refresh:
        d = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        d.index = pd.DatetimeIndex(d.index).tz_localize(None)
        return d

    d = yf.Ticker(ticker).history(period="max", auto_adjust=True, actions=False)
    if d.empty:
        return d
    d.index = pd.DatetimeIndex(d.index).tz_localize(None)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in d.columns]
    d = d[keep].dropna(subset=["Close"]).copy()
    d.to_csv(cache_path, index_label="Date")
    time.sleep(0.05)
    return d


def first_row_on_or_after(d: pd.DataFrame, date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Series] | None:
    x = d.loc[d.index >= pd.Timestamp(date)]
    if x.empty:
        return None
    return x.index[0], x.iloc[0]


def close_on_or_after(d: pd.DataFrame, date: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    z = first_row_on_or_after(d, date)
    if z is None:
        return None
    return z[0], float(z[1]["Close"])


def make_listing(row: pd.Series, hist: pd.DataFrame, overrides: pd.DataFrame) -> Listing | None:
    if hist.empty or "Open" not in hist.columns:
        return None

    first = hist.dropna(subset=["Open", "Close"]).iloc[0]
    first_date = hist.dropna(subset=["Open", "Close"]).index[0]
    include = True
    reason = "first observed Yahoo trading date"

    if first_date <= LEGACY_YAHOO_BOUNDARY:
        include = False
        reason = "Yahoo history begins at the legacy 1962 boundary"

    if row.ticker in overrides.index:
        o = overrides.loc[row.ticker]
        if isinstance(o, pd.DataFrame):
            o = o.iloc[-1]
        if pd.notna(o.get("include")):
            include = bool(int(o["include"]))
        if pd.notna(o.get("reason")) and str(o["reason"]).strip():
            reason = str(o["reason"]).strip()
        if pd.notna(o.get("ipo_date")):
            z = first_row_on_or_after(hist, pd.Timestamp(o["ipo_date"]))
            if z is not None:
                first_date, first = z

    return Listing(
        ticker=row.ticker,
        company=row.company,
        index_sp500=bool(row.index_sp500),
        index_nasdaq100=bool(row.index_nasdaq100),
        first_date=pd.Timestamp(first_date),
        first_open=float(first["Open"]),
        first_close=float(first["Close"]),
        include=include,
        reason=reason,
    )


def build_cohort(members: pd.DataFrame, cache_dir: Path, overrides: pd.DataFrame, refresh: bool) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    listings: list[Listing] = []
    histories: dict[str, pd.DataFrame] = {}

    for i, row in members.iterrows():
        print(f"[{i + 1:>3}/{len(members)}] {row.ticker}")
        hist = download_history(row.ticker, cache_dir, refresh=refresh)
        if hist.empty:
            continue
        histories[row.ticker] = hist
        item = make_listing(row, hist, overrides)
        if item is not None:
            listings.append(item)

    audit = pd.DataFrame([x.__dict__ for x in listings])
    if audit.empty:
        raise RuntimeError("No usable price histories were downloaded.")
    audit = audit.sort_values(["include", "first_date", "ticker"], ascending=[False, True, True]).reset_index(drop=True)
    return audit, histories


def compute_forward_returns(audit: pd.DataFrame, histories: dict[str, pd.DataFrame], benchmark: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for x in audit.loc[audit.include].itertuples(index=False):
        hist = histories[x.ticker]
        b0 = close_on_or_after(benchmark, x.first_date)
        if b0 is None:
            continue
        _, bench0 = b0

        row = {
            "ticker": x.ticker,
            "company": x.company,
            "ipo_date": x.first_date,
            "first_open": x.first_open,
            "first_close": x.first_close,
            "first_day_move": x.first_close / x.first_open - 1,
        }

        for label, offset in HORIZONS.items():
            target = x.first_date + offset
            s1 = close_on_or_after(hist, target)
            b1 = close_on_or_after(benchmark, target)
            if s1 is None or b1 is None:
                row[f"return_{label}"] = np.nan
                row[f"sp500_return_{label}"] = np.nan
                row[f"excess_{label}"] = np.nan
                continue
            _, stock1 = s1
            _, bench1 = b1
            sr = stock1 / x.first_close - 1
            br = bench1 / bench0 - 1
            row[f"return_{label}"] = sr
            row[f"sp500_return_{label}"] = br
            row[f"excess_{label}"] = sr - br

        rows.append(row)
    return pd.DataFrame(rows)


def compute_dips(audit: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for x in audit.loc[audit.include].itertuples(index=False):
        hist = histories[x.ticker]
        row = {"ticker": x.ticker, "ipo_date": x.first_date}
        for label, offset in DIP_WINDOWS.items():
            end = x.first_date + offset
            window = hist.loc[(hist.index > x.first_date) & (hist.index <= end), "Close"]
            if window.empty:
                row[f"ever_below_{label}"] = np.nan
                row[f"worst_return_{label}"] = np.nan
            else:
                row[f"ever_below_{label}"] = bool(window.min() < x.first_close)
                row[f"worst_return_{label}"] = float(window.min() / x.first_close - 1)

            fixed = close_on_or_after(hist, end)
            row[f"cheaper_at_{label}"] = np.nan if fixed is None else bool(fixed[1] < x.first_close)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_placebo_dips(audit: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for x in audit.loc[audit.include].itertuples(index=False):
        hist = histories[x.ticker]
        anchor = close_on_or_after(hist, x.first_date + pd.DateOffset(years=2))
        if anchor is None:
            continue
        anchor_date, anchor_close = anchor
        row = {"ticker": x.ticker, "anchor_date": anchor_date}
        for label, offset in DIP_WINDOWS.items():
            end = anchor_date + offset
            window = hist.loc[(hist.index > anchor_date) & (hist.index <= end), "Close"]
            row[f"ever_below_{label}"] = np.nan if window.empty else bool(window.min() < anchor_close)
            row[f"worst_return_{label}"] = np.nan if window.empty else float(window.min() / anchor_close - 1)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_median_ci(x: pd.Series, n_boot: int = 5000, seed: int = 2026) -> tuple[float, float]:
    a = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(a) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    chunk = 500
    meds = []
    left = n_boot
    while left:
        n = min(chunk, left)
        samples = rng.choice(a, size=(n, len(a)), replace=True)
        meds.append(np.median(samples, axis=1))
        left -= n
    boot = np.concatenate(meds)
    return tuple(np.quantile(boot, [0.025, 0.975]))


def sign_test_p(x: pd.Series) -> float:
    a = pd.Series(x).dropna().to_numpy(dtype=float)
    a = a[a != 0]
    if len(a) == 0:
        return np.nan
    return float(binomtest(int((a > 0).sum()), len(a), p=0.5, alternative="two-sided").pvalue)


def inference_table(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in HORIZONS:
        for kind, prefix in [("raw", "return_"), ("sp500_relative", "excess_")]:
            x = returns[prefix + label].dropna()
            lo, hi = bootstrap_median_ci(x)
            rows.append({
                "horizon": label,
                "outcome": kind,
                "n": len(x),
                "median": x.median() if len(x) else np.nan,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": sign_test_p(x),
            })
    out = pd.DataFrame(rows)
    mask = out.p_value.notna()
    out["q_value_bh"] = np.nan
    if mask.any():
        out.loc[mask, "q_value_bh"] = multipletests(out.loc[mask, "p_value"], method="fdr_bh")[1]
    out["ci_excludes_zero"] = (out.ci_low > 0) | (out.ci_high < 0)
    return out


def dip_summary(dips: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in DIP_WINDOWS:
        e = dips[f"ever_below_{label}"].dropna().astype(float)
        f = dips[f"cheaper_at_{label}"].dropna().astype(float)
        w = dips[f"worst_return_{label}"].dropna()
        rows.append({
            "window": label,
            "n_ever_below": len(e),
            "fraction_ever_below": e.mean() if len(e) else np.nan,
            "n_fixed_wait": len(f),
            "fraction_cheaper_at_fixed_date": f.mean() if len(f) else np.nan,
            "median_worst_return": w.median() if len(w) else np.nan,
        })
    return pd.DataFrame(rows)


def placebo_summary(placebo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in DIP_WINDOWS:
        x = placebo[f"ever_below_{label}"].dropna().astype(float)
        w = placebo[f"worst_return_{label}"].dropna()
        rows.append({
            "window": label,
            "n": len(x),
            "fraction_ever_below": x.mean() if len(x) else np.nan,
            "median_worst_return": w.median() if len(w) else np.nan,
        })
    return pd.DataFrame(rows)


def correlation_table(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in CORR_HORIZONS:
        for kind, col in [("raw", f"return_{label}"), ("sp500_relative", f"excess_{label}")]:
            d = returns[["first_day_move", col]].dropna()
            if len(d) < 4:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(d.first_day_move, d[col])
            rows.append({"horizon": label, "outcome": kind, "n": len(d), "spearman_rho": rho, "p_value": p})
    out = pd.DataFrame(rows)
    mask = out.p_value.notna()
    out["q_value_bh"] = np.nan
    if mask.any():
        out.loc[mask, "q_value_bh"] = multipletests(out.loc[mask, "p_value"], method="fdr_bh")[1]
    return out


def decade_table(returns: pd.DataFrame) -> pd.DataFrame:
    d = returns.dropna(subset=["ipo_date"]).copy()
    d["ipo_decade"] = (pd.to_datetime(d.ipo_date).dt.year // 10) * 10
    rows = []
    for decade, g in d.groupby("ipo_decade"):
        raw = g["return_1y"].dropna()
        excess = g["excess_1y"].dropna()
        rows.append({
            "ipo_decade": int(decade),
            "n": len(g),
            "n_1y": len(excess),
            "median_return_1y": raw.median() if len(raw) else np.nan,
            "median_excess_1y": excess.median() if len(excess) else np.nan,
        })

    recent = d[pd.to_datetime(d.ipo_date).dt.year >= 2010]
    rows.append({
        "ipo_decade": 2010,
        "n": len(recent),
        "n_1y": recent["excess_1y"].notna().sum(),
        "median_return_1y": recent["return_1y"].median(),
        "median_excess_1y": recent["excess_1y"].median(),
        "group": "2010+ pooled",
    })
    return pd.DataFrame(rows)


def make_plots(inference: pd.DataFrame, dip: pd.DataFrame, placebo: pd.DataFrame, decade: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rel = inference[inference.outcome.eq("sp500_relative")].set_index("horizon").reindex(HORIZONS.keys())
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(rel.index, 100 * rel["median"], marker="o")
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Median return vs. S&P 500 (%)")
    ax.set_xlabel("Time from first-day close")
    ax.set_title("Median benchmark-relative return")
    fig.tight_layout()
    fig.savefig(out_dir / "median_excess_return.png", dpi=180)
    plt.close(fig)

    dd = dip.set_index("window").reindex(DIP_WINDOWS.keys())
    x = np.arange(len(dd))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, 100 * dd.fraction_ever_below, width, label="Ever cheaper during window")
    ax.bar(x + width / 2, 100 * dd.fraction_cheaper_at_fixed_date, width, label="Cheaper at fixed date")
    ax.set_xticks(x, dd.index)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Stocks (%)")
    ax.set_title("A dip was common; catching it on a schedule was not")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "dip_vs_fixed_wait.png", dpi=180)
    plt.close(fig)

    pp = placebo.set_index("window").reindex(DIP_WINDOWS.keys())
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(dd.index, 100 * dd.fraction_ever_below, marker="o", label="After IPO")
    ax.plot(pp.index, 100 * pp.fraction_ever_below, marker="o", label="Starting two years later")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Stocks trading below starting close (%)")
    ax.set_xlabel("Forward window")
    ax.set_title("Is the post-IPO dip unusual?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "ipo_vs_two_year_placebo.png", dpi=180)
    plt.close(fig)

    dec = decade[decade.get("group", pd.Series(index=decade.index, dtype=object)).isna()].dropna(subset=["median_excess_1y"])
    if not dec.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(dec.ipo_decade.astype(str), 100 * dec.median_excess_1y)
        ax.axhline(0, linewidth=1)
        ax.set_ylabel("Median 1-year return vs. S&P 500 (%)")
        ax.set_xlabel("IPO decade")
        ax.set_title("One-year excess return by IPO decade")
        fig.tight_layout()
        fig.savefig(out_dir / "one_year_excess_by_decade.png", dpi=180)
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parent
    cache_dir = root / "cache" / "prices"
    out_dir = root / "outputs"
    overrides = load_overrides(root / "data" / "listing_overrides.csv")

    members = fetch_index_members()
    members.to_csv(out_dir / "index_members_runtime.csv", index=False)
    print(f"Current union: {len(members)} securities")

    audit, histories = build_cohort(members, cache_dir, overrides, refresh=args.refresh)
    audit.to_csv(out_dir / "cohort_audit.csv", index=False)

    cohort_n = int(audit.include.sum())
    print(f"Included after automatic + manual listing filters: {cohort_n}")
    if cohort_n != VIDEO_COHORT_N:
        warnings.warn(
            f"This live run has {cohort_n} included securities; the video used {VIDEO_COHORT_N}. "
            "Index membership and the manual listing audit are time-sensitive. Review outputs/cohort_audit.csv "
            "before comparing exact values with the video."
        )

    benchmark = download_history(BENCHMARK, cache_dir, refresh=args.refresh)
    if benchmark.empty:
        raise RuntimeError("Could not download S&P 500 index history (^GSPC).")

    returns = compute_forward_returns(audit, histories, benchmark)
    dips = compute_dips(audit, histories)
    placebo = compute_placebo_dips(audit, histories)

    inference = inference_table(returns)
    dip = dip_summary(dips)
    placebo_out = placebo_summary(placebo)
    corr = correlation_table(returns)
    decade = decade_table(returns)

    returns.to_csv(out_dir / "company_level_returns.csv", index=False)
    dips.to_csv(out_dir / "company_level_dips.csv", index=False)
    placebo.to_csv(out_dir / "company_level_placebo_dips.csv", index=False)
    inference.to_csv(out_dir / "timespan_inference_summary.csv", index=False)
    dip.to_csv(out_dir / "dip_summary.csv", index=False)
    placebo_out.to_csv(out_dir / "placebo_dip_summary.csv", index=False)
    corr.to_csv(out_dir / "first_day_move_correlations.csv", index=False)
    decade.to_csv(out_dir / "decade_summary.csv", index=False)

    make_plots(inference, dip, placebo_out, decade, out_dir)

    print("\nOne-year result")
    print(inference[inference.horizon.eq("1y")].to_string(index=False))
    print("\nDip vs fixed wait")
    print(dip.to_string(index=False))
    print("\nTwo-years-later placebo")
    print(placebo_out.to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-run the IPO survivor-cohort analysis.")
    p.add_argument("--refresh", action="store_true", help="Redownload Yahoo price histories instead of using the local cache.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
