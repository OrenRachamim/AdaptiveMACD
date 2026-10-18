"""Backtest of the daily screen: average price change N days after each new signal.

A signal day is the first day a stock passes the screen (it did not pass the
previous trading day). The return is measured from that day's close to the
close 1, 2, 3... trading days later, and compared with the average return of
all eligible stocks over the same days.

The screen is re-computed for every historical day exactly as the daily scan
would have seen it, including the still-forming monthly bar.

Usage:
    python -m screener.backtest                 # full US universe, 10 years
    python -m screener.backtest --limit 500     # quick run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from screener.macd import FAST, SIGNAL, SLOW, macd
from screener.scan import download, ticker_frame
from screener.universe import load_universe

log = logging.getLogger("backtest")


def daily_screen(close: pd.Series, lookback: int, below_zero: bool, min_hist_pct: float,
                 allow_whipsaw: bool) -> pd.Series:
    m = macd(close)
    hist = m["hist"]
    cross = (hist.shift(1) <= 0) & (hist > 0)
    ok = (hist > 0) & (cross.astype(int).rolling(lookback, min_periods=1).max() > 0)
    if below_zero:
        ok &= m["macd"].where(cross).ffill() < 0
    ok &= hist >= min_hist_pct / 100 * m["signal"].abs()
    if not allow_whipsaw:
        flips = (hist > 0).astype(int).diff().abs()
        ok &= flips.rolling(lookback, min_periods=1).sum() <= 1
    ok &= np.arange(len(close)) >= SLOW + SIGNAL - 1
    return ok.fillna(False)


def monthly_screen(close: pd.Series, lookback: int, below_zero: bool, min_hist_pct: float,
                   allow_whipsaw: bool) -> pd.Series:
    """Monthly screen as seen on each day, with the current month's bar built from that day's close."""
    if lookback > 2:
        raise ValueError("backtest supports monthly lookback 1 or 2")
    month = close.index.to_period("M")
    mclose = close.groupby(month).last()
    ef = mclose.ewm(span=FAST, adjust=False).mean()
    es = mclose.ewm(span=SLOW, adjust=False).mean()
    mm = ef - es
    ms = mm.ewm(span=SIGNAL, adjust=False).mean()
    mh = mm - ms

    def prev(series: pd.Series, n: int) -> np.ndarray:
        return series.shift(n).reindex(month).to_numpy()

    a_f, a_s, a_g = 2 / (FAST + 1), 2 / (SLOW + 1), 2 / (SIGNAL + 1)
    c = close.to_numpy()
    macd_now = (a_f * c + (1 - a_f) * prev(ef, 1)) - (a_s * c + (1 - a_s) * prev(es, 1))
    signal_now = a_g * macd_now + (1 - a_g) * prev(ms, 1)
    hist_now = macd_now - signal_now
    h1, h2, macd1 = prev(mh, 1), prev(mh, 2), prev(mm, 1)

    cross_now = (h1 <= 0) & (hist_now > 0)
    cross_prev = (h2 <= 0) & (h1 > 0)
    ok = hist_now > 0
    if lookback == 1:
        ok &= cross_now
        macd_at_cross = macd_now
    else:
        ok &= cross_now | cross_prev
        macd_at_cross = np.where(cross_now, macd_now, macd1)
    if below_zero:
        ok &= macd_at_cross < 0
    ok &= hist_now >= min_hist_pct / 100 * np.abs(signal_now)
    if not allow_whipsaw and lookback == 2:
        ok &= ~((h2 > 0) & (h1 <= 0))  # up, down, up again over the last 3 bars
    month_pos = pd.Series(np.arange(len(mclose)), index=mclose.index).reindex(month).to_numpy()
    ok &= month_pos >= SLOW + SIGNAL - 1
    return pd.Series(np.nan_to_num(ok, nan=0).astype(bool), index=close.index)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = load_universe()
    if args.limit:
        universe = universe.head(args.limit)
    tickers = universe["ticker"].tolist()
    horizons = args.horizons
    signals = []
    bench_sum = pd.DataFrame(dtype=float)
    bench_cnt = pd.DataFrame(dtype=float)

    for start in range(0, len(tickers), args.batch_size):
        batch = tickers[start:start + args.batch_size]
        data = download(batch, args.period, adjust_dividends=False)
        for ticker in batch:
            df = ticker_frame(data, ticker) if not data.empty else None
            if df is None or len(df) < 800:
                continue
            close = df["Close"]
            eligible = (close >= args.min_price) & (df["Volume"].rolling(50).mean() >= args.min_avg_volume)
            ok = eligible & daily_screen(close, args.daily_lookback, True, args.min_hist_pct, False) \
                & monthly_screen(close, args.monthly_lookback, True, args.min_hist_pct, False)
            new = ok & ~ok.shift(1, fill_value=False)
            fwd = pd.DataFrame({h: close.shift(-h) / close - 1 for h in horizons})
            # Benchmark: every eligible stock, every day.
            elig_fwd = fwd.where(eligible, np.nan)
            bench_sum = bench_sum.add(elig_fwd.fillna(0), fill_value=0)
            bench_cnt = bench_cnt.add(elig_fwd.notna().astype(float), fill_value=0)
            for day in close.index[new.to_numpy()]:
                row = {"ticker": ticker, "date": day.date(), "close": round(float(close[day]), 2)}
                row.update({f"ret_{h}d": fwd.at[day, h] for h in horizons})
                signals.append(row)
        log.info("%d/%d tickers, %d signals", min(start + args.batch_size, len(tickers)), len(tickers), len(signals))

    sig = pd.DataFrame(signals)
    bench = bench_sum / bench_cnt.replace(0, np.nan)
    return sig, bench


def summarize(sig: pd.DataFrame, bench: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(sig["date"])
    for h in horizons:
        r = sig[f"ret_{h}d"].dropna()
        b = bench[h].reindex(dates[r.index]).to_numpy()
        rows.append({
            "horizon": f"{h}d",
            "signals": len(r),
            "mean_%": round(100 * r.mean(), 2),
            "median_%": round(100 * r.median(), 2),
            "win_rate_%": round(100 * (r > 0).mean(), 1),
            "benchmark_mean_%": round(100 * np.nanmean(b), 2),
            "excess_mean_%": round(100 * (r.to_numpy() - b).mean(), 2),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--period", default="10y")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--daily-lookback", type=int, default=5)
    p.add_argument("--monthly-lookback", type=int, default=2)
    p.add_argument("--min-hist-pct", type=float, default=5.0)
    p.add_argument("--min-price", type=float, default=1.0)
    p.add_argument("--min-avg-volume", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--limit", type=int)
    p.add_argument("--out-dir", type=Path, default=Path("backtest_results"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    sig, bench = run(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sig.to_csv(args.out_dir / "signals.csv", index=False)
    summary = summarize(sig, bench, args.horizons)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
