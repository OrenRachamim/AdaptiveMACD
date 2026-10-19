"""Backtest of the daily screen: average price change N days after each new signal.

A signal day is a day on which the daily MACD crosses above its signal line
and the whole screen passes. The trade is entered at the next day's open and
the return is measured to the close 1, 2, 3... trading days after entry
(1 = the close of the entry day). It is compared with the average return of
all eligible stocks over the same open-to-close window.

The screen is re-computed for every historical day exactly as the daily scan
would have seen it, including the still-forming weekly/monthly bar.

Usage:
    python -m screener.backtest                 # full US universe, 10 years
    python -m screener.backtest --limit 500     # quick run
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from screener.macd import FAST, SIGNAL, SLOW, macd
from screener.scan import download, ticker_frame
from screener.universe import load_universe

log = logging.getLogger("backtest")


def daily_precross_screen(close: pd.Series, bars: int, zero: str) -> tuple[pd.Series, pd.Series]:
    """MACD still below its signal line but converging fast enough to cross within ``bars`` bars.

    Returns the screen and, for each day, whether the cross really happened within ``bars`` bars after it.
    """
    m = macd(close)
    hist = m["hist"]
    step = hist.diff()
    ok = (hist < 0) & (step > 0) & (step.shift(1) > 0) & (-hist / step <= bars)
    ok &= zero_ok(m["macd"], zero)
    ok &= np.arange(len(close)) >= SLOW + SIGNAL - 1
    crossed_later = pd.concat([hist.shift(-k) > 0 for k in range(1, bars + 1)], axis=1).any(axis=1)
    return ok.fillna(False), crossed_later


def cross_price(close: pd.Series) -> pd.Series:
    """Close needed on the next bar for the MACD line to end above its signal line.

    MACD is linear in the new close, so the threshold is exact:
    hist' > 0  <=>  c' > (signal - (1-a_f)*EMA_fast + (1-a_s)*EMA_slow) / (a_f - a_s)
    """
    a_f, a_s = 2 / (FAST + 1), 2 / (SLOW + 1)
    ef = close.ewm(span=FAST, adjust=False).mean()
    es = close.ewm(span=SLOW, adjust=False).mean()
    sig = (ef - es).ewm(span=SIGNAL, adjust=False).mean()
    return (sig - (1 - a_f) * ef + (1 - a_s) * es) / (a_f - a_s)


def daily_crossprice_screen(close: pd.Series, max_rise_pct: float, zero: str) -> tuple[pd.Series, pd.Series]:
    """MACD still below its signal line and a close at most ``max_rise_pct`` above today's would cross it.

    Returns the screen and the required rise in % for every day.
    """
    m = macd(close)
    rise = (cross_price(close) / close - 1) * 100
    ok = (m["hist"] < 0) & (rise <= max_rise_pct)
    ok &= zero_ok(m["macd"], zero)
    ok &= np.arange(len(close)) >= SLOW + SIGNAL - 1
    return ok.fillna(False), rise


def zero_ok(macd_at_cross, zero: str):
    """Where the MACD line was on the cross bar: 'below' 0, 'above' 0 or 'any'."""
    if zero == "below":
        return macd_at_cross < 0
    if zero == "above":
        return macd_at_cross > 0
    return True


def daily_screen(close: pd.Series, lookback: int, zero: str, min_hist_pct: float,
                 allow_whipsaw: bool) -> pd.Series:
    m = macd(close)
    hist = m["hist"]
    cross = (hist.shift(1) <= 0) & (hist > 0)
    ok = (hist > 0) & (cross.astype(int).rolling(lookback, min_periods=1).max() > 0)
    ok &= zero_ok(m["macd"].where(cross).ffill(), zero)
    ok &= hist >= min_hist_pct / 100 * m["signal"].abs()
    if not allow_whipsaw:
        flips = (hist > 0).astype(int).diff().abs()
        ok &= flips.rolling(lookback, min_periods=1).sum() <= 1
    ok &= np.arange(len(close)) >= SLOW + SIGNAL - 1
    return ok.fillna(False)


PERIODS = {"monthly": "M", "weekly": "W-FRI"}


def higher_screen(close: pd.Series, timeframe: str, lookback: int, zero: str, min_hist_pct: float,
                  allow_whipsaw: bool) -> pd.Series:
    """Weekly/monthly screen as seen on each day, with the current bar built from that day's close."""
    if lookback > 2:
        raise ValueError("backtest supports a weekly/monthly lookback of 1 or 2")
    month = close.index.to_period(PERIODS[timeframe])
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
    ok &= zero_ok(macd_at_cross, zero)
    ok &= hist_now >= min_hist_pct / 100 * np.abs(signal_now)
    if not allow_whipsaw and lookback == 2:
        ok &= ~((h2 > 0) & (h1 <= 0))  # up, down, up again over the last 3 bars
    month_pos = pd.Series(np.arange(len(mclose)), index=mclose.index).reindex(month).to_numpy()
    ok &= month_pos >= SLOW + SIGNAL - 1
    return pd.Series(np.nan_to_num(ok, nan=0).astype(bool), index=close.index)


def cached_download(batch: list[str], period: str, cache_dir: Path | None) -> pd.DataFrame:
    if cache_dir is None:
        return download(batch, period, adjust_dividends=False)
    key = hashlib.md5(f"{period}|{','.join(batch)}".encode()).hexdigest()
    path = cache_dir / f"{key}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    data = download(batch, period, adjust_dividends=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data.to_pickle(path)
    return data


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
        data = cached_download(batch, args.period, args.cache_dir)
        for ticker in batch:
            df = ticker_frame(data, ticker) if not data.empty else None
            if df is None or len(df) < 800:
                continue
            close = df["Close"]
            eligible = (close >= args.min_price) & (df["Volume"].rolling(50).mean() >= args.min_avg_volume)
            rise = pd.Series(np.nan, index=close.index)
            if args.daily_mode == "crossprice":
                daily, rise = daily_crossprice_screen(close, args.max_rise_pct, args.daily_zero)
                hist = macd(close)["hist"]
                crossed_later = hist.shift(-1) > 0
            elif args.daily_mode == "precross":
                daily, crossed_later = daily_precross_screen(close, args.precross_bars, args.daily_zero)
            else:
                daily = daily_screen(close, args.daily_lookback, args.daily_zero, args.min_hist_pct, False)
                crossed_later = pd.Series(np.nan, index=close.index)
            ok = eligible & daily \
                & higher_screen(close, args.higher_tf, args.higher_lookback, args.higher_zero, args.min_hist_pct,
                                False)
            new = ok & ~ok.shift(1, fill_value=False)
            entry = df["Open"].shift(-1).where(lambda o: o > 0)
            fwd = pd.DataFrame({h: close.shift(-h) / entry - 1 for h in horizons})
            # Benchmark: every eligible stock, every day.
            elig_fwd = fwd.where(eligible, np.nan)
            bench_sum = bench_sum.add(elig_fwd.fillna(0), fill_value=0)
            bench_cnt = bench_cnt.add(elig_fwd.notna().astype(float), fill_value=0)
            for day in close.index[new.to_numpy()]:
                row = {"ticker": ticker, "date": day.date(), "close": round(float(close[day]), 2),
                       "entry_open": round(float(entry[day]), 2), "crossed_later": crossed_later[day],
                       "required_rise_%": round(float(rise[day]), 3)}
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
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 5, 10],
                   help="exit after N trading days, counting the entry day as 1")
    p.add_argument("--daily-lookback", type=int, default=1,
                   help="1 = signal on the daily cross day itself (entry next open)")
    p.add_argument("--daily-mode", choices=["cross", "precross", "crossprice"], default="cross",
                   help="cross: signal on the daily cross; precross: signal 1-2 bars before an expected cross; "
                        "crossprice: lines not crossed yet but a small rise tomorrow would cross them")
    p.add_argument("--max-rise-pct", type=float, default=1.0,
                   help="crossprice: max rise (%%) from today's close needed for tomorrow's cross")
    p.add_argument("--precross-bars", type=int, default=2,
                   help="precross: lines must be on course to cross within this many bars")
    p.add_argument("--higher-tf", choices=list(PERIODS), default="monthly",
                   help="higher timeframe that must also cross (default: monthly)")
    p.add_argument("--higher-lookback", "--monthly-lookback", dest="higher_lookback", type=int, default=2)
    zero = dict(choices=["below", "above", "any"], default="below")
    p.add_argument("--daily-zero", **zero, help="daily MACD line position vs 0 on the cross bar")
    p.add_argument("--higher-zero", "--monthly-zero", dest="higher_zero", **zero,
                   help="weekly/monthly MACD line position vs 0 on the cross bar")
    p.add_argument("--min-hist-pct", type=float, default=5.0)
    p.add_argument("--min-price", type=float, default=1.0)
    p.add_argument("--min-avg-volume", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--limit", type=int)
    p.add_argument("--cache-dir", type=Path, help="cache downloaded prices here to speed up repeated runs")
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
