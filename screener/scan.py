"""Daily scan: US stocks whose MACD crosses above its signal line on the
monthly, weekly and daily timeframes at the same time.

Usage:
    python -m screener.scan                      # full US universe
    python -m screener.scan --tickers AAPL MSFT  # specific tickers
    python -m screener.scan --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from screener.macd import bullish_cross, resample_close
from screener.universe import load_universe

log = logging.getLogger("screener")

TIMEFRAMES = ("monthly", "weekly", "daily")


def download(tickers: list[str], period: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        try:
            data = yf.download(
                tickers,
                period=period,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if data is not None and not data.empty:
                return data
        except Exception as exc:  # network / rate limit
            log.warning("download failed (attempt %d/%d): %s", attempt, retries, exc)
        time.sleep(5 * attempt)
    return pd.DataFrame()


def ticker_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            return None
        df = data[ticker]
    else:
        df = data
    df = df.dropna(subset=["Close"])
    return df if not df.empty else None


def evaluate(df: pd.DataFrame, args: argparse.Namespace) -> dict | None:
    close = df["Close"]
    last_close = float(close.iloc[-1])
    avg_volume = float(df["Volume"].tail(50).mean())
    if last_close < args.min_price or avg_volume < args.min_avg_volume:
        return None

    row = {"close": round(last_close, 2), "avg_volume_50d": int(avg_volume), "date": close.index[-1].date()}
    # Daily first: it is the cheapest and most selective filter.
    for tf in ("daily", "weekly", "monthly"):
        series = resample_close(close, tf, closed_bars_only=args.closed_bars_only and tf != "daily")
        res = bullish_cross(series, lookback=getattr(args, f"{tf}_lookback"))
        if res is None or not res.crossed:
            return None
        if getattr(args, f"{tf}_below_zero") and not res.macd_at_cross < 0:
            return None
        if args.require_acceleration and not res.accelerating:
            return None
        row[f"{tf}_bars_since_cross"] = res.bars_since_cross
        row[f"{tf}_macd_at_cross"] = round(res.macd_at_cross, 4)
        row[f"{tf}_macd"] = round(res.macd, 4)
        row[f"{tf}_signal"] = round(res.signal, 4)
        row[f"{tf}_hist"] = round(res.hist, 4)
        row[f"{tf}_accelerating"] = res.accelerating
    return row


def run(args: argparse.Namespace) -> pd.DataFrame:
    if args.tickers:
        universe = pd.DataFrame({"ticker": args.tickers, "name": "", "exchange": ""})
    else:
        universe = load_universe(include_etfs=args.include_etfs)
    if args.limit:
        universe = universe.head(args.limit)
    log.info("scanning %d tickers", len(universe))

    meta = universe.set_index("ticker")
    tickers = universe["ticker"].tolist()
    matches = []
    for start in range(0, len(tickers), args.batch_size):
        batch = tickers[start:start + args.batch_size]
        data = download(batch, args.period)
        for ticker in batch:
            df = ticker_frame(data, ticker) if not data.empty else None
            if df is None:
                continue
            try:
                row = evaluate(df, args)
            except Exception as exc:
                log.debug("%s: %s", ticker, exc)
                continue
            if row:
                matches.append({"ticker": ticker, "name": meta.at[ticker, "name"],
                                "exchange": meta.at[ticker, "exchange"], **row})
        log.info("%d/%d done, %d matches", min(start + args.batch_size, len(tickers)), len(tickers), len(matches))
        time.sleep(args.pause)

    result = pd.DataFrame(matches)
    if not result.empty:
        result = result.sort_values("avg_volume_50d", ascending=False).reset_index(drop=True)
    return result


def write_outputs(result: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    result.to_csv(out_dir / f"{stamp}.csv", index=False)
    result.to_csv(out_dir / "latest.csv", index=False)

    lines = [
        f"# MACD triple-timeframe bullish crossover - {stamp}",
        "",
        f"MACD({12},{26},{9}) line crossed above the signal line on the monthly, weekly and daily charts.",
        f"Lookback (bars): monthly={args.monthly_lookback}, weekly={args.weekly_lookback}, daily={args.daily_lookback}. "
        f"MACD below 0 at the cross: monthly={args.monthly_below_zero}, weekly={args.weekly_below_zero}, "
        f"daily={args.daily_below_zero}. "
        f"Min price ${args.min_price}, min 50-day avg volume {args.min_avg_volume:,}.",
        "",
        f"**{len(result)} matches**",
        "",
    ]
    if not result.empty:
        lines += ["| Ticker | Name | Exchange | Close | Avg Vol 50d | M hist | W hist | D hist |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        for r in result.itertuples():
            lines.append(f"| {r.ticker} | {str(r.name)[:40]} | {r.exchange} | {r.close} | {r.avg_volume_50d:,} "
                         f"| {r.monthly_hist} | {r.weekly_hist} | {r.daily_hist} |")
    (out_dir / "latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", help="scan only these tickers instead of the whole US market")
    p.add_argument("--include-etfs", action="store_true", help="include ETFs in the universe")
    p.add_argument("--daily-lookback", type=int, default=5,
                   help="cross must have happened within the last N daily bars (1 = today only)")
    p.add_argument("--weekly-lookback", type=int, default=2,
                   help="cross must have happened within the last N weekly bars (1 = this week only)")
    p.add_argument("--monthly-lookback", type=int, default=2,
                   help="cross must have happened within the last N monthly bars (1 = this month only)")
    p.add_argument("--closed-bars-only", action="store_true",
                   help="ignore the still-forming week/month bar")
    for tf in ("daily", "weekly", "monthly"):
        p.add_argument(f"--{tf}-below-zero", action=argparse.BooleanOptionalAction, default=True,
                       help=f"require the {tf} MACD line to be below 0 on the cross bar")
    p.add_argument("--require-acceleration", action="store_true",
                   help="also require the histogram to be growing on every timeframe")
    p.add_argument("--min-price", type=float, default=1.0)
    p.add_argument("--min-avg-volume", type=int, default=50_000)
    p.add_argument("--period", default="6y", help="history to download (needs >= ~3y for monthly MACD)")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--pause", type=float, default=1.0, help="seconds to wait between batches")
    p.add_argument("--limit", type=int, help="scan only the first N tickers (for testing)")
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    result = run(args)
    write_outputs(result, args.out_dir, args)
    log.info("%d matches written to %s", len(result), args.out_dir)
    if not result.empty:
        print(result[["ticker", "name", "close", "avg_volume_50d"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
