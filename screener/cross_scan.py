"""Daily scan: US stocks whose daily MACD is likely to cross above its signal line at the next close.

For every stock whose MACD line is still below the signal line, the exact close
needed tomorrow for a cross is computed, scaled by the stock's 60-day daily
volatility, and turned into a probability with a calibration table fitted on
2.46 million historical stock-days (screener/data/daily_cross_calibration.csv).
Stocks at or above the probability threshold are reported.

Usage:
    python -m screener.cross_scan                    # full US universe, threshold 70%
    python -m screener.cross_scan --min-prob 0.5
    python -m screener.cross_scan --tickers AAPL MSFT
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

from screener.cross_forecast import Calibration, daily_features, daily_vol, z_score
from screener.scan import download, ticker_frame
from screener.universe import load_universe

log = logging.getLogger("cross_scan")

CALIBRATION = Path(__file__).parent / "data" / "daily_cross_calibration.csv"
VOL_WINDOW = 60


def evaluate(df: pd.DataFrame, calibration: Calibration, args: argparse.Namespace) -> dict | None:
    close = df["Close"]
    if len(close) < 200:
        return None
    last_close = float(close.iloc[-1])
    avg_volume = float(df["Volume"].tail(50).mean())
    if last_close < args.min_price or avg_volume < args.min_avg_volume:
        return None
    last = daily_features(close).iloc[-1]
    if last["hist"] >= 0:  # already above the signal line
        return None
    vol = float(daily_vol(close, VOL_WINDOW).iloc[-1])
    z = float(z_score([last["rise_pct"]], [vol], [1])[0])
    prob = float(calibration([z])[0])
    if prob < args.min_prob:
        return None
    return {
        "date": close.index[-1].date(),
        "close": round(last_close, 2),
        "cross_price": round(float(last["cross_price"]), 2),
        "required_change_pct": round(float(last["rise_pct"]), 2),
        "cross_prob": round(prob, 3),
        "daily_vol_pct": round(vol, 2),
        "macd": round(float(last["macd"]), 4),
        "hist": round(float(last["hist"]), 4),
        "avg_volume_50d": int(avg_volume),
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    calibration = Calibration.from_table(pd.read_csv(CALIBRATION))
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
                row = evaluate(df, calibration, args)
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
        result = result.sort_values(["cross_prob", "avg_volume_50d"], ascending=False).reset_index(drop=True)
    return result


def write_outputs(result: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    result.to_csv(out_dir / f"{stamp}.csv", index=False)
    result.to_csv(out_dir / "latest.csv", index=False)
    lines = [
        f"# Daily MACD cross forecast - {stamp}",
        "",
        f"Stocks whose daily MACD is below its signal line and is likely (>= {args.min_prob:.0%}) to cross above it "
        "at the next close. Required change = move from the last close to the cross price.",
        "",
        f"**{len(result)} stocks**",
        "",
    ]
    if not result.empty:
        lines += ["| Ticker | Name | Close | Cross above | Required change | Probability |",
                  "|---|---|---:|---:|---:|---:|"]
        for r in result.to_dict("records"):
            lines.append(f"| {r['ticker']} | {str(r['name'])[:40]} | {r['close']} | {r['cross_price']} | "
                         f"{r['required_change_pct']:+.2f}% | {r['cross_prob']:.0%} |")
    (out_dir / "latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", help="scan only these tickers instead of the whole US market")
    p.add_argument("--min-prob", type=float, default=0.70, help="minimum cross probability (default 0.70)")
    p.add_argument("--include-etfs", action="store_true")
    p.add_argument("--min-price", type=float, default=1.0)
    p.add_argument("--min-avg-volume", type=int, default=50_000)
    p.add_argument("--period", default="2y", help="history to download")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--pause", type=float, default=1.0)
    p.add_argument("--limit", type=int)
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
    log.info("%d stocks written to %s", len(result), args.out_dir)
    if not result.empty:
        print(result[["ticker", "close", "cross_price", "required_change_pct", "cross_prob"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
