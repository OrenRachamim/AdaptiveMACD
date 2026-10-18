"""Send the latest scan results to a Telegram chat.

Needs two environment variables:
    TELEGRAM_BOT_TOKEN  token from @BotFather
    TELEGRAM_CHAT_ID    chat / channel / group id to post to

Usage:
    python -m screener.notify_telegram                 # send results/latest.csv
    python -m screener.notify_telegram --dry-run       # print the message instead
"""

from __future__ import annotations

import argparse
import html
import os
import sys
from pathlib import Path

import pandas as pd
import requests

API = "https://api.telegram.org/bot{token}/{method}"
MAX_MESSAGE = 4000  # Telegram limit is 4096 characters

# Exchange names from the universe -> TradingView exchange prefixes.
TV_EXCHANGES = {
    "NASDAQ": "NASDAQ",
    "NYSE": "NYSE",
    "NYSE American": "AMEX",
    "NYSE Arca": "AMEX",
    "Cboe": "CBOE",
    "NYSE Texas": "NYSE",
}

TF_LABELS = {"monthly": "חודשי", "weekly": "שבועי", "daily": "יומי"}


def tradingview_url(ticker: str, exchange: str) -> str:
    symbol = ticker.replace("-", ".")  # BRK-B -> BRK.B
    prefix = TV_EXCHANGES.get(str(exchange))
    if prefix:
        symbol = f"{prefix}:{symbol}"
    return f"https://www.tradingview.com/chart/?symbol={symbol}"


def _cross_label(tf: str, value) -> str:
    day = pd.Timestamp(value)
    if tf == "monthly":
        return day.strftime("%m/%Y")
    if tf == "weekly":
        return "שבוע " + day.strftime("%d/%m")
    return day.strftime("%d/%m")


def build_messages(result: pd.DataFrame, scan_date: str) -> list[str]:
    header = f"📈 <b>סריקת MACD – {scan_date}</b>\n"
    if result.empty:
        return [header + "לא נמצאו מניות שעומדות בתנאים היום."]

    result = result.fillna({"name": "", "exchange": ""})
    timeframes = [tf for tf in TF_LABELS if f"{tf}_cross_bar" in result.columns]
    header += (f"חציית MACD מעל ה-Signal בגרף ה{' וה'.join(TF_LABELS[tf] for tf in timeframes)}\n"
               f"<b>{len(result)} מניות</b>\n")

    blocks = []
    for r in result.to_dict("records"):
        url = tradingview_url(r["ticker"], r.get("exchange", ""))
        crosses = " | ".join(f"{TF_LABELS[tf]}: {_cross_label(tf, r[f'{tf}_cross_bar'])}" for tf in timeframes)
        blocks.append(
            f'\n<a href="{url}"><b>{html.escape(r["ticker"])}</b></a> – {html.escape(str(r.get("name", ""))[:40])}\n'
            f"סגירה ${r['close']} | מחזור ממוצע {int(r['avg_volume_50d']):,}\n"
            f"חצייה: {crosses}\n"
        )

    messages, current = [], header
    for block in blocks:
        if len(current) + len(block) > MAX_MESSAGE:
            messages.append(current)
            current = ""
        current += block
    messages.append(current)
    return messages


def send(messages: list[str], token: str, chat_id: str) -> None:
    for text in messages:
        resp = requests.post(
            API.format(token=token, method="sendMessage"),
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(f"Telegram API error {resp.status_code}: {resp.text}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results/latest.csv"))
    p.add_argument("--dry-run", action="store_true", help="print the message instead of sending it")
    args = p.parse_args(argv)

    try:
        result = pd.read_csv(args.results)
    except pd.errors.EmptyDataError:  # no matches -> CSV without columns
        result = pd.DataFrame()
    scan_date = str(result["date"].iloc[0]) if "date" in result.columns and len(result) else \
        pd.Timestamp(args.results.stat().st_mtime, unit="s").date().isoformat()
    messages = build_messages(result, scan_date)

    if args.dry_run:
        print("\n-----\n".join(messages))
        return 0

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set - skipping Telegram notification")
        return 0
    send(messages, token, chat_id)
    print(f"sent {len(messages)} Telegram message(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
