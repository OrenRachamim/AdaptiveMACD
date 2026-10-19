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


def build_forecast_messages(result: pd.DataFrame, scan_date: str, min_prob: float = 0.70) -> list[str]:
    """Message for screener.cross_scan: stocks likely to get a daily MACD cross at the next close."""
    header = (f"🔮 <b>תחזית הצטלבות MACD יומית – {scan_date}</b>\n"
              f"מניות שה-MACD היומי שלהן צפוי לחצות את ה-Signal כלפי מעלה בסגירה הבאה (סיכוי ≥ {min_prob:.0%})\n")
    if result.empty:
        return [header + "\nאין מניות מעל הסף היום."]
    result = result.fillna({"name": "", "exchange": ""})
    header += (f"<b>{len(result)} מניות</b>\n"
               "<i>שינוי נדרש שלילי = ההצטלבות תקרה גם אם המניה תרד עד השיעור הזה</i>\n")
    blocks = []
    for r in result.to_dict("records"):
        url = tradingview_url(r["ticker"], r.get("exchange", ""))
        blocks.append(
            f'\n<a href="{url}"><b>{html.escape(r["ticker"])}</b></a> – {html.escape(str(r.get("name", ""))[:40])}\n'
            f"סגירה ${r['close']} → הצטלבות בסגירה מעל <b>${r['cross_price']}</b> "
            f"({r['required_change_pct']:+.2f}%)\n"
            f"סיכוי <b>{r['cross_prob']:.0%}</b>\n"
        )
    return _split(header, blocks)


def _split(header: str, blocks: list[str]) -> list[str]:
    messages, current = [], header
    for block in blocks:
        if len(current) + len(block) > MAX_MESSAGE:
            messages.append(current)
            current = ""
        current += block
    messages.append(current)
    return messages


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

    return _split(header, blocks)


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
    p.add_argument("--kind", choices=["forecast", "crossover"], default="forecast",
                   help="forecast = screener.cross_scan results (default), crossover = screener.scan results")
    p.add_argument("--min-prob", type=float, default=0.70, help="threshold shown in the forecast header")
    args = p.parse_args(argv)

    try:
        result = pd.read_csv(args.results)
    except pd.errors.EmptyDataError:  # no matches -> CSV without columns
        result = pd.DataFrame()
    scan_date = str(result["date"].iloc[0]) if "date" in result.columns and len(result) else \
        pd.Timestamp(args.results.stat().st_mtime, unit="s").date().isoformat()
    if args.kind == "forecast":
        messages = build_forecast_messages(result, scan_date, args.min_prob)
    else:
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
