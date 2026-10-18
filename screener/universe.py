"""US-listed stock universe from the NASDAQ Trader symbol directory.

Covers NASDAQ, NYSE, NYSE American, NYSE Arca and Cboe listings.
"""

from __future__ import annotations

import io
import re

import pandas as pd
import requests

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

EXCHANGES = {"Q": "NASDAQ", "N": "NYSE", "A": "NYSE American", "P": "NYSE Arca", "Z": "Cboe", "V": "IEX", "F": "NYSE Texas"}

# Securities that are not common stock.
NON_COMMON = re.compile(
    r"\b(?:warrants?|units?|rights?|preferred|notes?|debentures?)\b",
    re.IGNORECASE,
)


def _read(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), sep="|", dtype=str, keep_default_na=False)
    # Last row is "File Creation Time: ..."
    return df[~df.iloc[:, 0].str.startswith("File Creation Time")]


def load_universe(include_etfs: bool = False) -> pd.DataFrame:
    """Return a DataFrame with columns ticker, name, exchange."""
    nasdaq = _read(NASDAQ_URL)
    nasdaq = nasdaq[nasdaq["Test Issue"] == "N"]
    nasdaq = pd.DataFrame({
        "ticker": nasdaq["Symbol"],
        "name": nasdaq["Security Name"],
        "exchange": "NASDAQ",
        "etf": nasdaq["ETF"] == "Y",
    })

    other = _read(OTHER_URL)
    other = other[other["Test Issue"] == "N"]
    other = pd.DataFrame({
        "ticker": other["ACT Symbol"],
        "name": other["Security Name"],
        "exchange": other["Exchange"].map(EXCHANGES).fillna(other["Exchange"]),
        "etf": other["ETF"] == "Y",
    })

    df = pd.concat([nasdaq, other], ignore_index=True)
    if not include_etfs:
        df = df[~df["etf"]]
    # Preferreds and when-issued symbols carry '$' / '#'.
    df = df[~df["ticker"].str.contains(r"[$#^]", regex=True)]
    df = df[~df["name"].str.contains(NON_COMMON)]
    # Yahoo uses '-' for share classes: BRK.B -> BRK-B
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False).str.strip()
    df = df[df["ticker"] != ""]
    return df.drop(columns="etf").drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
