"""MACD calculation, timeframe resampling and crossover detection."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FAST = 12
SLOW = 26
SIGNAL = 9

# Rule names for pandas resample. Bars are labelled by their last day.
TIMEFRAME_RULES = {
    "daily": None,
    "weekly": "W-FRI",
    "monthly": "ME",
}


def macd(close: pd.Series, fast: int = FAST, slow: int = SLOW, signal: int = SIGNAL) -> pd.DataFrame:
    """Standard MACD (EMA fast - EMA slow), signal line and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def resample_close(daily_close: pd.Series, timeframe: str, closed_bars_only: bool = False) -> pd.Series:
    """Build weekly/monthly closes from daily closes.

    By default the current, still-forming week/month is kept as the last bar
    (the same way charting platforms show it). With ``closed_bars_only`` the
    in-progress bar is dropped.
    """
    rule = TIMEFRAME_RULES[timeframe]
    daily_close = daily_close.dropna()
    if rule is None:
        return daily_close
    out = daily_close.resample(rule).last().dropna()
    if closed_bars_only and len(out) and len(daily_close):
        last_day = daily_close.index[-1]
        if _period_is_open(last_day, timeframe):
            out = out.iloc[:-1]
    return out


def _period_is_open(last_day: pd.Timestamp, timeframe: str) -> bool:
    """True when ``last_day`` is not the final trading day of its week/month."""
    next_bday = last_day + pd.offsets.BDay(1)
    if timeframe == "weekly":
        return next_bday.isocalendar()[:2] == last_day.isocalendar()[:2]
    if timeframe == "monthly":
        return (next_bday.year, next_bday.month) == (last_day.year, last_day.month)
    return False


@dataclass
class CrossResult:
    crossed: bool
    bars_since_cross: int | None
    macd: float
    signal: float
    hist: float
    accelerating: bool


def bullish_cross(close: pd.Series, lookback: int = 1, min_bars: int | None = None) -> CrossResult | None:
    """Check for a MACD line crossing above its signal line.

    A cross happens on bar ``i`` when the histogram was <= 0 on bar ``i-1`` and
    is > 0 on bar ``i``. ``crossed`` is True when such a cross happened within
    the last ``lookback`` bars and MACD is still above the signal line now.
    ``accelerating`` means the histogram grew versus the previous bar.

    Returns None when there is not enough history for a meaningful MACD.
    """
    min_bars = min_bars if min_bars is not None else SLOW + SIGNAL
    close = close.dropna()
    if len(close) < max(min_bars, lookback + 1):
        return None
    m = macd(close)
    hist = m["hist"].to_numpy()

    bars_since = None
    for back in range(len(hist) - 1, 0, -1):
        if hist[back - 1] <= 0 < hist[back]:
            bars_since = len(hist) - 1 - back
            break

    last = m.iloc[-1]
    is_open = last["hist"] > 0
    crossed = bool(is_open and bars_since is not None and bars_since < lookback)
    return CrossResult(
        crossed=crossed,
        bars_since_cross=bars_since,
        macd=float(last["macd"]),
        signal=float(last["signal"]),
        hist=float(last["hist"]),
        accelerating=bool(hist[-1] > hist[-2]),
    )
