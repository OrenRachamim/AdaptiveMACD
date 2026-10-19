"""Standalone per-stock forecast: will the daily MACD line reach (touch or cross) its signal line
at the next close, while the lines are still apart today?

Self-contained (numpy + pandas only). Port of screener/cross_forecast.py + screener/cross_scan.py
(touch mode) from OrenRachamim/AdaptiveMACD.

Usage:
    import yfinance as yf
    df = yf.download("AAPL", period="2y", interval="1d", auto_adjust=False, progress=False)
    result = macd_touch_forecast(df["Close"].squeeze())
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FAST, SLOW, SIGNAL = 12, 26, 9
A_F, A_S, A_G = 2 / (FAST + 1), 2 / (SLOW + 1), 2 / (SIGNAL + 1)
TOLERANCE = 0.05   # lines "touch" when signal - MACD <= 5% of |signal|
VOL_WINDOW = 60    # daily volatility: std of daily % returns over 60 days
MIN_BARS = 200

# Calibration z -> probability of touching/crossing at the next close, fitted on 2.2M
# historical US stock-days (1965-2026) where the lines were apart. Linear interpolation.
_Z = [-3, -2.5, -2, -1.5, -1.4, -1.3, -1.2, -1.1, -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
      0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 2.5, 3]
_P = [0.944, 0.944, 0.944, 0.915, 0.906, 0.897, 0.887, 0.874, 0.856, 0.839, 0.814, 0.788, 0.755, 0.726,
      0.693, 0.649, 0.599, 0.562, 0.505, 0.431, 0.392, 0.339, 0.297, 0.260, 0.224, 0.194, 0.165, 0.140,
      0.121, 0.106, 0.087, 0.080, 0.067, 0.055, 0.025, 0.012, 0.006]


def macd_touch_forecast(close: pd.Series, tolerance: float = TOLERANCE) -> dict:
    """Forecast for the next daily close, computed from daily closes up to today.

    ``close`` must be daily closes, split-adjusted but NOT dividend-adjusted (yfinance
    auto_adjust=False), oldest first, at least ~200 bars (2 years recommended).
    """
    close = close.dropna().astype(float)
    if len(close) < MIN_BARS:
        return {"status": "insufficient_data"}

    ef = close.ewm(span=FAST, adjust=False).mean()
    es = close.ewm(span=SLOW, adjust=False).mean()
    macd = ef - es
    signal = macd.ewm(span=SIGNAL, adjust=False).mean()
    m, s, e_f, e_s, c = (float(x.iloc[-1]) for x in (macd, signal, ef, es, close))
    hist = m - s
    eps = tolerance * abs(s)
    gap_pct = -hist / abs(s) * 100 if s else float("nan")

    # MACD is linear in tomorrow's close, so these thresholds are exact:
    #   cross:  close' > (s - (1-a_f)*EMA12 + (1-a_s)*EMA26) / (a_f - a_s)
    #   touch:  close' >= cross price - eps / ((1-a_g) * (a_f - a_s))
    cross_price = (s - (1 - A_F) * e_f + (1 - A_S) * e_s) / (A_F - A_S)
    touch_price = cross_price - eps / ((1 - A_G) * (A_F - A_S))

    base = {
        "date": close.index[-1],
        "close": round(c, 4),
        "macd": round(m, 6),
        "signal": round(s, 6),
        "hist": round(hist, 6),
        "gap_pct_of_signal": round(gap_pct, 2),
        "touch_price": round(touch_price, 4),
        "cross_price": round(cross_price, 4),
    }
    if hist >= 0:
        return {**base, "status": "above_signal"}       # already crossed
    if hist >= -eps:
        return {**base, "status": "touching"}           # lines already touching today

    vol = float(close.pct_change().rolling(VOL_WINDOW).std().iloc[-1] * 100)
    required = (touch_price / c - 1) * 100
    z = required / vol
    prob = float(np.interp(z, _Z, _P))
    return {
        **base,
        "status": "apart",
        "required_change_pct": round(required, 2),     # negative = touch happens even on a drop this size
        "daily_vol_pct": round(vol, 2),
        "z": round(z, 3),
        "touch_prob": round(prob, 3),
        "signal_70": prob >= 0.70,
    }
