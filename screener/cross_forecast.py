"""Forecast whether MACD will cross above its signal line at the close of the current bar.

MACD is linear in the bar's close, so for every bar there is an exact close above
which the MACD line ends above the signal line. The question "will it cross?"
becomes "will the bar close above that price?". The required move is scaled by
the stock's volatility over the time left until the bar closes:

    z = required_rise_% / (daily_vol_% * sqrt(trading days left in the bar))

and z is mapped to a probability with a calibration table fitted on history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.macd import FAST, SIGNAL, SLOW

A_F, A_S = 2 / (FAST + 1), 2 / (SLOW + 1)
PERIODS = {"weekly": "W-FRI", "monthly": "M"}


def _threshold(ef_prev, es_prev, sig_prev):
    """Close that puts the new MACD exactly on the new signal line, given the previous bar's EMAs."""
    return (sig_prev - (1 - A_F) * ef_prev + (1 - A_S) * es_prev) / (A_F - A_S)


def daily_features(close: pd.Series) -> pd.DataFrame:
    """Per day: histogram, close needed tomorrow to cross, and whether it crossed tomorrow."""
    ef = close.ewm(span=FAST, adjust=False).mean()
    es = close.ewm(span=SLOW, adjust=False).mean()
    m = ef - es
    sig = m.ewm(span=SIGNAL, adjust=False).mean()
    hist = m - sig
    price = _threshold(ef, es, sig)
    return pd.DataFrame({
        "hist": hist,
        "macd": m,
        "cross_price": price,
        "rise_pct": (price / close - 1) * 100,
        "days_left": 1,
        "crossed": hist.shift(-1) > 0,
        "bar_no": np.arange(len(close)),
    }, index=close.index)


def daily_touch_features(close: pd.Series, tolerance: float = 0.05) -> pd.DataFrame:
    """Per day: is MACD still clearly below its signal line, and the close needed tomorrow to touch it.

    The lines "touch" when the gap (signal - MACD) is at most ``tolerance`` x |signal|. ``apart`` marks
    days on which they do not touch yet; ``touched`` whether they touched or crossed on the next bar.
    The gap scale is today's |signal|, so the touch price stays exact (MACD is linear in the close).
    """
    a_g = 2 / (SIGNAL + 1)
    ef = close.ewm(span=FAST, adjust=False).mean()
    es = close.ewm(span=SLOW, adjust=False).mean()
    m = ef - es
    sig = m.ewm(span=SIGNAL, adjust=False).mean()
    hist = m - sig
    eps = tolerance * sig.abs()
    # hist' = (1 - a_g) * (macd' - sig) >= -eps  <=>  close' >= cross price - eps / ((1 - a_g) * (A_F - A_S))
    price = _threshold(ef, es, sig) - eps / ((1 - a_g) * (A_F - A_S))
    return pd.DataFrame({
        "hist": hist,
        "macd": m,
        "signal": sig,
        "apart": hist < -eps,
        "touch_price": price,
        "rise_pct": (price / close - 1) * 100,
        "touched": hist.shift(-1) >= -eps,
        "bar_no": np.arange(len(close)),
    }, index=close.index)


def higher_features(close: pd.Series, timeframe: str) -> pd.DataFrame:
    """Per day: state of the current weekly/monthly bar and the bar close needed to cross on it.

    ``prev_hist`` is the histogram of the last completed bar (<= 0 means not crossed yet),
    ``crossed`` whether the current bar really closed with MACD above the signal line.
    """
    per = close.index.to_period(PERIODS[timeframe])
    bar_close = close.groupby(per).last()
    ef = bar_close.ewm(span=FAST, adjust=False).mean()
    es = bar_close.ewm(span=SLOW, adjust=False).mean()
    m = ef - es
    sig = m.ewm(span=SIGNAL, adjust=False).mean()
    hist = m - sig

    def prev(s: pd.Series) -> np.ndarray:
        return s.shift(1).reindex(per).to_numpy()

    price = _threshold(prev(ef), prev(es), prev(sig))
    grp = close.groupby(per)
    left = (grp.transform("size") - grp.cumcount() - 1).to_numpy()
    return pd.DataFrame({
        "prev_hist": prev(hist),
        "cross_price": price,
        "rise_pct": (price / close.to_numpy() - 1) * 100,
        "days_left": left,
        "crossed": hist.reindex(per).to_numpy() > 0,
        "bar_no": pd.Series(np.arange(len(bar_close)), index=bar_close.index).reindex(per).to_numpy(),
    }, index=close.index)


def daily_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Standard deviation of daily % returns."""
    return close.pct_change().rolling(window).std() * 100


def z_score(rise_pct, vol_pct, days_left):
    """Required rise in units of the volatility left until the bar closes (+-inf when already decided)."""
    rise_pct, vol_pct, days_left = (np.asarray(a, dtype=float) for a in (rise_pct, vol_pct, days_left))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = rise_pct / (vol_pct * np.sqrt(days_left))
    decided = days_left <= 0
    z[decided] = np.where(rise_pct[decided] <= 0, -np.inf, np.inf)
    return z


class Calibration:
    """Monotone map from z to the observed probability of a cross (isotonic regression on z bins)."""

    def __init__(self, z: np.ndarray, crossed: np.ndarray, bins: int = 200):
        finite = np.isfinite(z)
        z, y = z[finite], crossed[finite].astype(float)
        edges = np.unique(np.quantile(z, np.linspace(0, 1, bins + 1)))
        idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, len(edges) - 2)
        cnt = np.bincount(idx, minlength=len(edges) - 1).astype(float)
        hit = np.bincount(idx, weights=y, minlength=len(edges) - 1)
        centers = np.bincount(idx, weights=z, minlength=len(edges) - 1) / np.maximum(cnt, 1)
        keep = cnt > 0
        self.z = centers[keep]
        self.p = _pav_decreasing(hit[keep] / cnt[keep], cnt[keep])

    @classmethod
    def from_table(cls, table: pd.DataFrame) -> "Calibration":
        obj = cls.__new__(cls)
        obj.z = table["z"].to_numpy(dtype=float)
        obj.p = table["probability"].to_numpy(dtype=float)
        return obj

    def __call__(self, z) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        p = np.interp(z, self.z, self.p, left=self.p[0], right=self.p[-1])
        p[np.isneginf(z)] = 1.0
        p[np.isposinf(z)] = 0.0
        p[np.isnan(z)] = np.nan
        return p

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"z": self.z, "probability": self.p})


def _pav_decreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: closest non-increasing sequence (weighted least squares)."""
    v, w, n = [], [], []
    for val, wt in zip(values, weights):
        v.append(val), w.append(wt), n.append(1)
        while len(v) > 1 and v[-2] < v[-1]:
            tw = w[-2] + w[-1]
            v[-2] = (v[-2] * w[-2] + v[-1] * w[-1]) / tw
            w[-2], n[-2] = tw, n[-2] + n[-1]
            v.pop(), w.pop(), n.pop()
    return np.repeat(v, n)
