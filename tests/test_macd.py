import numpy as np
import pandas as pd

from screener.macd import bullish_cross, resample_close


def _series(values, freq="B"):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq=freq), dtype=float)


def test_cross_on_last_bar():
    # Long decline then a sharp rebound on the very last bar.
    values = list(np.linspace(100, 50, 80)) + [70]
    res = bullish_cross(_series(values), lookback=1)
    assert res.crossed
    assert res.bars_since_cross == 0
    assert res.hist > 0


def test_no_cross_while_falling():
    res = bullish_cross(_series(np.linspace(100, 50, 80)), lookback=1)
    assert not res.crossed
    assert res.hist < 0


def test_old_cross_outside_lookback():
    values = list(np.linspace(100, 50, 80)) + list(np.linspace(52, 80, 20))
    res = bullish_cross(_series(values), lookback=1)
    assert not res.crossed
    assert res.bars_since_cross > 0
    assert bullish_cross(_series(values), lookback=res.bars_since_cross + 1).crossed


def test_not_enough_history():
    assert bullish_cross(_series([1, 2, 3])) is None


def test_resample_keeps_and_drops_open_bar():
    # 2024-01-01 .. 2024-02-14: February is still open.
    daily = pd.Series(1.0, index=pd.bdate_range("2024-01-01", "2024-02-14"))
    assert len(resample_close(daily, "monthly")) == 2
    assert len(resample_close(daily, "monthly", closed_bars_only=True)) == 1
    # Ends on the last business day of January: bar is complete.
    daily = pd.Series(1.0, index=pd.bdate_range("2024-01-01", "2024-01-31"))
    assert len(resample_close(daily, "monthly", closed_bars_only=True)) == 1
