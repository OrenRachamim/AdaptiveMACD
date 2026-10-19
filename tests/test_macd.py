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


def test_macd_at_cross_sign():
    # Rebound after a long decline: cross happens with MACD still below zero.
    below = bullish_cross(_series(list(np.linspace(100, 50, 80)) + [70]), lookback=1)
    assert below.crossed and below.macd_at_cross < 0
    # Uptrend, small pullback, resumption: cross happens with MACD above zero.
    values = list(np.linspace(50, 100, 80)) + list(np.linspace(99, 96, 6)) + [110]
    above = bullish_cross(_series(values), lookback=1)
    assert above.crossed and above.macd_at_cross > 0


def test_whipsaw_detected():
    # Rebound crosses up, dips back below, then crosses up again.
    values = list(np.linspace(100, 50, 80)) + [70, 40, 35, 80]
    res = bullish_cross(_series(values), lookback=5)
    assert res.crossed and res.whipsaw
    assert not bullish_cross(_series(list(np.linspace(100, 50, 80)) + [70]), lookback=5).whipsaw


def test_tradingview_url_and_message():
    from screener.notify_telegram import build_messages, tradingview_url

    assert tradingview_url("BRK-B", "NYSE") == "https://www.tradingview.com/chart/?symbol=NYSE:BRK.B"
    assert tradingview_url("XYZ", "NYSE American").endswith("symbol=AMEX:XYZ")
    result = pd.DataFrame([{
        "ticker": "AVY", "name": "Avery Dennison", "exchange": "NYSE", "close": 175.31,
        "avg_volume_50d": 880154, "monthly_cross_bar": "2026-10-31", "daily_cross_bar": "2026-10-13",
    }])
    (msg,) = build_messages(result, "2026-10-16")
    assert "NYSE:AVY" in msg and "10/2026" in msg and "13/10" in msg
    assert "לא נמצאו" in build_messages(pd.DataFrame(), "2026-10-16")[0]


def test_cross_price_is_exact_on_every_timeframe():
    from screener.cross_forecast import daily_features, higher_features
    from screener.macd import macd

    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 900))),
                      index=pd.bdate_range("2020-01-01", periods=900))
    day = close.index[700]
    # Daily: tomorrow's close at the cross price puts MACD exactly on the signal line.
    price = daily_features(close).at[day, "cross_price"]
    nxt = pd.concat([close[:day], pd.Series([price], index=[close.index[701]])])
    assert abs(macd(nxt)["hist"].iloc[-1]) < 1e-9
    # Weekly/monthly: the current bar closing at the cross price does the same.
    for tf in ("weekly", "monthly"):
        price = higher_features(close, tf).at[day, "cross_price"]
        bars = resample_close(close[:day], tf)
        bars.iloc[-1] = price
        assert abs(macd(bars)["hist"].iloc[-1]) < 1e-9


def test_calibration_is_monotone_and_decided_bars_are_certain():
    from screener.cross_forecast import Calibration, z_score

    rng = np.random.default_rng(2)
    z = rng.normal(0, 1.5, 20000)
    crossed = rng.random(20000) < 1 / (1 + np.exp(2 * z))
    cal = Calibration(z, crossed)
    assert np.all(np.diff(cal.p) <= 1e-12)
    assert cal(np.array([-3.0]))[0] > 0.9 and cal(np.array([3.0]))[0] < 0.1
    zz = z_score([-1.0, 1.0], [2.0, 2.0], [0, 0])
    assert list(cal(zz)) == [1.0, 0.0]


def test_forecast_message():
    from screener.notify_telegram import build_forecast_messages

    result = pd.DataFrame([{
        "ticker": "AVY", "name": "Avery Dennison", "exchange": "NYSE", "close": 175.31,
        "cross_price": 174.1, "required_change_pct": -0.69, "cross_prob": 0.78,
    }])
    (msg,) = build_forecast_messages(result, "2026-10-16")
    assert "NYSE:AVY" in msg and "$174.1" in msg and "-0.69%" in msg and "78%" in msg
    assert "אין מניות" in build_forecast_messages(pd.DataFrame(), "2026-10-16")[0]
