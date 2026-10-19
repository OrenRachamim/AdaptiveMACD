"""Find the probability thresholds that best predict an upcoming MACD cross.

For each set of timeframes (daily only, daily + weekly, daily + monthly) the
event to predict is "every timeframe in the set crosses": the daily MACD at
tomorrow's close and the weekly/monthly MACD at the close of the current bar.
Each timeframe gets a calibrated cross probability; a grid of thresholds is
searched on a training period and the chosen thresholds are then checked on a
later test period that the search never saw.

Usage:
    python -m screener.optimize_cross --cache-dir <backtest price cache> --out-dir results_cross
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from screener.cross_forecast import Calibration, daily_features, daily_vol, higher_features, z_score

log = logging.getLogger("optimize_cross")

SETS = {"daily": ["daily"], "daily+weekly": ["daily", "weekly"], "daily+monthly": ["daily", "monthly"]}
VOL_WINDOWS = (10, 20, 60)
GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def build_dataset(cache_dir: Path, sample: float, min_price: float, min_avg_volume: float,
                  seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    files = sorted(glob.glob(str(cache_dir / "*.pkl")))
    for n, path in enumerate(files, 1):
        data = pd.read_pickle(path)
        for ticker in data.columns.get_level_values(0).unique():
            df = data[ticker].dropna(subset=["Close"])
            if len(df) < 800:
                continue
            close = df["Close"]
            d = daily_features(close)
            row = pd.DataFrame({"ticker": ticker, "d_rise": d["rise_pct"], "d_crossed": d["crossed"]})
            for w in VOL_WINDOWS:
                row[f"vol{w}"] = daily_vol(close, w)
            for tf in ("weekly", "monthly"):
                h = higher_features(close, tf)
                row[f"{tf[0]}_prev"] = h["prev_hist"]
                row[f"{tf[0]}_rise"] = h["rise_pct"]
                row[f"{tf[0]}_left"] = h["days_left"]
                row[f"{tf[0]}_crossed"] = h["crossed"]
                row[f"{tf[0]}_ok"] = h["bar_no"] >= 40
            eligible = (close >= min_price) & (df["Volume"].rolling(50).mean() >= min_avg_volume)
            keep = (d["hist"] < 0) & (d["bar_no"] >= 100) & eligible
            keep.iloc[-30:] = False  # outcome of the still-open bars is unknown
            keep &= rng.random(len(keep)) < sample
            parts.append(row[keep.to_numpy()])
        log.info("%d/%d price files", n, len(files))
    out = pd.concat(parts)
    out.index.name = "date"
    return out.reset_index().dropna(subset=["vol60"])


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Probability that a random positive is ranked above a random negative."""
    ranks = pd.Series(score).rank().to_numpy()
    y = y.astype(bool)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tf_columns(tf: str) -> tuple[str, str, str]:
    if tf == "daily":
        return "d_rise", "d_crossed", ""
    return f"{tf[0]}_rise", f"{tf[0]}_crossed", f"{tf[0]}_left"


def tf_mask(df: pd.DataFrame, tf: str) -> pd.Series:
    """Rows where the timeframe has not crossed yet and the outcome is still uncertain."""
    if tf == "daily":
        return pd.Series(True, index=df.index)
    k = tf[0]
    return (df[f"{k}_prev"] <= 0) & df[f"{k}_ok"] & (df[f"{k}_left"] > 0)


def tf_z(df: pd.DataFrame, tf: str, window: int) -> np.ndarray:
    rise, _, left = tf_columns(tf)
    days = 1 if tf == "daily" else df[left]
    return z_score(df[rise], df[f"vol{window}"], days)


def metrics(signal: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    n = int(signal.sum())
    tp = int((signal & y).sum())
    precision = tp / n if n else np.nan
    recall = tp / y.sum() if y.sum() else np.nan
    f1 = 2 * precision * recall / (precision + recall) if n and tp else 0.0
    return precision, recall, f1, n


def optimize(df: pd.DataFrame, train_end: str, sample: float, out_dir: Path) -> str:
    train = df["date"] <= train_end
    lines = [f"# MACD cross forecast – optimal thresholds", "",
             f"Rows: {len(df):,} stock-days ({sample:.0%} random sample, MACD below signal). "
             f"Train: up to {train_end}, test: after.", ""]

    # 1. Volatility window and calibration per timeframe (fitted on the training period only).
    calib, window, prob = {}, {}, {}
    lines += ["## Volatility window (AUC on train)", "", "| timeframe | " + " | ".join(f"{w}d" for w in VOL_WINDOWS)
              + " | chosen |", "|---|" + "---:|" * (len(VOL_WINDOWS) + 1)]
    for tf in ("daily", "weekly", "monthly"):
        mask = tf_mask(df, tf)
        _, crossed, _ = tf_columns(tf)
        scores = {}
        for w in VOL_WINDOWS:
            z = tf_z(df[mask & train], tf, w)
            fin = np.isfinite(z)
            scores[w] = auc(-z[fin], df.loc[mask & train, crossed].to_numpy()[fin])
        window[tf] = max(scores, key=scores.get)
        lines.append(f"| {tf} | " + " | ".join(f"{scores[w]:.4f}" for w in VOL_WINDOWS) + f" | {window[tf]}d |")
        z_all = tf_z(df, tf, window[tf])
        calib[tf] = Calibration(z_all[(mask & train).to_numpy()], df.loc[mask & train, crossed].to_numpy())
        p = np.full(len(df), np.nan)
        p[mask.to_numpy()] = calib[tf](z_all[mask.to_numpy()])
        prob[tf] = p
        calib[tf].table().to_csv(out_dir / f"calibration_{tf}.csv", index=False)
        test_mask = (mask & ~train).to_numpy()
        lines_tf_auc = auc(prob[tf][test_mask], df.loc[mask & ~train, crossed].to_numpy())
        log.info("%s: window %dd, test AUC %.4f", tf, window[tf], lines_tf_auc)
    lines.append("")

    # 2. Calibration check on the test period.
    lines += ["## Calibration on the test period (predicted vs. actual cross rate)", "",
              "| predicted | " + " | ".join(("daily", "weekly", "monthly")) + " |", "|---|---:|---:|---:|"]
    bands = [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]
    for lo, hi in bands:
        cells = []
        for tf in ("daily", "weekly", "monthly"):
            _, crossed, _ = tf_columns(tf)
            m = (~train).to_numpy() & (prob[tf] >= lo) & (prob[tf] < hi)
            cells.append(f"{df[crossed].to_numpy()[m].mean() * 100:.1f}% (n={m.sum():,})" if m.sum() else "–")
        lines.append(f"| {lo:.0%}–{min(hi, 1):.0%} | " + " | ".join(cells) + " |")
    lines.append("")

    # 3. Threshold search per set.
    summary = []
    test_days = df.loc[~train, "date"].nunique()
    for name, tfs in SETS.items():
        rows = np.ones(len(df), bool)
        for tf in tfs:
            rows &= tf_mask(df, tf).to_numpy()
        y = np.ones(len(df), bool)
        for tf in tfs:
            y &= df[tf_columns(tf)[1]].to_numpy().astype(bool)
        tr, te = rows & train.to_numpy(), rows & ~train.to_numpy()
        grid = [(t,) for t in GRID] if len(tfs) == 1 else [(a, b) for a in GRID for b in GRID]
        results = []
        for th in grid:
            sig = np.ones(len(df), bool)
            for tf, t in zip(tfs, th):
                sig &= prob[tf] >= t
            p_tr, r_tr, f_tr, n_tr = metrics(sig[tr], y[tr])
            p_te, r_te, f_te, n_te = metrics(sig[te], y[te])
            results.append(dict(set=name, **{f"th_{tf}": t for tf, t in zip(tfs, th)},
                                train_precision=p_tr, train_recall=r_tr, train_f1=f_tr, train_n=n_tr,
                                test_precision=p_te, test_recall=r_te, test_f1=f_te, test_n=n_te,
                                signals_per_day=n_te / sample / test_days))
        res = pd.DataFrame(results)
        res.to_csv(out_dir / f"grid_{name.replace('+', '_')}.csv", index=False)
        base_tr, base_te = y[tr].mean(), y[te].mean()
        lines += [f"## {name}", "",
                  f"Base rate of the event (all {'timeframes cross' if len(tfs) > 1 else 'daily cross tomorrow'}): "
                  f"train {base_tr:.1%}, test {base_te:.1%} of {te.sum():,} test stock-days.", "",
                  "| choice | " + " | ".join(f"{tf} threshold" for tf in tfs)
                  + " | precision (test) | recall (test) | F1 (test) | signals/day (market) |",
                  "|---|" + "---:|" * (len(tfs) + 4)]
        picks = [("best F1", res.loc[res["train_f1"].idxmax()])]
        for target in (0.6, 0.7, 0.8, 0.9):
            ok = res[(res["train_precision"] >= target) & (res["train_n"] >= 100)]
            if len(ok):
                picks.append((f"precision ≥ {target:.0%}", ok.loc[ok["train_recall"].idxmax()]))
        for label, r in picks:
            lines.append(f"| {label} | " + " | ".join(f"{r[f'th_{tf}']:.0%}" for tf in tfs)
                         + f" | {r['test_precision']:.1%} | {r['test_recall']:.1%} | {r['test_f1']:.3f}"
                         f" | {r['signals_per_day']:.1f} |")
            summary.append(dict(set=name, choice=label, **{f"th_{tf}": r[f"th_{tf}"] for tf in tfs},
                                test_precision=r["test_precision"], test_recall=r["test_recall"],
                                test_f1=r["test_f1"], signals_per_day=r["signals_per_day"]))
        lines.append("")
    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame({"timeframe": list(window), "vol_window": list(window.values())}).to_csv(
        out_dir / "vol_windows.csv", index=False)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=Path, required=True, help="price cache written by screener.backtest")
    p.add_argument("--out-dir", type=Path, default=Path("backtest_results/cross_forecast"))
    p.add_argument("--sample", type=float, default=0.3, help="fraction of stock-days to keep")
    p.add_argument("--train-end", default="2014-12-31")
    p.add_argument("--min-price", type=float, default=1.0)
    p.add_argument("--min-avg-volume", type=int, default=50_000)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = args.out_dir / "dataset.pkl"
    if dataset.exists():
        df = pd.read_pickle(dataset)
    else:
        df = build_dataset(args.cache_dir, args.sample, args.min_price, args.min_avg_volume)
        df.to_pickle(dataset)
    log.info("%d rows", len(df))
    report = optimize(df, args.train_end, args.sample, args.out_dir)
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
