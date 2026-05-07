from __future__ import annotations

import holidays
import numpy as np
import pandas as pd


def add_calendar_features(idx: pd.DatetimeIndex, country: str = "US") -> pd.DataFrame:
    cal = holidays.country_holidays(country)
    df = pd.DataFrame(index=idx)
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["holiday_week"] = np.array(
        [int(any(d in cal for d in pd.date_range(i, periods=7, freq="D"))) for i in idx],
        dtype=np.int32,
    )
    return df


def supervised_frame_from_series(
    y: pd.Series,
    country: str = "US",
    lags: tuple[int, ...] = (1, 7, 30),
    rolling_windows: tuple[int, ...] = (7, 30),
) -> pd.DataFrame:
    y = y.astype(float).copy()
    feat = add_calendar_features(y.index, country=country)
    out = pd.concat([y.rename("y"), feat], axis=1)
    for lag in lags:
        out[f"lag_{lag}"] = y.shift(lag)
    for w in rolling_windows:
        out[f"roll_mean_{w}"] = y.shift(1).rolling(w, min_periods=1).mean()
        out[f"roll_std_{w}"] = y.shift(1).rolling(w, min_periods=2).std().fillna(0.0)
    return out.dropna()


def next_step_feature_row(
    hist: pd.Series,
    next_ts: pd.Timestamp,
    country: str = "US",
    lags: tuple[int, ...] = (1, 7, 30),
    rolling_windows: tuple[int, ...] = (7, 30),
) -> pd.Series:
    y = hist.astype(float).copy()
    row: dict[str, float | int] = {}
    cal = add_calendar_features(pd.DatetimeIndex([next_ts]), country=country).iloc[0]
    row["dow"] = int(cal["dow"])
    row["month"] = int(cal["month"])
    row["holiday_week"] = int(cal["holiday_week"])
    for lag in lags:
        row[f"lag_{lag}"] = float(y.iloc[-lag]) if len(y) >= lag else float(np.nan)
    for w in rolling_windows:
        tail = y.iloc[-w:]
        row[f"roll_mean_{w}"] = float(tail.mean())
        row[f"roll_std_{w}"] = float(tail.std()) if len(tail) > 1 else 0.0
    return pd.Series(row)
