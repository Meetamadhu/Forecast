from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip()).lower()


def infer_sales_column(columns: Iterable[str]) -> str:
    cols = list(columns)
    lowered = {_normalize_name(c): c for c in cols}
    for key in ("sales", "weekly_sales", "revenue", "amount", "units", "qty", "quantity", "total"):
        for lk, orig in lowered.items():
            if key in lk.replace(" ", "_"):
                return orig
    for lk, orig in lowered.items():
        if "sale" in lk or "revenue" in lk or lk == "total":
            return orig
    raise ValueError(f"Could not infer sales column from {cols}")


def infer_date_column(columns: Iterable[str]) -> str:
    cols = list(columns)
    lowered = {_normalize_name(c): c for c in cols}
    for token in ("date", "week", "period", "time", "ds"):
        for lk, orig in lowered.items():
            if token in lk:
                return orig
    raise ValueError(f"Could not infer date column from {cols}")


def infer_state_column(columns: Iterable[str]) -> str:
    cols = list(columns)
    lowered = {_normalize_name(c): c for c in cols}
    for token in ("state", "region", "province", "location"):
        for lk, orig in lowered.items():
            if token in lk:
                return orig
    raise ValueError(f"Could not infer state column from {cols}")


def coerce_sales_numeric(series: pd.Series) -> pd.Series:
    """Parse sales from Excel: commas/currency/accounting negatives often break pd.to_numeric."""
    s = series
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    t = s.astype(str).str.strip()
    t = t.str.replace(",", "", regex=False)
    t = t.str.replace("$", "", regex=False).str.replace("€", "", regex=False).str.replace("£", "", regex=False)
    t = t.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    return pd.to_numeric(t, errors="coerce")


def load_raw_sales(path: str | pd.PathLike, sheet: int | str = 0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    date_c = infer_date_column(df.columns)
    state_c = infer_state_column(df.columns)
    sales_c = infer_sales_column(df.columns)
    out = df[[date_c, state_c, sales_c]].copy()
    out.columns = ["date", "state", "sales"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["state"] = out["state"].astype(str).str.strip()
    out["sales"] = coerce_sales_numeric(out["sales"])
    out = out.dropna(subset=["date", "state", "sales"])
    # Sum duplicate state–date rows (e.g. multiple Category lines).
    out = out.groupby(["date", "state"], as_index=False)["sales"].sum()
    return out


def weekly_state_series(
    df: pd.DataFrame,
    freq: str = "W-SUN",
    sales_agg: str = "sum",
) -> dict[str, pd.Series]:
    """Aggregate to weekly series per state.

    Uses ``resample`` so bucket timestamps match ``pd.date_range(..., freq=freq)``.
    The previous ``to_period(...).start_time`` + ``reindex`` path could mis-align
    labels and yield all-NaN weeks (then imputed as zeros).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    states: dict[str, pd.Series] = {}
    for state, sub in df.groupby("state"):
        ts = sub.set_index("date")["sales"].sort_index()
        ts = ts.groupby(ts.index).sum()
        weekly = ts.resample(freq).agg(sales_agg).astype(float)
        if weekly.empty:
            continue
        # Normalize to midnight so labels match pd.date_range(..., freq=freq); otherwise
        # reindex can miss every row (all NaN → imputed zeros).
        idx = pd.DatetimeIndex(weekly.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        weekly = weekly.set_axis(idx.normalize(), axis=0)
        full_idx = pd.date_range(weekly.index.min(), weekly.index.max(), freq=freq)
        s = weekly.reindex(full_idx).astype(float)
        s.index.name = "week"
        s.name = "sales"
        states[state] = s
    return states


def impute_series(s: pd.Series) -> pd.Series:
    x = s.astype(float).copy()
    x = x.interpolate(limit_direction="both")
    x = x.ffill().bfill()
    x = x.fillna(0.0)
    return x
