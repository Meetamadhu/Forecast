from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from forecasting.config import Settings, settings
from forecasting.io import impute_series, load_raw_sales, weekly_state_series
from forecasting.quiet_env import configure_quiet_training
from forecasting.model_zoo import evaluate_models_on_val, fit_best_and_forecast, pick_best
from forecasting.splits import temporal_split

logger = logging.getLogger(__name__)


@dataclass
class StateForecastBundle:
    state: str
    best_model: str
    validation_rmse: dict[str, float]
    forecast_weeks: list[dict[str, object]]


def train_all(cfg: Settings | None = None) -> list[StateForecastBundle]:
    cfg = cfg or settings
    configure_quiet_training()
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_sales(cfg.data_path)
    if len(raw) == 0:
        raise ValueError("No rows with valid date, state, and numeric sales after parsing the Excel file.")
    pos = (raw["sales"] > 0).sum()
    if pos == 0:
        logger.warning(
            "All parsed sales values are zero or negative; forecasts may be useless. "
            "Check the Total/sales column format in Excel."
        )
    states_series = weekly_state_series(raw, freq=cfg.weekly_anchor)
    to_train: list[tuple[str, pd.Series]] = []
    for state, series in sorted(states_series.items()):
        y = impute_series(series)
        if len(y) > cfg.val_weeks + 10:
            to_train.append((state, y))
    n_states = len(to_train)
    short = len(states_series) - n_states
    if short:
        logger.warning("Skipping %s state(s): weekly series too short for validation window.", short)
    if cfg.fast_mode:
        logger.info("Fast mode: fewer LSTM epochs and XGB trees (set FORECAST_FAST_MODE=0 for full quality).")
    logger.info("Training %s state(s) × 4 models; LSTM+TensorFlow is usually the slowest step.", n_states)

    bundles: list[StateForecastBundle] = []
    for i, (state, y) in enumerate(to_train, start=1):
        t_state = time.perf_counter()
        train_sl, val_sl = temporal_split(len(y), cfg.val_weeks)
        y_train = y.iloc[train_sl]
        y_val = y.iloc[val_sl]

        logger.info("(%d/%d) %s — fitting SARIMA, Prophet, XGBoost, LSTM (validation)…", i, n_states, state)
        scores = evaluate_models_on_val(
            y_train,
            y_val,
            country="US",
            seed=cfg.random_seed,
            fast=cfg.fast_mode,
        )
        best = pick_best(scores)
        fc, _ = fit_best_and_forecast(
            y,
            best_name=best,
            horizon=cfg.horizon_weeks,
            country="US",
            seed=cfg.random_seed,
            freq=cfg.weekly_anchor,
            fast=cfg.fast_mode,
        )

        forecast_weeks = [
            {"week_start": ix.strftime("%Y-%m-%d"), "forecast_sales": float(v)}
            for ix, v in fc.items()
        ]
        bundles.append(
            StateForecastBundle(
                state=state,
                best_model=best,
                validation_rmse={k: (float(v) if math.isfinite(float(v)) else None) for k, v in scores.items()},
                forecast_weeks=forecast_weeks,
            )
        )
        logger.info("(%d/%d) %s done in %.1fs — best=%s", i, n_states, state, time.perf_counter() - t_state, best)

    manifest_path = cfg.artifacts_dir / "manifest.json"
    payload = [asdict(b) for b in bundles]
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%s states)", manifest_path, len(bundles))
    return bundles


def load_manifest(path: Path | None = None) -> list[dict]:
    p = path or settings.artifacts_dir / "manifest.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))
