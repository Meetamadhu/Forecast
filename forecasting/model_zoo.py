from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from forecasting.features import next_step_feature_row, supervised_frame_from_series

logger = logging.getLogger(__name__)


def rmse(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(np.asarray(a), np.asarray(b))))


def _seasonal_period(n: int) -> int:
    if n >= 110:
        return 52
    if n >= 40:
        return 13
    return 4


@dataclass
class SarimaModel:
    fitted: object | None = None
    order: tuple[int, int, int] = (1, 1, 1)

    def fit(self, y: pd.Series) -> None:
        m = _seasonal_period(len(y))
        seasonal_order = (1, 1, 1, m) if len(y) >= 2 * m + 8 else (0, 0, 0, 0)
        model = SARIMAX(
            y.astype(float),
            order=self.order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.fitted = model.fit(disp=False)

    def forecast_steps(self, steps: int) -> np.ndarray:
        assert self.fitted is not None
        return np.asarray(self.fitted.forecast(steps=steps))


class ProphetModel:
    def __init__(self) -> None:
        self._m = None

    def fit(self, y: pd.Series) -> None:
        from forecasting.quiet_env import apply_prophet_log_quiet
        from prophet import Prophet

        apply_prophet_log_quiet()

        ds = pd.to_datetime(y.index).strftime("%Y-%m-%d")
        df = pd.DataFrame({"ds": ds, "y": y.values.astype(float)})
        self._m = Prophet(weekly_seasonality=True, yearly_seasonality=len(y) >= 53 * 2)
        self._m.fit(df)

    def forecast_index(self, future_index: pd.DatetimeIndex) -> np.ndarray:
        assert self._m is not None
        fut = pd.DataFrame({"ds": pd.to_datetime(future_index).strftime("%Y-%m-%d")})
        fc = self._m.predict(fut)
        return fc["yhat"].values.astype(float)


class XgbLagModel:
    def __init__(self, seed: int = 42, n_estimators: int = 300) -> None:
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=-1,
        )
        self.feature_columns: list[str] | None = None

    def fit(self, y: pd.Series, country: str = "US") -> None:
        frame = supervised_frame_from_series(y, country=country)
        self.feature_columns = [c for c in frame.columns if c != "y"]
        self.model.fit(frame[self.feature_columns], frame["y"])

    def recursive_forecast(self, hist: pd.Series, future_index: pd.DatetimeIndex, country: str = "US") -> np.ndarray:
        assert self.feature_columns is not None
        series = hist.astype(float).copy()
        preds: list[float] = []
        for ts in future_index:
            row = next_step_feature_row(series, ts, country=country).reindex(self.feature_columns).astype(float)
            row = row.fillna(0.0)
            yhat = float(self.model.predict(row.values.reshape(1, -1))[0])
            preds.append(yhat)
            series.loc[ts] = yhat
        return np.asarray(preds)


class LstmModel:
    def __init__(self, lookback: int = 12, seed: int = 42) -> None:
        self.lookback = lookback
        self.seed = seed
        self.feat_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self._keras_model = None
        self._feature_cols: list[str] = []

    def _build(self, n_features: int):
        import tensorflow as tf

        tf.random.set_seed(self.seed)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.lookback, n_features)),
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="mse")
        return model

    def fit(self, y: pd.Series, country: str = "US", epochs: int = 80, batch_size: int = 16) -> None:
        frame = supervised_frame_from_series(y, country=country)
        self._feature_cols = [c for c in frame.columns if c != "y"]
        Xraw = frame[self._feature_cols].values.astype(np.float32)
        yraw = frame["y"].values.astype(np.float32).reshape(-1, 1)
        Xs = self.feat_scaler.fit_transform(Xraw)
        ys = self.y_scaler.fit_transform(yraw).ravel()
        lb = self.lookback
        if len(Xs) <= lb + 2:
            raise ValueError("Not enough rows for LSTM")
        X_seq = np.stack([Xs[i - lb : i] for i in range(lb, len(Xs))], axis=0)
        y_seq = ys[lb:]
        self._keras_model = self._build(X_seq.shape[2])
        self._keras_model.fit(
            X_seq,
            y_seq,
            epochs=epochs,
            batch_size=min(batch_size, len(X_seq)),
            verbose=0,
            shuffle=False,
        )

    def recursive_forecast(self, hist: pd.Series, future_index: pd.DatetimeIndex, country: str = "US") -> np.ndarray:
        assert self._keras_model is not None
        series = hist.astype(float).copy()
        preds: list[float] = []
        for ts in future_index:
            tmp = supervised_frame_from_series(series, country=country)
            if len(tmp) < self.lookback:
                row_block = tmp[self._feature_cols].values.astype(np.float32)
                pad_rows = self.lookback - len(row_block)
                row_block = np.vstack([np.tile(row_block[0], (pad_rows, 1)), row_block])
            else:
                row_block = tmp[self._feature_cols].iloc[-self.lookback :].values.astype(np.float32)
            row_scaled = self.feat_scaler.transform(row_block).reshape(1, self.lookback, -1)
            yhat_s = float(self._keras_model.predict(row_scaled, verbose=0).reshape(-1)[0])
            yhat = float(self.y_scaler.inverse_transform(np.array([[yhat_s]]))[0, 0])
            preds.append(yhat)
            series.loc[ts] = yhat
        return np.asarray(preds)


def _lstm_epochs(train_len: int, *, fast: bool, for_final: bool) -> int:
    """Keep validation cheap; final fit can be a bit longer."""
    if fast:
        if for_final:
            return min(45, max(20, train_len // 4))
        return min(22, max(12, train_len // 6))
    if for_final:
        return min(120, max(40, train_len))
    return min(100, max(28, train_len // 2))


def evaluate_models_on_val(
    y_train: pd.Series,
    y_val: pd.Series,
    country: str = "US",
    seed: int = 42,
    fast: bool = False,
    xgb_trees: int | None = None,
) -> dict[str, float]:
    scores: dict[str, float] = {}

    try:
        sm = SarimaModel()
        sm.fit(y_train)
        p = sm.forecast_steps(len(y_val))
        scores["sarima"] = rmse(y_val.values, p)
    except Exception as e:
        logger.warning("SARIMA failed: %s", e)
        scores["sarima"] = np.inf

    try:
        pm = ProphetModel()
        pm.fit(y_train)
        p = pm.forecast_index(y_val.index)
        scores["prophet"] = rmse(y_val.values, p)
    except Exception as e:
        logger.warning("Prophet failed: %s", e)
        scores["prophet"] = np.inf

    try:
        trees = xgb_trees if xgb_trees is not None else (120 if fast else 300)
        xm = XgbLagModel(seed=seed, n_estimators=trees)
        xm.fit(y_train, country=country)
        p = xm.recursive_forecast(y_train, y_val.index, country=country)
        scores["xgboost"] = rmse(y_val.values, p)
    except Exception as e:
        logger.warning("XGBoost failed: %s", e)
        scores["xgboost"] = np.inf

    try:
        lm = LstmModel(seed=seed)
        lm.fit(y_train, country=country, epochs=_lstm_epochs(len(y_train), fast=fast, for_final=False))
        p = lm.recursive_forecast(y_train, y_val.index, country=country)
        scores["lstm"] = rmse(y_val.values, p)
    except Exception as e:
        logger.warning("LSTM failed: %s", e)
        scores["lstm"] = np.inf

    return scores


def pick_best(scores: dict[str, float]) -> str:
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not finite:
        return "sarima"
    return min(finite, key=finite.get)


def fit_best_and_forecast(
    y_full_imputed: pd.Series,
    best_name: str,
    horizon: int,
    country: str = "US",
    seed: int = 42,
    freq: str = "W-SUN",
    fast: bool = False,
    xgb_trees: int | None = None,
) -> tuple[pd.Series, object]:
    last = y_full_imputed.index[-1]
    future_index = pd.date_range(last + pd.Timedelta(days=7), periods=horizon, freq=freq)

    if best_name == "sarima":
        m = SarimaModel()
        m.fit(y_full_imputed)
        pred = m.forecast_steps(horizon)
        return pd.Series(pred, index=future_index), m

    if best_name == "prophet":
        m = ProphetModel()
        m.fit(y_full_imputed)
        pred = m.forecast_index(future_index)
        return pd.Series(pred, index=future_index), m

    if best_name == "xgboost":
        trees = xgb_trees if xgb_trees is not None else (120 if fast else 300)
        m = XgbLagModel(seed=seed, n_estimators=trees)
        m.fit(y_full_imputed, country=country)
        pred = m.recursive_forecast(y_full_imputed, future_index, country=country)
        return pd.Series(pred, index=future_index), m

    if best_name == "lstm":
        m = LstmModel(seed=seed)
        m.fit(
            y_full_imputed,
            country=country,
            epochs=_lstm_epochs(len(y_full_imputed), fast=fast, for_final=True),
        )
        pred = m.recursive_forecast(y_full_imputed, future_index, country=country)
        return pd.Series(pred, index=future_index), m

    raise ValueError(best_name)
