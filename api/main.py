"""REST API serving cached per-state 8-week forecasts (train via CLI or POST /train)."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from forecasting.trainer import load_manifest, train_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Weekly sales forecasting", version="1.0.0")


class RootLinks(BaseModel):
    service: str
    docs: str
    health: str
    states: str
    forecast: str
    example: str
    train: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"status": "ok"}})

    status: str


class StatesResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"states": ["Alabama", "Texas"]}})

    states: list[str]


class ForecastWeek(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"week_start": "2024-01-07", "forecast_sales": 125000.5}})

    week_start: str
    forecast_sales: float


class ValidationRmse(BaseModel):
    """Per-model RMSE on the trailing validation window (lower is better)."""

    model_config = ConfigDict(extra="ignore")

    sarima: float | None = None
    prophet: float | None = None
    xgboost: float | None = None
    lstm: float | None = None


class ForecastResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "state": "Alabama",
                "best_model": "prophet",
                "validation_rmse": {"sarima": 120.0, "prophet": 95.0, "xgboost": 102.0, "lstm": 110.0},
                "forecast_weeks": [
                    {"week_start": "2024-01-07", "forecast_sales": 125000.5},
                    {"week_start": "2024-01-14", "forecast_sales": 126200.0},
                ],
            }
        }
    )

    state: str
    best_model: str
    validation_rmse: ValidationRmse
    forecast_weeks: list[ForecastWeek]


class TrainResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"states_trained": 43}})

    states_trained: int


@app.get("/", response_model=RootLinks)
def root() -> RootLinks:
    return RootLinks(
        service="Weekly sales forecasting API",
        docs="/docs",
        health="/health",
        states="/states",
        forecast="/forecast/{state}",
        example="/forecast/Alabama",
        train="POST /train",
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/states", response_model=StatesResponse)
def states() -> StatesResponse:
    m = load_manifest()
    return StatesResponse(states=sorted({str(row["state"]) for row in m}))


@app.get("/forecast/{state}", response_model=ForecastResponse)
def forecast_state(state: str) -> ForecastResponse:
    m = load_manifest()
    if not m:
        raise HTTPException(
            status_code=503,
            detail="No manifest found; run training first (POST /train or scripts/train.py).",
        )
    for row in m:
        if str(row["state"]).lower() == state.lower():
            weeks = [ForecastWeek(**w) for w in row["forecast_weeks"]]
            rmse_raw = row["validation_rmse"]
            return ForecastResponse(
                state=str(row["state"]),
                best_model=str(row["best_model"]),
                validation_rmse=ValidationRmse.model_validate(rmse_raw),
                forecast_weeks=weeks,
            )
    raise HTTPException(status_code=404, detail=f"Unknown state: {state}")


@app.post("/train", response_model=TrainResponse)
def train() -> TrainResponse:
    bundles = train_all()
    return TrainResponse(states_trained=len(bundles))
