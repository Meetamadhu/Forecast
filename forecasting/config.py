from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FORECAST_", env_file=".env", extra="ignore")

    data_path: Path = Path(__file__).resolve().parents[1] / "Forecasting Case- Study.xlsx"
    artifacts_dir: Path = Path(__file__).resolve().parents[1] / "artifacts"
    horizon_weeks: int = 8
    val_weeks: int = 8
    weekly_anchor: str = "W-SUN"
    random_seed: int = 42
    # Fewer LSTM epochs / smaller XGB — much faster; use full runs for final submission quality.
    fast_mode: bool = False


settings = Settings()
