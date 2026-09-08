# Weekly sales forecasting (case study)

**Demo video (Loom):** [Recording 1](https://www.loom.com/share/2a522b9251bb4e4790f35aefe62a7062) · [Recording 2](https://www.loom.com/share/e8867c02068746debbf6860feb7888f8)

End-to-end pipeline: load `Forecasting Case- Study.xlsx`, engineer lag / calendar / rolling features, compare **SARIMAX**, **Prophet**, **XGBoost** (recursive), and **LSTM**, pick the best model per state on a trailing validation window, then forecast **8 weeks** ahead.

## Setup

Use Python **3.10–3.12** if possible (Prophet and TensorFlow wheels may not be ready for the newest Python).

```bash
cd quickhyre
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Environment overrides (optional): `FORECAST_DATA_PATH`, `FORECAST_ARTIFACTS_DIR`, `FORECAST_HORIZON_WEEKS`, `FORECAST_VAL_WEEKS`, `FORECAST_WEEKLY_ANCHOR` (default `W-SUN`), `FORECAST_FAST_MODE=1` (faster, less accurate training).

## Train

From the project root (so `import forecasting` resolves):

```bash
python scripts/train.py
python scripts/train.py --data "Forecasting Case- Study.xlsx" --horizon 8 --val-weeks 8
```

**Why it can take a long time:** for each state the script fits **four** models (SARIMA, Prophet, XGBoost, LSTM). **LSTM with TensorFlow** is usually the bottleneck. Use **fast mode** while developing:

```bash
python scripts/train.py --fast
# or: set FORECAST_FAST_MODE=1
```

Full (non-fast) runs use more LSTM epochs and larger XGBoost models for better validation scores.

Writes `artifacts/manifest.json` with per-state chosen model, validation RMSEs, and 8 weekly point forecasts.

## Feature engineering

Implemented in `forecasting/features.py` and used by **XGBoost** and **LSTM** (tabular supervised learning). **SARIMAX** and **Prophet** use their own seasonal/trend structure rather than this matrix.

Sales are aggregated to **weekly** series (`W-SUN`); lag and rolling windows are therefore in **weeks**, not days.


| Requirement                         | Implementation                                                                                                                                                                                                                                                       |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Lags t−1, t−7, t−30**             | Columns `lag_1`, `lag_7`, `lag_30` via `shift` on the weekly target (`supervised_frame_from_series`). Same lags for recursive XGBoost steps via `next_step_feature_row`.                                                                                             |
| **Rolling mean / std**              | `roll_mean_7`, `roll_std_7`, `roll_mean_30`, `roll_std_30`. Rolling uses **`shift(1)`** before `rolling(...)` so the current week’s target is not included in its own rolling stats.                                                                 |
| **Day of week, month, holiday**     | `dow` (index weekday), `month`, `holiday_week` (US holidays touching the 7 calendar days from the week label), via `holidays` + `add_calendar_features`.                                                                                                             |
| **Train / validation (no leakage)** | `forecasting/splits.py`: last **`val_weeks`** observations are validation; all earlier data are train-only for scoring (`trainer.py` → `evaluate_models_on_val`). No shuffle. Final 8-week forecast refits the chosen model on the **full** history after selection. |


## API

From the project root:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

On some Windows setups binding to `0.0.0.0` or port `8000` triggers **WinError 10013** (socket permission). If that happens:

1. Use **127.0.0.1** and another port, e.g. **8080** or **8765** — `uvicorn api.main:app --host 127.0.0.1 --port 8765`
2. Omit **`--reload`** if the error persists (reload uses extra watchers on Windows) — same command as above without `--reload`.
3. Check whether Hyper‑V / Docker reserved a port range (PowerShell): `netsh interface ipv4 show excludedportrange protocol=tcp` — pick a port **outside** those ranges.

- `GET /health` — liveness  
- `GET /states` — states in the manifest  
- `GET /forecast/{state}` — manifest row for one state  
- `POST /train` — rebuild manifest (can be slow)

OpenAPI: `http://127.0.0.1:8000/docs` (change port if you used a different one)

## Excel columns

The loader auto-detects columns whose names suggest **date**, **state** (or region), and **sales** (or revenue / units / **Total**). Rename columns in the workbook if detection fails.

**Totals formatted as text** (e.g. `1,234.56` or `$1,234`) are normalized before parsing. If forecasts and RMSEs in `manifest.json` are all zeros, re-run training after pulling the latest code—your `Total` column may have been misread as NaN before this fix.

## For evaluators

Repository: [github.com/Meetamadhu/Demand-Forecasting-Pipeline](https://github.com/Meetamadhu/Demand-Forecasting-Pipeline)


| What to verify                  | How                                                                                                                                                                                                                |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Data & case brief**           | `Case - Study - Forecasting.pdf`, `Forecasting Case- Study.xlsx` in repo root.                                                                                                                                     |
| **Four models + selection**     | `forecasting/model_zoo.py` — SARIMAX, Prophet, XGBoost, LSTM; `pick_best` on validation RMSE; `forecasting/trainer.py` refits winner and writes 8-week horizon.                                                    |
| **Feature engineering**         | See table under **Feature engineering** above; code in `forecasting/features.py`.                                                                                                                                  |
| **No leakage**                  | `forecasting/splits.py` — strictly chronological train/val.                                                                                                                                                        |
| **API**                         | `api/main.py` — `GET /health`, `GET /states`, `GET /forecast/{state}`, `POST /train`. Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** after `uvicorn api.main:app --host 127.0.0.1 --port 8000`. |
| **Prebuilt outputs (optional)** | `artifacts/manifest.json` is committed so the API can be tested **without** retraining. To refresh: `python scripts/train.py`.                                                                                     |


## Demo video (submission)

Evaluators asked for a **short video** together with **code** and **documentation**. Typical submission bundle:

1. **GitHub repo** — [Meetamadhu/Demand-Forecasting-Pipeline](https://github.com/Meetamadhu/Demand-Forecasting-Pipeline) (README + PDF + Excel + code).
2. **Video** — hosted on **Loom** for this submission:
   - [Demo — part 1](https://www.loom.com/share/2a522b9251bb4e4790f35aefe62a7062)
   - [Demo — part 2](https://www.loom.com/share/e8867c02068746debbf6860feb7888f8)  
   *(You can also use YouTube Unlisted, Drive, or your course portal if required.)*

Links are repeated at the top of this README for quick access.



