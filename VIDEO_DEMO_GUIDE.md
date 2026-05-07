# Demo video guide — what to show + what to say

Use this as **speaker notes** while recording. Approx. **6–8 minutes** total.

---

## 1. Opening — problem & deliverables (~45 s)

### On screen
- GitHub repo **README** (top): title + first paragraph, or assignment PDF first page + repo tab **Meetamadhu/Forecast**.
- **Excel** `Forecasting Case- Study.xlsx`: columns **State**, **Date**, **Total** (scroll a few rows).

### Say (text)
> I’m submitting an end-to-end **weekly sales forecasting** solution. The brief is to forecast **eight weeks ahead** **per state** using the attached Excel data. I train **four models**—**SARIMAX**, **Prophet**, **XGBoost with lag features**, and **LSTM**—compare them on a **time-ordered validation window**, pick the **best RMSE**, refit on **full history**, and expose results through a **REST API**. Code and documentation are in this GitHub repo.

---

## 2. Four models & selection (~60 s)

### On screen
- **`forecasting/model_zoo.py`**: scroll to **`SarimaModel`**, **`ProphetModel`**, **`XgbLagModel`**, **`LstmModel`**, then **`evaluate_models_on_val`** and **`pick_best`**.
- Optionally **`forecasting/trainer.py`**: the loop that calls **`evaluate_models_on_val`** and **`fit_best_and_forecast`**.

### Say (text)
> All four models are implemented in **`model_zoo.py`**. On the validation slice I compute **RMSE** for each; **`pick_best`** chooses the winner. Then **`fit_best_and_forecast`** refits **only that model** on the **complete** weekly series and produces **eight** future weekly forecasts. That’s written to **`artifacts/manifest.json`** per state.

---

## 3. Feature engineering & no leakage (~90 s)

### On screen
- **`forecasting/features.py`**: **`supervised_frame_from_series`** — point at **`lag_1`, `lag_7`, `lag_30`**, **`roll_mean_` / `roll_std_`**, **`add_calendar_features`** (**`dow`**, **`month`**, **`holiday_week`**).
- **`forecasting/splits.py`**: entire **`temporal_split`** function — train slice **`0 : n−val_weeks`**, val **`n−val_weeks : n`**.

### Say (text)
> Feature engineering is in **`features.py`**. Because the target is **weekly**, **t−1, t−7, t−30** mean **one, seven, and thirty weeks**. Rolling mean and std use windows **seven** and **thirty**, with **`shift(1)`** before rolling so we don’t leak the current week into its own stats. Calendar features are **day of week**, **month**, and a **US holiday** flag for the week. **XGBoost** and **LSTM** use this matrix; **SARIMAX** and **Prophet** use their own seasonal structure. For leakage: **`splits.py`** uses a strict **chronological** split—**no shuffle**—train on the past, validate on the **last eight weeks**.

---

## 4. Weekly data & Excel load (optional ~45 s)

### On screen
- **`forecasting/io.py`**: **`coerce_sales_numeric`**, **`weekly_state_series`** (resample to **`W-SUN`**).

### Say (text)
> Raw rows are cleaned—commas and currency in **Total**—then aggregated to **weekly** sales per state so every model sees a regular weekly signal.

---

## 5. Manifest / outputs (~60 s)

### On screen
- **`artifacts/manifest.json`**: open **one state** (e.g. **Texas**). Show **`best_model`**, **`validation_rmse`** for all four models, and **`forecast_weeks`** (eight **`week_start`** / **`forecast_sales`** pairs).

### Say (text)
> After training, **`manifest.json`** stores one block per state: which model won, validation RMSE for all four, and the **eight-week** forecast path the API serves.

---

## 6. REST API & Swagger (~90 s)

### On screen
- **Terminal**:  
  `uvicorn api.main:app --host 127.0.0.1 --port 8765`
- **Browser**: `http://127.0.0.1:8765/docs` (zoom in).
- Run **`GET /health`** → **`GET /states`** → **`GET /forecast/Texas`** (Execute each; pause on JSON).

### Say (text)
> The service is **FastAPI** in **`api/main.py`**. I start **Uvicorn** on **localhost**. **`/docs`** is OpenAPI: **`/health`** for liveness, **`/states`** lists states from the manifest, **`/forecast/{state}`** returns the same content as the JSON file—**best model**, **RMSEs**, and **eight forecasts**. There’s also **`POST /train`** to rebuild the manifest; it’s heavy, so I’m only showing reads here.

---

## 7. Closing (~30 s)

### On screen
- README **For evaluators** + **Feature engineering** table, or GitHub repo root.
- **Demo video** line at top of README (after you add your link).

### Say (text)
> That covers **models**, **features**, **time-based validation**, **artifacts**, and the **API**. Full setup is in the **README**; the repo includes the **case PDF**, **Excel**, and committed **`manifest.json`** for quick evaluation. Thanks.

---

## Recording tips

| Tip | Detail |
|-----|--------|
| Font | Zoom editor & browser (**Ctrl + +**) so text is readable at 1080p. |
| Order | GitHub/README → `features.py` → `splits.py` → `model_zoo.py` → `manifest.json` → terminal → Swagger. |
| Audio | Pause talking **before** clicking **Execute** if clicks are loud. |
| Length | Skip section 4 or shorten section 2 if you need **under 6 minutes**. |
