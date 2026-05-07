"""Train models from Excel and write artifacts/manifest.json."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from forecasting.config import Settings, settings
from forecasting.trainer import train_all


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(settings.data_path))
    ap.add_argument("--artifacts", type=str, default=str(settings.artifacts_dir))
    ap.add_argument("--horizon", type=int, default=settings.horizon_weeks)
    ap.add_argument("--val-weeks", type=int, default=settings.val_weeks)
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Faster training: smaller XGBoost and many fewer LSTM epochs (good for dev; use full run for best accuracy).",
    )
    args = ap.parse_args()

    cfg = settings.model_copy(
        update={
            "data_path": Path(args.data),
            "artifacts_dir": Path(args.artifacts),
            "horizon_weeks": args.horizon,
            "val_weeks": args.val_weeks,
            "fast_mode": args.fast or settings.fast_mode,
        }
    )
    train_all(cfg)


if __name__ == "__main__":
    main()
