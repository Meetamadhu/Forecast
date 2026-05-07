"""Reduce noisy logs/warnings from statsmodels, Prophet, TensorFlow during training."""

from __future__ import annotations

import logging
import os
import warnings


class _DropProphetPlotImportNoise(logging.Filter):
    """Prophet logs plotly/matplotlib import failures from logger ``prophet.plot``."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage().lower()
        if "importing plotly failed" in msg:
            return False
        if "importing matplotlib failed" in msg:
            return False
        return True


class _DropProphetAutoSeasonalityInfo(logging.Filter):
    """Hide INFO lines like 'Disabling daily seasonality...' (normal for weekly data)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.INFO:
            return True
        msg = record.getMessage().lower()
        if "disabling" in msg and "seasonality" in msg:
            return False
        return True


def apply_prophet_log_quiet() -> None:
    """Undo Prophet's ``forecaster.py`` forcing ``prophet`` logger to INFO and drop benign noise."""
    plot_log = logging.getLogger("prophet.plot")
    root_prophet = logging.getLogger("prophet")

    if not getattr(plot_log, "_qh_plot_filters", False):
        plot_log.addFilter(_DropProphetPlotImportNoise())
        plot_log._qh_plot_filters = True  # type: ignore[attr-defined]

    if not getattr(root_prophet, "_qh_prophet_filters", False):
        root_prophet.addFilter(_DropProphetAutoSeasonalityInfo())
        root_prophet._qh_prophet_filters = True  # type: ignore[attr-defined]

    root_prophet.setLevel(logging.WARNING)


def configure_quiet_training() -> None:
    """Call before fitting models (especially before first TensorFlow import)."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    try:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        warnings.simplefilter("ignore", ConvergenceWarning)
    except Exception:
        warnings.filterwarnings(
            "ignore",
            message=".*Maximum Likelihood optimization failed to converge.*",
        )

    apply_prophet_log_quiet()

    for name in ("cmdstanpy", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass
