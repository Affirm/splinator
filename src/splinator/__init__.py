from ._version import __version__
from .estimators import LinearSplineLogisticRegression
from .metrics import (
    expected_calibration_error,
    spiegelhalters_z_statistic,
    # Log-loss decomposition (via Temperature Scaling)
    ts_refinement_loss,
    ts_brier_refinement,  # Brier after TS (for fair comparison)
    calibration_loss,
    loss_decomposition,
    # Brier score decomposition (Berta et al. 2025)
    brier_decomposition,
    brier_refinement_score,
    brier_calibration_score,
)
from .metric_wrappers import make_metric_wrapper
from .monotonic_spline import Monotonicity
from .temperature_scaling import (
    find_optimal_temperature,
    apply_temperature_scaling,
    TemperatureScaling,
)

__all__ = [
    "__version__",
    # Estimators
    "LinearSplineLogisticRegression",
    "TemperatureScaling",
    # Calibration metrics
    "expected_calibration_error",
    "spiegelhalters_z_statistic",
    # Log-loss decomposition (via Temperature Scaling)
    "ts_refinement_loss",
    "ts_brier_refinement",  # Brier after TS (for fair comparison)
    "calibration_loss",
    "loss_decomposition",
    # Brier score decomposition (Berta et al. 2025)
    "brier_decomposition",
    "brier_refinement_score",
    "brier_calibration_score",
    # Wrapper factory
    "make_metric_wrapper",
    # Temperature scaling utilities
    "find_optimal_temperature",
    "apply_temperature_scaling",
    # Enums
    "Monotonicity",
]
