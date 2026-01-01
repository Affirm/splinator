# Splinator 📈

**Probability Calibration for Python**

A scikit-learn compatible toolkit for measuring and improving probability calibration.

[![PyPI version](https://img.shields.io/pypi/v/splinator)](https://pypi.org/project/splinator/)
[![Downloads](https://static.pepy.tech/badge/splinator)](https://pepy.tech/project/splinator)
[![Downloads/Month](https://static.pepy.tech/badge/splinator/month)](https://pepy.tech/project/splinator)
[![Documentation Status](https://readthedocs.org/projects/splinator/badge/?version=latest)](https://splinator.readthedocs.io/en/latest/)
[![Build](https://img.shields.io/github/actions/workflow/status/affirm/splinator/.github/workflows/python-package.yml)](https://github.com/affirm/splinator/actions)

## Installation

```bash
pip install splinator
```

## What's Inside

| Category | Components |
|----------|------------|
| **Calibrators** | `LinearSplineLogisticRegression` (piecewise), `TemperatureScaling` (single param) |
| **Refinement Metrics** | `spline_refinement_loss`, `ts_refinement_loss` |
| **Decomposition** | `logloss_decomposition`, `brier_decomposition` |
| **Calibration Metrics** | ECE, Spiegelhalter's z |

## Quick Start

```python
from splinator import LinearSplineLogisticRegression, TemperatureScaling

# Piecewise linear calibration (flexible, monotonic)
spline = LinearSplineLogisticRegression(n_knots=10, monotonicity='increasing')
spline.fit(scores.reshape(-1, 1), y_true)
calibrated = spline.predict_proba(scores.reshape(-1, 1))[:, 1]

# Temperature scaling (simple, single parameter)
ts = TemperatureScaling()
ts.fit(probs.reshape(-1, 1), y_true)
calibrated = ts.predict(probs.reshape(-1, 1))
```

## Calibration Metrics

```python
from splinator import (
    expected_calibration_error,
    spiegelhalters_z_statistic,
    logloss_decomposition,      # Log loss → refinement + calibration
    brier_decomposition,        # Brier score → refinement + calibration
    spline_refinement_loss,     # Log loss after piecewise spline
)

# Assess calibration quality
ece = expected_calibration_error(y_true, probs)
z_stat = spiegelhalters_z_statistic(y_true, probs)

# Decompose log loss into fixable vs irreducible parts
decomp = logloss_decomposition(y_true, probs)
print(f"Refinement (irreducible): {decomp['refinement_loss']:.4f}")
print(f"Calibration (fixable):    {decomp['calibration_loss']:.4f}")

# Refinement using splinator's piecewise calibrator
spline_ref = spline_refinement_loss(y_val, probs, n_knots=5)
```

## XGBoost / LightGBM Integration

Use calibration-aware metrics for early stopping:

```python
from splinator import ts_refinement_loss
from splinator.metric_wrappers import make_metric_wrapper

metric = make_metric_wrapper(ts_refinement_loss, framework='xgboost')
model = xgb.train(params, dtrain, custom_metric=metric, early_stopping_rounds=10, ...)
```

## Examples

| Notebook | Description |
|----------|-------------|
| [calibrator_model_comparison](examples/calibrator_model_comparison.ipynb) | Compare with sklearn calibrators |
| [spline_model_comparison](examples/spline_model_comparison.ipynb) | Compare with pyGAM |
| [ts_refinement_xgboost](examples/ts_refinement_xgboost.py) | Early stopping with refinement loss |

## References

- Zhang, J. & Yang, Y. (2004). [Probabilistic score estimation with piecewise logistic regression](https://pal.sri.com/wp-content/uploads/publications/radar/2004/icml04zhang.pdf). ICML.
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). [On calibration of modern neural networks](https://arxiv.org/abs/1706.04599). ICML.
- Berta, E., Holzmüller, D., Jordan, M. I. & Bach, F. (2025). [Rethinking Early Stopping: Refine, Then Calibrate](https://arxiv.org/abs/2501.19195). arXiv:2501.19195.

See also: [probmetrics](https://github.com/dholzmueller/probmetrics) (PyTorch calibration by the refinement paper authors)

## Development

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv
uv sync --dev && uv run pytest tests -v          # Setup and test
```
