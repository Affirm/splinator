# Splinator 📈

**Probablistic Calibration with Regression Splines**

[scikit-learn](https://scikit-learn.org) compatible

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Documentation Status](https://readthedocs.org/projects/splinator/badge/?version=latest)](https://splinator.readthedocs.io/en/latest/)
[![Build](https://img.shields.io/github/actions/workflow/status/affirm/splinator/.github/workflows/python-package.yml)](https://github.com/affirm/splinator/actions)

## Installation

`pip install splinator`

## Algorithm

Supported models:

- Linear Spline Logistic Regression

Supported metrics:

- Spiegelhalter’s z statistic
- Expected Calibration Error (ECE)

\[1\] You can find more information in the [Linear Spline Logistic
Regression](https://github.com/Affirm/splinator/wiki/Linear-Spline-Logistic-Regression).

\[2\] Additional readings

- Zhang, Jian, and Yiming Yang. [Probabilistic score estimation with
    piecewise logistic
    regression](https://pal.sri.com/wp-content/uploads/publications/radar/2004/icml04zhang.pdf).
    Proceedings of the twenty-first international conference on Machine
    learning. 2004.
- Guo, Chuan, et al. "On calibration of modern neural networks." International conference on machine learning. PMLR, 2017.


## Examples

| comparison                                     | notebook                                                                                                                                                           |
|------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| scikit-learn's sigmoid and isotonic regression | [![colab1](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Affirm/splinator/blob/main/examples/calibrator_model_comparison.ipynb)    |
| pyGAM’s spline model                           | [![colab2](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Affirm/splinator/blob/main/examples/spline_model_comparison.ipynb) |

## Development

The dependencies are managed by [uv](https://github.com/astral-sh/uv).

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync --dev

# Run tests
uv run pytest tests -v

# Run type checking
uv run mypy src/splinator
```

## Example Usage

``` python
from splinator.estimators import LinearSplineLogisticRegression
import numpy as np

# random synthetic dataset
n_samples = 100
rng = np.random.RandomState(0)
X = rng.normal(loc=100, size=(n_samples, 2))
y = np.random.randint(2, size=n_samples)

lslr = LinearSplineLogisticRegression(n_knots=10)
lslr.fit(X, y)
```
