# Splinator 📈

**Probablistic Calibration with Regression Splines**

[scikit-learn](https://scikit-learn.org) compatible

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)
[![Documentation Status](https://readthedocs.org/projects/splinator/badge/?version=latest)](https://splinator.readthedocs.io/en/latest/)
[![Build](https://img.shields.io/github/actions/workflow/status/affirm/splinator/.github/workflows/python-package.yml)](https://github.com/affirm/splinator/actions)

## Installation

`pip install splinator`

## Algorithm

\[1\] You can find more information in the [Linear Spline Logistic
Regression](https://github.com/Affirm/splinator/wiki/Linear-Spline-Logistic-Regression).

\[2\] Additional readings

-   Zhang, Jian, and Yiming Yang. [Probabilistic score estimation with
    piecewise logistic
    regression](https://pal.sri.com/wp-content/uploads/publications/radar/2004/icml04zhang.pdf).
    Proceedings of the twenty-first international conference on Machine
    learning. 2004.

## Examples

| comparison                                     | notebook                                                                                                                                                           |
|------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| scikit-learn's sigmoid and isotonic regression | [![colab1](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Affirm/splinator/blob/main/examples/calibrator_model_comparison.ipynb)    |
| pyGAM’s spline model                           | [![colab2](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Affirm/splinator/blob/main/examples/spline_model_comparison.ipynb) |

## Development

The dependencies are managed by [pdm](https://pdm.fming.dev/latest/)

To run tests, run `pdm run -v pytest tests`

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
