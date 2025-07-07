# Splinator 📈

**Probabilistic Calibration with Regression Splines**

A scikit-learn compatible Python library for probability calibration of machine learning models using spline-based methods.

[![PyPI version](https://badge.fury.io/py/splinator.svg)](https://badge.fury.io/py/splinator)
[![Documentation Status](https://readthedocs.org/projects/splinator/badge/?version=latest)](https://splinator.readthedocs.io/en/latest/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/affirm/splinator/.github/workflows/python-package.yml)](https://github.com/affirm/splinator/actions)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python Version](https://img.shields.io/pypi/pyversions/splinator)](https://pypi.org/project/splinator/)

## Features

- **Linear Spline Logistic Regression**: Flexible non-linear classification with automatic knot placement
- **CDF Spline Calibration**: State-of-the-art probability calibration for multi-class classifiers
- **scikit-learn Compatible**: Seamless integration with existing ML pipelines
- **Comprehensive Metrics**: Calibration evaluation tools including ECE and Spiegelhalter's z-statistic

## Installation

```bash
pip install splinator
```

For development installation with extra dependencies:
```bash
pip install splinator[dev]
```

## Quick Start

### Linear Spline Logistic Regression
```python
from splinator.estimators import LinearSplineLogisticRegression
import numpy as np

# Generate sample data
n_samples = 1000
X = np.random.randn(n_samples, 2)
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

# Fit model with automatic knot selection
model = LinearSplineLogisticRegression(n_knots=10)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)
```

### Probability Calibration
```python
from splinator.estimators import CDFSplineCalibrator
from sklearn.model_selection import train_test_split

# Split data for calibration
X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2)

# Train base model and calibrator
base_model = LinearSplineLogisticRegression().fit(X_train, y_train)
calibrator = CDFSplineCalibrator()
calibrator.fit(base_model.predict_proba(X_cal), y_cal)

# Apply calibration
calibrated_probs = calibrator.transform(base_model.predict_proba(X_test))
```

## Documentation

Full documentation is available at [splinator.readthedocs.io](https://splinator.readthedocs.io/).

## Examples

Interactive notebooks demonstrating various features:

| Topic | Notebook |
|-------|----------|
| Calibrator Comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Affirm/splinator/blob/main/examples/calibrator_model_comparison.ipynb) |
| Spline Model Comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/Affirm/splinator/blob/main/examples/spline_model_comparison.ipynb) |

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Development

1. Clone the repository
2. Install in development mode: `pip install -e ".[dev]"`
3. Run tests: `pytest tests/`
4. Check types: `mypy src/splinator`
5. Format code: `black src/ tests/`

## Citation

If you use splinator in your research, please cite:

```bibtex
@software{splinator,
  title = {Splinator: Probabilistic Calibration with Regression Splines},
  author = {Xu, Jiarui},
  year = {2024},
  url = {https://github.com/affirm/splinator}
}
```

## References

- Zhang, J., & Yang, Y. (2004). [Probabilistic score estimation with piecewise logistic regression](https://pal.sri.com/wp-content/uploads/publications/radar/2004/icml04zhang.pdf). In Proceedings of the twenty-first international conference on Machine learning.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). [On calibration of modern neural networks](https://arxiv.org/abs/1706.04599). In International conference on machine learning (pp. 1321-1330). PMLR.
- Gupta, C., Koren, A., & Mishra, K. (2021). [Calibration of Neural Networks using Splines](https://arxiv.org/abs/2006.12800). In International Conference on Learning Representations (ICLR). Official implementation: [kartikgupta-at-anu/spline-calibration](https://github.com/kartikgupta-at-anu/spline-calibration).

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
