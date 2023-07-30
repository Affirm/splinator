.. -*- mode: rst -*-

Splinator 📈
============================================================
.. _scikit-learn: https://scikit-learn.org
.. _pdm: https://pdm.fming.dev/latest/
.. _PR: https://github.com/Affirm/splinator/pull/1

**Probablistic Calibration with Regression Splines**

scikit-learn_ compatible

.. image:: https://img.shields.io/badge/pdm-managed-blueviolet
   :target: https://pdm.fming.dev
   :alt: pdm-managed

.. image:: https://readthedocs.org/projects/splinator/badge/?version=latest
    :target: https://splinator.readthedocs.io/en/latest/
    :alt: Documentation Status

.. image:: https://img.shields.io/github/actions/workflow/status/affirm/splinator/.github/workflows/python-package.yml
    :target: https://github.com/affirm/splinator/actions
    :alt: Build

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://githubtocolab.com/Affirm/splinator/blob/main/examples/spline_model_comparison.ipynb
    :alt: colab

Installation
------------

``pip install splinator``

Algorithm
------------
You can find more information in the `Linear Spline Logistic Regression <https://github.com/Affirm/splinator/wiki/Linear-Spline-Logistic-Regression>`_.

Examples
------------
+------------------------------------------------+------------+
| comparison                                     |  notebook  |
+================================================+============+
| scikit-learn's sigmoid and isotonic regression |  |colab|   |
+------------------------------------------------+------------+
| pyGAM’s spline model                           |  |colab|   |
+------------------------------------------------+------------+

Development
------------
The dependencies are managed by pdm_

To run tests, run ``pdm run -v pytest tests``

Example Usage
--------------

.. code-block:: python

    from splinator.estimators import LinearSplineLogisticRegression
    import numpy as np

    # random synthetic dataset
    n_samples = 100
    rng = np.random.RandomState(0)
    X = rng.normal(loc=100, size=(n_samples, 2))
    y = np.random.randint(2, size=n_samples)

    lslr = LinearSplineLogisticRegression(n_knots=10)
    lslr.fit(X, y)
.. _documentation: https://splinator.readthedocs.io/en/latest/quick_start.html

