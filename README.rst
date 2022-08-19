.. -*- mode: rst -*-

Splinator
============================================================
.. _scikit-learn: https://scikit-learn.org
.. _poetry: https://python-poetry.org/docs/basic-usage/
.. _PR: https://github.com/Affirm/splinator/pull/1

**Spline-based Regression Models for Probability Calibration**

scikit-learn_ compatible

Installation
------------

``pip install splinator``

Algorithm
------------
(Link to medium blog and arxiv PDF)

Releases
------------
alpha version in active development: PR_

Development
------------
The dependencies are managed by poetry_

To run tests, run ``poetry run pytest splinator/tests``

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
