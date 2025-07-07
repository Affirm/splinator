.. splinator documentation master file, created by
   sphinx-quickstart on Mon Jan 20 01:28:16 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to splinator's documentation!
=====================================

**splinator** is a Python library for fitting linear-spline based logistic regression for calibration.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples

Features
--------

* Linear spline-based logistic regression models
* CDF-based spline calibration for probability predictions
* scikit-learn compatible API
* Support for multi-class classification

Installation
------------

Install splinator using pip::

   pip install splinator

For development installation::

   pip install -e ".[dev]"

Quick Example
-------------

.. code-block:: python

   from splinator.estimators import LinearSplineLogisticRegression
   
   # Create and fit model
   model = LinearSplineLogisticRegression(n_knots=5)
   model.fit(X_train, y_train)
   
   # Make predictions
   predictions = model.predict_proba(X_test)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

