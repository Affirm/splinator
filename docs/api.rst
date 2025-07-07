API Reference
=============

This page contains the full API reference for all public classes and functions in splinator.

Main Module
-----------

.. automodule:: splinator
   :members:
   :undoc-members:
   :show-inheritance:

Estimators
----------

.. automodule:: splinator.estimators
   :members:
   :undoc-members:
   :show-inheritance:

Linear Spline Logistic Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: splinator.estimators.linear_spline_logistic_regression
   :members:
   :undoc-members:
   :show-inheritance:

KS Spline Calibrator
~~~~~~~~~~~~~~~~~~~~

.. automodule:: splinator.estimators.ks_spline_calibrator
   :members:
   :undoc-members:
   :show-inheritance:

CDF Spline Calibrator
~~~~~~~~~~~~~~~~~~~~~

The CDF Spline Calibrator implements the method from Gupta et al. (2021) [1]_ for
smooth probability calibration using cubic splines on cumulative distribution functions.

.. automodule:: splinator.estimators.cdf_spline_calibrator
   :members:
   :undoc-members:
   :show-inheritance:

.. [1] Gupta, C., Koren, A., & Mishra, K. (2021). "Calibration of Neural Networks 
       using Splines". International Conference on Learning Representations (ICLR).
       https://arxiv.org/abs/2006.12800

See the authors' official implementation at: https://github.com/kartikgupta-at-anu/spline-calibration

Metrics
-------

.. automodule:: splinator.metrics
   :members:
   :undoc-members:
   :show-inheritance: 