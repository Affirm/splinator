Examples
========

This section provides detailed examples demonstrating various features of splinator.

Example Notebooks
-----------------

The `examples/` directory contains several Jupyter notebooks with detailed examples:

* **calibration_comparison.py** - Compares different calibration methods
* **calibrator_model_comparison.ipynb** - Comparison of calibrator models
* **fit_on_splines.ipynb** - Fitting models on spline features
* **metrics.ipynb** - Using splinator metrics for evaluation
* **spline_model_comparison.ipynb** - Comparing different spline models
* **sklearn_check.ipynb** - Verifying scikit-learn compatibility

Calibration Example
-------------------

Here's a complete example of using splinator for probability calibration:

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from splinator.estimators import CDFSplineCalibrator
    from splinator.metrics import expected_calibration_error
    
    # Generate synthetic data
    X, y = make_classification(n_samples=2000, n_features=20, 
                               n_informative=15, n_classes=3,
                               n_clusters_per_class=1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )
    
    # Train base model
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    
    # Get uncalibrated probabilities (all classes)
    proba_uncal = lr.predict_proba(X_test)
    
    # Calculate ECE before calibration (for binary case, use positive class)
    # For multi-class, you might want to compute ECE per class
    # ece_before = expected_calibration_error(y_test, proba_uncal[:, 1])
    
    # Calibrate using CDF Spline
    calibrator = CDFSplineCalibrator(num_knots=6)
    calibrator.fit(lr.predict_proba(X_cal), y_cal)
    
    # Get calibrated probabilities
    proba_cal = calibrator.transform(proba_uncal)
    
    # Calculate ECE after calibration
    # ece_after = expected_calibration_error(y_test, proba_cal[:, 1])
    
    print(f"Calibrated probabilities shape: {proba_cal.shape}")
    print(f"Sum of probabilities per sample (should be 1): {proba_cal.sum(axis=1)[:5]}")

Linear Spline Logistic Regression Example
-----------------------------------------

Example showing how to use the LinearSplineLogisticRegression model:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from splinator.estimators import LinearSplineLogisticRegression
    from sklearn.datasets import make_classification
    
    # Generate non-linear classification data
    np.random.seed(42)
    X = np.random.randn(500, 1) * 2
    y = (np.sin(X.squeeze()) + np.random.randn(500) * 0.3 > 0).astype(int)
    
    # Fit models with different numbers of knots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for ax, n_knots in zip(axes, [2, 5, 10]):
        model = LinearSplineLogisticRegression(n_knots=n_knots)
        model.fit(X, y)
        
        # Plot decision boundary
        X_plot = np.linspace(-4, 4, 200).reshape(-1, 1)
        y_proba = model.predict_proba(X_plot)[:, 1]
        
        ax.scatter(X, y, alpha=0.5, c=y, cmap='viridis')
        ax.plot(X_plot, y_proba, 'r-', linewidth=2, label=f'P(y=1|x)')
        ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.set_title(f'{n_knots} knots')
    
    plt.tight_layout()
    plt.show()

Running the Examples
--------------------

To run the example notebooks:

1. Install splinator with development dependencies::

    pip install -e ".[dev]"

2. Start Jupyter::

    jupyter notebook

3. Navigate to the `examples/` directory and open any notebook

For more examples, check the `examples/` directory in the repository. 