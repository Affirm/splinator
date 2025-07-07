Quick Start Guide
=================

This guide will help you get started with splinator quickly.

Basic Usage
-----------

Linear Spline Logistic Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main estimator in splinator is the LinearSplineLogisticRegression, which fits a logistic regression model using linear splines:

.. code-block:: python

    import numpy as np
    from splinator.estimators import LinearSplineLogisticRegression
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 1)
    y = (X.squeeze() + np.random.randn(1000) * 0.5 > 0).astype(int)
    
    # Create and fit the model
    model = LinearSplineLogisticRegression(n_knots=5)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

Calibration with Splines
~~~~~~~~~~~~~~~~~~~~~~~~

Splinator provides calibrators that use splines to calibrate probability predictions:

.. code-block:: python

    from splinator.estimators import CDFSplineCalibrator
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.3)
    
    # Train a base model
    base_model = LinearSplineLogisticRegression(n_knots=3)
    base_model.fit(X_train, y_train)
    
    # Get uncalibrated probabilities (all classes)
    proba_uncalibrated = base_model.predict_proba(X_cal)
    
    # Calibrate using CDF Spline Calibrator
    cdf_calibrator = CDFSplineCalibrator(num_knots=6)
    cdf_calibrator.fit(proba_uncalibrated, y_cal)
    
    # Transform test probabilities
    proba_test = base_model.predict_proba(X_test)
    proba_calibrated = cdf_calibrator.transform(proba_test)

Integration with scikit-learn
-----------------------------

All estimators in splinator follow the scikit-learn API and can be used in pipelines:

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LinearSplineLogisticRegression(n_knots=5))
    ])
    
    # Cross-validation
    scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"Cross-validation scores: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})") 