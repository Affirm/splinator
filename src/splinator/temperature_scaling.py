"""Temperature Scaling utilities and estimator.

This module provides temperature scaling for probability calibration,
based on Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017).

Temperature scaling rescales logits by a single learned parameter T:
    calibrated_prob = sigmoid(logit(p) / T)

- T > 1: softens probabilities (less confident)
- T < 1: sharpens probabilities (more confident)
- T = 1: leaves probabilities unchanged
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_random_state, validate_data
from sklearn.exceptions import NotFittedError
from typing import Optional, Union
import warnings


def _weighted_cross_entropy(y_true, y_pred, sample_weight=None, eps=1e-15):
    """Compute (weighted) binary cross-entropy loss.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
    eps : float
        Small constant for numerical stability.
        
    Returns
    -------
    loss : float
        Mean (weighted) cross-entropy loss.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    
    # Handle NaN/Inf values (can happen with extreme logits)
    y_pred = np.nan_to_num(y_pred, nan=0.5, posinf=1-eps, neginf=eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    ce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    if sample_weight is None:
        return np.mean(ce)
    else:
        return np.average(ce, weights=sample_weight)


def find_optimal_temperature(
    y_true,
    y_pred,
    sample_weight=None,
    bounds=(0.01, 100.0),
    method='bounded',
):
    """Find the optimal temperature that minimizes negative log-likelihood.
    
    Solves: T* = argmin_T L(y, sigmoid(logit(p) / T))
    
    This is used for the variational decomposition of loss into
    refinement error and calibration error.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities in (0, 1).
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights for weighted NLL optimization.
    bounds : tuple of (float, float), default=(0.01, 100.0)
        Bounds for temperature search.
    method : str, default='bounded'
        Optimization method for minimize_scalar.
        
    Returns
    -------
    temperature : float
        Optimal temperature that minimizes NLL.
        
    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0.1, 0.3, 0.7, 0.9])
    >>> T = find_optimal_temperature(y_true, y_pred)
    >>> print(f"Optimal temperature: {T:.3f}")
    
    Notes
    -----
    The optimization is convex in log(T), so we optimize over log-space
    for better numerical stability.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Validate inputs
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must contain only 0 and 1")
    
    # Clip predictions to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Convert to logits once
    logits = logit(y_pred)
    
    def nll_at_temperature(T):
        """Compute NLL at given temperature."""
        scaled_probs = expit(logits / T)
        return _weighted_cross_entropy(y_true, scaled_probs, sample_weight)
    
    # Optimize
    result = minimize_scalar(
        nll_at_temperature,
        bounds=bounds,
        method=method,
    )
    
    if not result.success:
        warnings.warn(f"Temperature optimization did not converge: {result.message}")
    
    return float(result.x)


def apply_temperature_scaling(y_pred, temperature, eps=1e-15):
    """Apply temperature scaling to predicted probabilities.
    
    Computes: calibrated = sigmoid(logit(p) / T)
    
    Parameters
    ----------
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities in (0, 1).
    temperature : float
        Temperature parameter. T > 1 softens, T < 1 sharpens.
        
    Returns
    -------
    calibrated : ndarray of shape (n_samples,)
        Temperature-scaled probabilities.
        
    Examples
    --------
    >>> import numpy as np
    >>> y_pred = np.array([0.2, 0.5, 0.8])
    >>> # T > 1 pushes probabilities toward 0.5
    >>> apply_temperature_scaling(y_pred, temperature=2.0)
    array([0.33..., 0.5, 0.66...])
    >>> # T < 1 pushes probabilities toward 0 or 1
    >>> apply_temperature_scaling(y_pred, temperature=0.5)
    array([0.05..., 0.5, 0.94...])
    """
    y_pred = np.asarray(y_pred)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    logits = logit(y_pred)
    scaled_logits = logits / temperature
    
    # Clip output to prevent exactly 0 or 1 (causes log issues)
    return np.clip(expit(scaled_logits), eps, 1 - eps)


class TemperatureScaling(RegressorMixin, TransformerMixin, BaseEstimator):
    """Temperature Scaling post-hoc calibrator.
    
    Learns a single temperature parameter T that rescales logits:
        calibrated_prob = sigmoid(logit(p) / T)
    
    - T > 1: softens probabilities (less confident)
    - T < 1: sharpens probabilities (more confident)
    - T = 1: leaves probabilities unchanged
    
    This is the simplest post-hoc calibration method, from Guo et al.
    "On Calibration of Modern Neural Networks" (ICML 2017).
    
    Parameters
    ----------
    bounds : tuple of (float, float), default=(0.01, 100.0)
        Bounds for temperature search during optimization.
        
    Attributes
    ----------
    temperature_ : float
        Learned temperature parameter.
    n_features_in_ : int
        Number of features seen during fit.
        
    Examples
    --------
    >>> from splinator import TemperatureScaling
    >>> import numpy as np
    >>> # Overconfident model predictions
    >>> val_probs = np.array([0.05, 0.1, 0.9, 0.95])
    >>> y_val = np.array([0, 0, 1, 1])
    >>> ts = TemperatureScaling()
    >>> ts.fit(val_probs.reshape(-1, 1), y_val)
    TemperatureScaling()
    >>> print(f"Optimal temperature: {ts.temperature_:.3f}")
    >>> # Apply to test predictions
    >>> test_probs = np.array([[0.1], [0.9]])
    >>> calibrated = ts.predict(test_probs)
    
    Notes
    -----
    - Input X should be predicted probabilities, shape (n_samples,) or (n_samples, 1)
    - Works in sklearn pipelines
    - For multi-class, input should be logits of shape (n_samples, n_classes)
      (multi-class not yet implemented)
    
    See Also
    --------
    splinator.LinearSplineLogisticRegression : More flexible spline-based calibrator
    """
    
    def __init__(self, bounds=(0.01, 100.0)):
        self.bounds = bounds
    
    def fit(self, X, y, sample_weight=None):
        """Fit temperature parameter by minimizing NLL on calibration set.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Predicted probabilities to calibrate.
        y : array-like of shape (n_samples,)
            True labels.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Handle 1D input
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Validate data
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            ensure_2d=True,
            dtype=[np.float64, np.float32],
            y_numeric=True,
        )
        
        if X.shape[1] != 1:
            raise ValueError(
                f"TemperatureScaling expects 1D probabilities, got shape {X.shape}"
            )
        
        # Extract probabilities
        probs = X[:, 0]
        
        # Find optimal temperature
        self.temperature_ = find_optimal_temperature(
            y_true=y,
            y_pred=probs,
            sample_weight=sample_weight,
            bounds=self.bounds,
        )
        
        return self
    
    def transform(self, X):
        """Apply temperature scaling to probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Predicted probabilities to calibrate.
            
        Returns
        -------
        calibrated : ndarray of shape (n_samples,)
            Temperature-scaled probabilities.
        """
        if not self.is_fitted:
            raise NotFittedError(
                "TemperatureScaling is not fitted. Call fit() first."
            )
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Validate without resetting n_features_in_
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            ensure_2d=True,
            dtype=[np.float64, np.float32],
            reset=False,
        )
        
        probs = X[:, 0]
        return apply_temperature_scaling(probs, self.temperature_)
    
    def predict(self, X):
        """Return calibrated probabilities (alias for transform).
        
        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Predicted probabilities to calibrate.
            
        Returns
        -------
        calibrated : ndarray of shape (n_samples,)
            Temperature-scaled probabilities.
        """
        return self.transform(X)
    
    @property
    def is_fitted(self):
        """Check if the estimator is fitted."""
        return hasattr(self, 'temperature_')
    
    def __sklearn_tags__(self):
        """Define sklearn tags for scikit-learn >= 1.6."""
        from sklearn.utils import Tags, TargetTags, RegressorTags
        
        tags = super().__sklearn_tags__()
        tags.target_tags = TargetTags(
            required=True,
            one_d_labels=True,
            two_d_labels=False,
            positive_only=False,
            multi_output=False,
            single_output=True,
        )
        tags.regressor_tags = RegressorTags(poor_score=True)
        return tags
    
    def _more_tags(self):
        """Override default sklearn tags for scikit-learn < 1.6."""
        return {"poor_score": True, "binary_only": True, "requires_y": True}

