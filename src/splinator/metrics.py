"""Calibration metrics and loss decomposition.

This module provides metrics for evaluating probability calibration,
including the TS-Refinement metrics based on [1]_.

Key insight: Total loss = Refinement Loss + Calibration Loss
- Refinement Loss: Irreducible error (model's discriminative ability)
- Calibration Loss: Fixable by post-hoc calibration (e.g., temperature scaling)

Use ts_refinement_loss as an early stopping criterion instead of raw
validation loss to train longer for better discrimination, then apply
post-hoc calibration.

References
----------
.. [1] Berta, E., Holzmüller, D., Jordan, M. I., & Bach, F. (2025). Rethinking Early Stopping:
       Refine, Then Calibrate. arXiv preprint arXiv:2501.19195.
       https://arxiv.org/abs/2501.19195
"""

import numpy as np
from sklearn.calibration import calibration_curve

from splinator.temperature_scaling import (
    find_optimal_temperature,
    apply_temperature_scaling,
    _weighted_cross_entropy,
)


def spiegelhalters_z_statistic(
    labels,  # type: np.array
    preds,  # type: np.array
):
    # type: (...) -> float
    a = ((labels - preds) * (1 - 2 * preds)).sum()
    b = ((1 - 2 * preds) ** 2 * preds * (1 - preds)).sum()
    return float(a / b ** 0.5)


def expected_calibration_error(labels, preds, n_bins=10):
    # type: (np.array, np.array, int) -> float
    fop, mpv = calibration_curve(y_true=labels, y_prob=preds, n_bins=n_bins, strategy='quantile')
    diff = np.array(fop) - np.array(mpv)
    ece = sum([abs(delta) for delta in diff]) / float(n_bins)
    return ece


def ts_refinement_loss(y_true, y_pred, sample_weight=None):
    """Refinement Error: Cross-entropy AFTER optimal temperature scaling.
    
    This is the irreducible loss given perfect calibration — it measures
    the model's fundamental discriminative ability. Use this as the
    early stopping criterion instead of raw validation loss.
    
    Formula: L(y, TS(p)) where TS applies optimal temperature scaling
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities in (0, 1).
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
        
    Returns
    -------
    refinement_loss : float
        Cross-entropy after optimal temperature scaling.
        
    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0.2, 0.4, 0.6, 0.8])
    >>> loss = ts_refinement_loss(y_true, y_pred)
    
    Use as early stopping criterion:
    
    >>> for epoch in range(max_epochs):
    ...     model.train_one_epoch()
    ...     val_probs = model.predict_proba(X_val)[:, 1]
    ...     ts_loss = ts_refinement_loss(y_val, val_probs)
    ...     if ts_loss < best_loss:
    ...         best_loss = ts_loss
    ...         best_model = copy.deepcopy(model)
    
    sklearn GridSearchCV:
    
    >>> from sklearn.metrics import make_scorer
    >>> scorer = make_scorer(
    ...     ts_refinement_loss,
    ...     greater_is_better=False,
    ...     needs_proba=True,
    ...     response_method='predict_proba',
    ... )
    
    XGBoost custom eval:
    
    >>> def xgb_ts_refinement(y_pred, dtrain):
    ...     from scipy.special import expit
    ...     y_true = dtrain.get_label()
    ...     weights = dtrain.get_weight()
    ...     if len(weights) == 0:
    ...         weights = None
    ...     return 'ts_refinement', ts_refinement_loss(y_true, expit(y_pred), weights)
    
    See Also
    --------
    calibration_loss : The "fixable" portion of the loss
    logloss_decomposition : Complete decomposition with all components
    make_metric_wrapper : Factory to create framework-specific wrappers
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Clip predictions to avoid numerical issues
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Find optimal temperature
    T_opt = find_optimal_temperature(y_true, y_pred, sample_weight=sample_weight)
    
    # Apply temperature scaling
    calibrated = apply_temperature_scaling(y_pred, T_opt)
    
    # Compute loss on calibrated predictions
    return _weighted_cross_entropy(y_true, calibrated, sample_weight)


def ts_brier_refinement(y_true, y_pred, sample_weight=None):
    """Brier score AFTER temperature scaling (for fair comparison with TS-refinement).
    
    Unlike brier_refinement_score which uses spline/isotonic recalibration,
    this uses temperature scaling - the same 1-parameter recalibrator as
    ts_refinement_loss. This allows direct comparison of log-loss vs Brier
    scoring rules under the same recalibration method.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities in (0, 1).
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
        
    Returns
    -------
    refinement : float
        Brier score after optimal temperature scaling.
        
    See Also
    --------
    ts_refinement_loss : Log-loss after temperature scaling
    brier_refinement_score : Brier after spline/isotonic recalibration
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Clip predictions to avoid numerical issues
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Find optimal temperature (same as ts_refinement_loss)
    T_opt = find_optimal_temperature(y_true, y_pred, sample_weight=sample_weight)
    
    # Apply temperature scaling
    calibrated = apply_temperature_scaling(y_pred, T_opt)
    
    # Compute Brier score on calibrated predictions
    if sample_weight is None:
        return float(np.mean((y_true - calibrated) ** 2))
    else:
        sample_weight = np.asarray(sample_weight)
        return float(np.average((y_true - calibrated) ** 2, weights=sample_weight))


def spline_refinement_loss(y_true, y_pred, n_knots=5, C=1.0, sample_weight=None):
    """Refinement Error: Cross-entropy AFTER piecewise spline recalibration.
    
    Uses splinator's LinearSplineLogisticRegression as the recalibrator.
    
    Compared to ts_refinement_loss (1 parameter), this uses a piecewise linear
    calibrator with more flexibility. Use fewer knots (2-3) and strong
    regularization (C=1) for stable early stopping signals.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities in (0, 1).
    n_knots : int, default=5
        Number of knots for the piecewise calibrator.
        Fewer knots = more stable for early stopping.
    C : float, default=1.0
        Inverse regularization strength. Smaller = more regularization.
        Use C=1 or lower for early stopping stability.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights (note: currently not passed to spline fitting).
        
    Returns
    -------
    refinement_loss : float
        Cross-entropy after optimal piecewise spline recalibration.
        
    Examples
    --------
    >>> from splinator import spline_refinement_loss
    >>> loss = spline_refinement_loss(y_val, model_probs, n_knots=5, C=1.0)
    
    Notes
    -----
    For early stopping, prefer ts_refinement_loss (1 parameter) for maximum
    stability. Use spline_refinement_loss when you want the recalibrator
    to match what you'll use post-hoc (LinearSplineLogisticRegression).
    
    See Also
    --------
    ts_refinement_loss : Temperature scaling version (more stable, 1 param)
    LinearSplineLogisticRegression : The piecewise calibrator used here
    
    References
    ----------
    .. [1] Berta, M., Ciobanu, S., & Heusinger, M. (2025). Rethinking Early
           Stopping: Refine, Then Calibrate. arXiv:2501.19195.
    """
    from splinator.estimators import LinearSplineLogisticRegression
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Clip predictions to avoid numerical issues
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Fit piecewise spline calibrator
    calibrator = LinearSplineLogisticRegression(
        n_knots=n_knots,
        C=C,
        monotonicity='increasing',
    )
    calibrator.fit(y_pred.reshape(-1, 1), y_true)
    calibrated = calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
    
    # Clip calibrated predictions
    calibrated = np.clip(calibrated, eps, 1 - eps)
    
    # Compute cross-entropy
    return _weighted_cross_entropy(y_true, calibrated, sample_weight)


def calibration_loss(y_true, y_pred, sample_weight=None):
    """Calibration Error: The "fixable" portion of the loss.
    
    This is the potential risk reduction from post-hoc recalibration.
    It measures how much loss is caused purely by poor probability scaling,
    not by the model's inability to discriminate.
    
    Formula: L(y, p) - L(y, TS(p))
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities in (0, 1).
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
        
    Returns
    -------
    calibration_loss : float
        Difference between total loss and refinement loss.
        Always >= 0.
        
    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0.1, 0.2, 0.8, 0.9])  # Well-calibrated
    >>> calibration_loss(y_true, y_pred)  # Should be small
    
    See Also
    --------
    ts_refinement_loss : The irreducible portion of the loss
    logloss_decomposition : Complete decomposition with all components
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    total = _weighted_cross_entropy(y_true, y_pred, sample_weight)
    refinement = ts_refinement_loss(y_true, y_pred, sample_weight)
    
    return total - refinement


def logloss_decomposition(y_true, y_pred, sample_weight=None):
    """Decompose log loss (cross-entropy) into refinement and calibration.
    
    Based on the variational approach from "Rethinking Early Stopping:
    Refine, Then Calibrate". This decomposes log loss as:
    
        Total Loss = Refinement Loss + Calibration Loss
    
    where:
    - Refinement Loss: L(y, TS(p)) — irreducible, measures discrimination
    - Calibration Loss: L(y, p) - L(y, TS(p)) — fixable by recalibration
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities in (0, 1).
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
        
    Returns
    -------
    decomposition : dict
        Dictionary containing:
        - 'total_loss': Total log loss (cross-entropy)
        - 'refinement_loss': Irreducible loss after calibration
        - 'calibration_loss': Fixable portion (total - refinement)
        - 'calibration_fraction': Fraction of loss due to miscalibration
        - 'optimal_temperature': Temperature that minimizes NLL
        
    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0.1, 0.3, 0.7, 0.9])
    >>> decomp = logloss_decomposition(y_true, y_pred)
    >>> print(f"Total: {decomp['total_loss']:.4f}")
    >>> print(f"Refinement: {decomp['refinement_loss']:.4f}")
    >>> print(f"Calibration: {decomp['calibration_loss']:.4f} ({decomp['calibration_fraction']:.1%})")
    
    Monitor during training:
    
    >>> history = {'epoch': [], 'total': [], 'refinement': [], 'calibration': []}
    >>> for epoch in range(max_epochs):
    ...     model.partial_fit(X_train, y_train)
    ...     val_probs = model.predict_proba(X_val)[:, 1]
    ...     decomp = logloss_decomposition(y_val, val_probs)
    ...     history['epoch'].append(epoch)
    ...     history['total'].append(decomp['total_loss'])
    ...     history['refinement'].append(decomp['refinement_loss'])
    ...     history['calibration'].append(decomp['calibration_loss'])
    
    See Also
    --------
    ts_refinement_loss : Just the refinement component
    calibration_loss : Just the calibration component
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Clip predictions to avoid numerical issues
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Total loss (raw predictions)
    total = _weighted_cross_entropy(y_true, y_pred, sample_weight)
    
    # Find optimal temperature
    T_opt = find_optimal_temperature(y_true, y_pred, sample_weight=sample_weight)
    
    # Apply temperature scaling
    calibrated = apply_temperature_scaling(y_pred, T_opt)
    
    # Refinement loss (after calibration)
    refinement = _weighted_cross_entropy(y_true, calibrated, sample_weight)
    
    # Calibration loss (fixable portion)
    calibration = total - refinement
    
    # Fraction of loss due to miscalibration
    calibration_fraction = calibration / total if total > 0 else 0.0
    
    return {
        'total_loss': float(total),
        'refinement_loss': float(refinement),
        'calibration_loss': float(calibration),
        'calibration_fraction': float(calibration_fraction),
        'optimal_temperature': float(T_opt),
    }


def brier_decomposition(y_true, y_pred, sample_weight=None):
    """Decompose Brier score into refinement and calibration (Berta et al. 2025).
    
    Uses the VARIATIONAL decomposition:
    
        Brier = Refinement + Calibration Error
        
    Where:
    - Refinement = min_g E[(y - g(p))²] = Brier AFTER optimal recalibration
    - Calibration = Brier - Refinement = loss reducible by recalibration
    
    This is the theoretically correct decomposition for early stopping.
    
    Also computes Spiegelhalter's 1986 algebraic terms for reference:
    - calibration_term_spiegelhalter: E[(x-p)(1-2p)] (expectation 0 if calibrated)
    - spread_term: E[p(1-p)] (NOT the same as refinement on raw predictions!)
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities in (0, 1).
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
        
    Returns
    -------
    decomposition : dict
        Dictionary containing:
        - 'brier_score': Total Brier score
        - 'refinement': Brier after optimal recalibration (irreducible)
        - 'calibration': Brier - refinement (fixable by recalibration)
        - 'calibration_term': Spiegelhalter's (x-p)(1-2p) term
        - 'spread_term': E[p(1-p)] (for reference, NOT true refinement)
        
    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0.1, 0.3, 0.7, 0.9])
    >>> decomp = brier_decomposition(y_true, y_pred)
    >>> print(f"Brier: {decomp['brier_score']:.4f}")
    >>> print(f"Refinement: {decomp['refinement']:.4f}")
    >>> print(f"Calibration: {decomp['calibration']:.4f}")
    
    Notes
    -----
    Uses isotonic regression as the optimal recalibration function.
    For Brier score, isotonic regression is the theoretically optimal
    recalibrator (minimizes expected squared error).
    
    Key insight: E[p(1-p)] on RAW predictions is NOT the same as refinement!
    Raw p values are miscalibrated, so p(1-p) includes calibration distortion.
    
    See Also
    --------
    brier_refinement_score : Just the refinement component
    spiegelhalters_z_statistic : Statistical test for calibration
    
    References
    ----------
    .. [1] Berta, M., Ciobanu, S., & Heusinger, M. (2025). Rethinking Early
           Stopping: Refine, Then Calibrate. arXiv:2501.19195.
           https://arxiv.org/abs/2501.19195
    .. [2] Spiegelhalter, D. J. (1986). Probabilistic prediction in patient
           management and clinical trials. Statistics in Medicine, 5(5), 421-433.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle sample weights
    if sample_weight is None:
        weights = None
    else:
        weights = np.asarray(sample_weight)
    
    # Brier score: E[(y - p)²]
    if weights is None:
        brier_score = np.mean((y_true - y_pred) ** 2)
    else:
        brier_score = np.average((y_true - y_pred) ** 2, weights=weights)
    
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds='clip')
    sorted_idx = np.argsort(y_pred)
    if weights is not None:
        iso.fit(y_pred[sorted_idx], y_true[sorted_idx], 
                sample_weight=weights[sorted_idx])
    else:
        iso.fit(y_pred[sorted_idx], y_true[sorted_idx])
    calibrated = iso.predict(y_pred)
    
    if weights is None:
        refinement = np.mean((y_true - calibrated) ** 2)
    else:
        refinement = np.average((y_true - calibrated) ** 2, weights=weights)
    
    calibration = brier_score - refinement
    
    if weights is None:
        calibration_term = np.mean((y_true - y_pred) * (1 - 2 * y_pred))
        spread_term = np.mean(y_pred * (1 - y_pred))
    else:
        calibration_term = np.average((y_true - y_pred) * (1 - 2 * y_pred), weights=weights)
        spread_term = np.average(y_pred * (1 - y_pred), weights=weights)
    
    return {
        'brier_score': float(brier_score),
        'refinement': float(refinement),
        'calibration': float(calibration),
        'calibration_term': float(calibration_term),
        'spread_term': float(spread_term),
    }


def brier_refinement_score(y_true, y_pred, sample_weight=None):
    """Brier-based refinement: Brier score AFTER optimal recalibration.
    
    This is the TRUE refinement from Berta et al. (2025):
        Refinement = min_g E[(y - g(p))²]
    
    where g is the optimal recalibration function (isotonic regression).
    
    This is the part of Brier score you CANNOT fix after training,
    making it the correct early stopping criterion.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
        
    Returns
    -------
    refinement : float
        Brier score after optimal recalibration.
        This is the irreducible error - cannot be fixed post-hoc.
        
    Examples
    --------
    Use as early stopping criterion:
    
    >>> for epoch in range(max_epochs):
    ...     model.train_one_epoch()
    ...     val_probs = model.predict_proba(X_val)[:, 1]
    ...     ref_score = brier_refinement_score(y_val, val_probs)
    ...     if ref_score < best_score:
    ...         best_score = ref_score
    ...         best_model = copy.deepcopy(model)
    
    Notes
    -----
    Uses isotonic regression as the optimal recalibration function.
    For Brier score, isotonic regression is the theoretically optimal
    recalibrator (minimizes expected squared error).
    
    See Also
    --------
    ts_refinement_loss : Log-loss based refinement (temperature scaling)
    brier_decomposition : Full Brier score decomposition
    
    References
    ----------
    .. [1] Berta, M., Ciobanu, S., & Heusinger, M. (2025). Rethinking Early
           Stopping: Refine, Then Calibrate. arXiv:2501.19195.
           https://arxiv.org/abs/2501.19195
    """
    from sklearn.isotonic import IsotonicRegression
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    iso = IsotonicRegression(out_of_bounds='clip')
    sorted_idx = np.argsort(y_pred)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        iso.fit(y_pred[sorted_idx], y_true[sorted_idx], 
                sample_weight=sample_weight[sorted_idx])
    else:
        iso.fit(y_pred[sorted_idx], y_true[sorted_idx])
    calibrated = iso.predict(y_pred)
    
    # Brier score AFTER recalibration = refinement
    if sample_weight is None:
        return float(np.mean((y_true - calibrated) ** 2))
    else:
        return float(np.average((y_true - calibrated) ** 2, weights=sample_weight))


def brier_calibration_score(y_true, y_pred, sample_weight=None):
    """Brier-based calibration error: the FIXABLE portion.
    
    This is Brier - Refinement from the variational decomposition
    (Berta et al. 2025), representing the loss that can be eliminated
    by post-hoc recalibration.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
        
    Returns
    -------
    calibration : float
        The calibration error (Brier - Refinement).
        Always >= 0. Lower is better.
        
    Notes
    -----
    This uses the variational definition from Berta et al. (2025):
        Calibration = Brier - min_g E[(y - g(p))²]
    
    NOT Spiegelhalter's (x-p)(1-2p) term (which can be negative).
        
    See Also
    --------
    calibration_loss : Log-loss based calibration error
    brier_decomposition : Full Brier score decomposition
    
    References
    ----------
    .. [1] Berta, M., Ciobanu, S., & Heusinger, M. (2025). Rethinking Early
           Stopping: Refine, Then Calibrate. arXiv:2501.19195.
           https://arxiv.org/abs/2501.19195
    """
    decomp = brier_decomposition(y_true, y_pred, sample_weight=sample_weight)
    return decomp['calibration']
