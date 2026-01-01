"""Framework-specific metric wrappers.

This module provides a factory function to create metric wrappers
compatible with various ML frameworks (sklearn, XGBoost, LightGBM, PyTorch).

The wrappers handle framework-specific signatures and automatically
extract sample weights where available.
"""

import numpy as np
from sklearn.metrics import make_scorer
from scipy.special import expit


def make_metric_wrapper(
    metric_fn,
    framework,
    name=None,
    higher_is_better=False,
):
    """Create framework-specific metric wrapper from any splinator metric function.
    
    This factory function creates wrappers for sklearn, XGBoost, LightGBM,
    and PyTorch, handling framework-specific signatures and data extraction.
    
    Parameters
    ----------
    metric_fn : callable
        Metric function with signature (y_true, y_pred, sample_weight=None).
        For example: ts_refinement_loss, calibration_loss.
    framework : {'sklearn', 'xgboost', 'lightgbm', 'pytorch'}
        Target framework for the wrapper.
    name : str, optional
        Metric name for display. Defaults to metric_fn.__name__.
    higher_is_better : bool, default=False
        Whether higher values are better. Typically False for loss functions.
        
    Returns
    -------
    wrapper : callable or sklearn scorer
        Framework-specific wrapper:
        - 'sklearn': sklearn scorer object
        - 'xgboost': function with signature (y_pred, dtrain) -> (name, value)
        - 'lightgbm': function with signature (y_pred, data) -> (name, value, higher_is_better)
        - 'pytorch': function that auto-converts tensors to numpy
        
    Examples
    --------
    sklearn GridSearchCV:
    
    >>> from splinator import ts_refinement_loss, make_metric_wrapper
    >>> scorer = make_metric_wrapper(ts_refinement_loss, 'sklearn')
    >>> grid = GridSearchCV(model, param_grid, scoring=scorer)
    
    XGBoost early stopping:
    
    >>> xgb_metric = make_metric_wrapper(ts_refinement_loss, 'xgboost')
    >>> model = xgb.train(
    ...     params, dtrain,
    ...     evals=[(dval, 'val')],
    ...     custom_metric=xgb_metric,
    ...     early_stopping_rounds=10,
    ... )
    
    LightGBM early stopping:
    
    >>> lgb_metric = make_metric_wrapper(ts_refinement_loss, 'lightgbm')
    >>> model = lgb.train(
    ...     params, dtrain,
    ...     valid_sets=[dval],
    ...     feval=lgb_metric,
    ...     callbacks=[lgb.early_stopping(10)],
    ... )
    
    PyTorch training loop:
    
    >>> ts_metric = make_metric_wrapper(ts_refinement_loss, 'pytorch')
    >>> for epoch in range(epochs):
    ...     with torch.no_grad():
    ...         val_probs = torch.sigmoid(model(X_val))
    ...         val_loss = ts_metric(y_val, val_probs)  # accepts tensors
    
    Notes
    -----
    For CatBoost, you need to subclass catboost.CatBoostMetric directly.
    See CatBoost documentation for custom metric examples.
    
    See Also
    --------
    splinator.ts_refinement_loss : Refinement loss metric
    splinator.calibration_loss : Calibration loss metric
    """
    if name is None:
        name = getattr(metric_fn, '__name__', 'custom_metric')
    
    if framework == 'sklearn':
        # sklearn make_scorer handles the y_pred extraction from predict_proba
        # The metric_fn expects (y_true, y_pred) where y_pred is probabilities
        def sklearn_metric(y_true, y_pred):
            # Handle predict_proba output (n_samples, 2) -> (n_samples,)
            if hasattr(y_pred, 'ndim') and y_pred.ndim == 2:
                y_pred = y_pred[:, 1]
            return metric_fn(y_true, y_pred)
        
        return make_scorer(
            sklearn_metric,
            greater_is_better=higher_is_better,
            needs_proba=True,
            response_method='predict_proba',
        )
    
    elif framework == 'xgboost':
        def xgb_wrapper(y_pred, dtrain):
            y_true = dtrain.get_label()
            # XGBoost passes raw margins (logits), convert to probabilities
            y_prob = expit(y_pred)
            # Extract weights if available
            weights = dtrain.get_weight()
            sample_weight = weights if len(weights) > 0 else None
            value = metric_fn(y_true, y_prob, sample_weight=sample_weight)
            return name, float(value)
        return xgb_wrapper
    
    elif framework == 'lightgbm':
        def lgb_wrapper(y_pred, data):
            y_true = data.get_label()
            # LightGBM passes raw margins (logits), convert to probabilities
            y_prob = expit(y_pred)
            # Extract weights if available
            weights = data.get_weight()
            sample_weight = weights if weights is not None and len(weights) > 0 else None
            value = metric_fn(y_true, y_prob, sample_weight=sample_weight)
            return name, float(value), higher_is_better
        return lgb_wrapper
    
    elif framework == 'pytorch':
        def torch_wrapper(y_true, y_pred, sample_weight=None):
            # Auto-convert tensors to numpy
            if hasattr(y_true, 'detach'):
                y_true = y_true.detach().cpu().numpy()
            if hasattr(y_pred, 'detach'):
                y_pred = y_pred.detach().cpu().numpy()
            if sample_weight is not None and hasattr(sample_weight, 'detach'):
                sample_weight = sample_weight.detach().cpu().numpy()
            return metric_fn(y_true, y_pred, sample_weight=sample_weight)
        return torch_wrapper
    
    else:
        raise ValueError(
            f"Unknown framework: {framework}. "
            f"Supported: 'sklearn', 'xgboost', 'lightgbm', 'pytorch'"
        )

