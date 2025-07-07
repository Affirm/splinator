from __future__ import annotations
from bisect import bisect_left
import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_consistent_length


def _find_fractile(score: float, sorted_calib_scores: np.ndarray) -> float:
    """
    Finds the fractile of a score within a sorted array of calibration scores.
    This is also known as the percentile rank.
    """
    if score < sorted_calib_scores[0]:
        return 0.0
    if score > sorted_calib_scores[-1]:
        return 1.0

    # Use binary search to find the insertion point
    idx = bisect_left(sorted_calib_scores, score)

    # Handle cases where score is exactly at a calibration point
    if idx < len(sorted_calib_scores) and sorted_calib_scores[idx] == score:
        # If there are duplicates, find the last occurrence for a better fractile estimate
        # searchsorted with side='right' gives insertion point that comes after existing entries
        # This will give the fraction of elements <= score
        return np.searchsorted(sorted_calib_scores, score, side='right') / len(sorted_calib_scores)

    if idx == 0:
        # Should be caught by score < sorted_calib_scores[0] but as a safeguard
        return 0.0

    # Linear interpolation of the fractile
    score_before = sorted_calib_scores[idx - 1]
    score_after = sorted_calib_scores[idx]

    # Find fractiles corresponding to these scores
    fractile_before = np.searchsorted(sorted_calib_scores, score_before, side='right') / len(sorted_calib_scores)
    fractile_after = np.searchsorted(sorted_calib_scores, score_after, side='right') / len(sorted_calib_scores)

    if score_after == score_before:
        return fractile_after

    t = (score - score_before) / (score_after - score_before)
    return fractile_before + t * (fractile_after - fractile_before)


class _InterpolatedFunction:
    """Helper class to perform linear interpolation."""
    def __init__(self, x: np.ndarray, y: np.ndarray):
        # Sort points by x to ensure correct interpolation
        sorted_indices = np.argsort(x)
        self.x = x[sorted_indices]
        self.y = y[sorted_indices]

    def __call__(self, x_new: float) -> float:
        if x_new <= self.x[0]:
            return self.y[0]
        if x_new >= self.x[-1]:
            return self.y[-1]

        # Find the insertion point for x_new
        idx = bisect_left(self.x, x_new)

        # Handle cases where x_new is exactly one of the points
        if idx < len(self.x) and self.x[idx] == x_new:
            return self.y[idx]

        x_before, y_before = self.x[idx - 1], self.y[idx - 1]
        x_after, y_after = self.x[idx], self.y[idx]
        
        # Perform linear interpolation
        # Special case for vertical line segments to avoid division by zero
        if x_after == x_before:
            return y_after
            
        t = (x_new - x_before) / (x_after - x_before)
        return y_before + t * (y_after - y_before)


class CDFSplineCalibrator(BaseEstimator, TransformerMixin):
    """
    Calibrates classifier scores using a spline-based method on cumulative distributions.

    This method implements the spline-based calibration technique described in [1]_.
    It is a binning-free calibration method that fits a cubic spline to the empirical 
    cumulative distribution functions (CDFs) of both the predicted scores and actual 
    outcomes. The derivative of this spline serves as the recalibration function, 
    providing smooth and well-calibrated probability estimates.
    
    The algorithm works by:
    1. Computing empirical CDFs for predicted scores and true class accuracies
    2. Fitting a cubic spline to model the relationship between these CDFs
    3. Using the spline derivative to transform uncalibrated scores

    Parameters
    ----------
    num_knots : int, default=6
        The number of knots to use for the cubic spline. Following the original paper,
        this helps prevent overfitting by controlling the smoothness of the calibration
        function. More knots allow for more flexible calibration curves.

    Attributes
    ----------
    n_classes_ : int
        Number of classes detected during fit.
    recalibration_functions_ : list
        List of interpolation functions for each class.

    Examples
    --------
    >>> import numpy as np
    >>> from splinator.estimators import CDFSplineCalibrator
    >>> # Generate some dummy data (e.g., from a model)
    >>> n_samples, n_classes = 100, 3
    >>> calib_scores = np.random.rand(n_samples, n_classes)
    >>> calib_labels = np.random.randint(0, n_classes, n_samples)
    >>> # Normalize scores to represent probabilities
    >>> calib_scores /= calib_scores.sum(axis=1, keepdims=True)
    >>>
    >>> # Fit the calibrator
    >>> calibrator = CDFSplineCalibrator()
    >>> calibrator.fit(calib_scores, calib_labels)
    >>>
    >>> # Recalibrate new scores
    >>> test_scores = np.random.rand(50, n_classes)
    >>> test_scores /= test_scores.sum(axis=1, keepdims=True)
    >>> calibrated_scores = calibrator.transform(test_scores)

    References
    ----------
    .. [1] Gupta, C., Koren, A., & Mishra, K. (2021). "Calibration of Neural Networks 
           using Splines". International Conference on Learning Representations (ICLR).
           arXiv:2006.12800. https://arxiv.org/abs/2006.12800

    See Also
    --------
    Official implementation by the authors: https://github.com/kartikgupta-at-anu/spline-calibration

    Notes
    -----
    This implementation extends the original method to handle edge cases and numerical
    stability issues. The spline fitting is performed independently for each class,
    making it suitable for multi-class calibration problems.
    """

    def __init__(self, num_knots: int = 6):
        # num_knots is kept for API consistency with the paper, though not used in this impl.
        self.num_knots = num_knots

    def fit(self, X: np.ndarray, y: np.ndarray) -> CDFSplineCalibrator:
        """
        Fit the spline-based calibration model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_classes)
            The predicted scores (probabilities) from a classifier for the calibration set.
        y : array-like of shape (n_samples,)
            The true class labels for the calibration set.

        Returns
        -------
        self : CDFSplineCalibrator
            The fitted calibrator instance.
        """
        X = check_array(X, ensure_2d=True, accept_sparse=False, dtype=[np.float64, np.float32])
        y = check_array(y, ensure_2d=False, ensure_min_samples=1, dtype=None)
        check_consistent_length(X, y)

        if X.shape[1] < 2:
            raise ValueError("CDFSplineCalibrator requires at least 2 classes.")

        self.n_classes_ = X.shape[1]
        self.splines_ = []
        self.recalibration_functions_ = []

        for k in range(self.n_classes_):
            scores_k = X[:, k]
            is_correct_k = (y == k).astype(int)

            sort_indices = np.argsort(scores_k)
            sorted_scores = scores_k[sort_indices]
            sorted_is_correct = is_correct_k[sort_indices]

            n_calib = len(sorted_scores)
            if n_calib == 0 or len(np.unique(sorted_scores)) < 2:
                self.recalibration_functions_.append(None)
                continue

            integrated_accuracy = np.cumsum(sorted_is_correct) / n_calib
            integrated_scores = np.cumsum(sorted_scores) / n_calib
            
            # The x-coordinates for the spline are percentiles
            percentiles = np.linspace(0.0, 1.0, n_calib)

            # The y-coordinates are the difference
            y_spline = integrated_accuracy - integrated_scores

            # Subsample to get knots for the spline, as specified in the paper
            # This prevents overfitting and creates a smoother calibration function.
            knot_indices = np.round(np.linspace(0, n_calib - 1, self.num_knots)).astype(int)
            knot_x = percentiles[knot_indices]
            knot_y = y_spline[knot_indices]

            # CubicSpline requires unique x coordinates.
            unique_knot_x, unique_indices = np.unique(knot_x, return_index=True)
            unique_knot_y = knot_y[unique_indices]
            
            # A cubic spline requires at least k+1 (i.e., 4) points.
            if len(unique_knot_x) < 4:
                self.recalibration_functions_.append(None)
                continue

            # Fit a natural cubic spline on the knots
            spline = CubicSpline(unique_knot_x, unique_knot_y, bc_type='natural')
            
            # The calibrated scores are the derivative of the spline + original scores
            calibrated_scores = spline(percentiles, nu=1) + sorted_scores
            
            # This function maps original scores to calibrated scores
            recal_func = _InterpolatedFunction(sorted_scores, calibrated_scores)
            self.recalibration_functions_.append(recal_func)

        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the learned calibration to new scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_classes)
            The scores to recalibrate.

        Returns
        -------
        recalibrated_X : ndarray of shape (n_samples, n_classes)
            The recalibrated scores (probabilities).
        """
        X = check_array(X, ensure_2d=True, accept_sparse=False, dtype=[np.float64, np.float32])
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError("This CDFSplineCalibrator instance is not fitted yet.")
        if X.shape[1] != self.n_classes_:
            raise ValueError(f"The number of classes in X ({X.shape[1]}) does not match "
                             f"the number of classes during fit ({self.n_classes_}).")

        recalibrated_X = np.zeros_like(X, dtype=float)

        for k in range(self.n_classes_):
            recal_func = self.recalibration_functions_[k]

            if recal_func is None:
                recalibrated_X[:, k] = X[:, k]  # Return original scores
                continue

            scores_to_recalibrate = X[:, k]
            recalibrated_scores = np.array([recal_func(s) for s in scores_to_recalibrate])
            
            recalibrated_X[:, k] = np.clip(recalibrated_scores, 0, 1)

        # A small adjustment to the normalization logic for numerical stability.
        # If all recalibrated scores for a sample are 0, this prevents NaN results
        # by distributing the probability mass uniformly.
        row_sums = recalibrated_X.sum(axis=1, keepdims=True)
        # Avoid division by zero for rows that sum to 0
        is_zero_sum = row_sums == 0
        # Replace 0s with 1s to avoid division by zero
        safe_row_sums = np.where(is_zero_sum, 1, row_sums)
        recalibrated_X /= safe_row_sums
        # For rows that originally summed to 0, set a uniform probability
        recalibrated_X[is_zero_sum.flatten(), :] = 1 / self.n_classes_

        return recalibrated_X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for samples in X.

        This is a convenience method that first transforms the scores and then
        predicts the class with the highest probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_classes)
            The scores to make predictions on.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        recalibrated_probs = self.transform(X)
        return np.argmax(recalibrated_probs, axis=1) 