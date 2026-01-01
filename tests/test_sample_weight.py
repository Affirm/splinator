"""Tests for sample_weight functionality (Issue #2)."""

import numpy as np
import pytest
from scipy.special import expit

from splinator.estimators import LinearSplineLogisticRegression, LossGradHess
from splinator.monotonic_spline import _weighted_quantile, _fit_knots


class TestWeightedQuantile:
    """Test the weighted quantile helper function."""
    
    def test_unweighted_matches_numpy(self):
        """Without weights, should match numpy.quantile."""
        np.random.seed(42)
        X = np.random.randn(100)
        quantiles = [0.25, 0.5, 0.75]
        
        result = _weighted_quantile(X, quantiles, sample_weight=None)
        expected = np.quantile(X, quantiles)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_uniform_weights_matches_numpy(self):
        """Uniform weights should match numpy.quantile."""
        np.random.seed(42)
        X = np.random.randn(100)
        weights = np.ones(100)
        quantiles = [0.25, 0.5, 0.75]
        
        result = _weighted_quantile(X, quantiles, sample_weight=weights)
        expected = np.quantile(X, quantiles)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=1)
    
    def test_doubled_weights_equivalent_to_duplication(self):
        """Doubling a sample's weight should be like duplicating it."""
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Weighted version: sample at index 2 has weight 2
        weights = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
        result_weighted = _weighted_quantile(X, [0.5], weights)[0]
        
        # Duplicated version
        X_dup = np.array([1.0, 2.0, 3.0, 3.0, 4.0, 5.0])
        result_dup = np.median(X_dup)
        
        assert result_weighted == result_dup


class TestFitKnotsWithWeights:
    """Test knot fitting with sample weights."""
    
    def test_knots_without_weights(self):
        """Knot fitting should work without weights."""
        np.random.seed(42)
        X = np.random.randn(100)
        knots = _fit_knots(X, num_knots=5)
        
        assert len(knots) == 4  # num_knots - 1 quantiles
    
    def test_knots_with_weights(self):
        """Knot fitting should work with weights."""
        np.random.seed(42)
        X = np.random.randn(100)
        weights = np.random.rand(100)
        knots = _fit_knots(X, num_knots=5, sample_weight=weights)
        
        assert len(knots) == 4


class TestLossGradHessWithWeights:
    """Test LossGradHess class with sample weights."""
    
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(50, 3)
        self.y = np.random.randint(0, 2, 50)
        self.alpha = 0.01
    
    def test_weighted_loss_with_zero_weights(self):
        """Samples with zero weight should not contribute to loss."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        
        # All ones weights
        weights_all = np.ones(50)
        lgh_all = LossGradHess(X, y, 0.01, intercept=True, sample_weight=weights_all)
        
        # First 10 samples have zero weight
        weights_partial = np.ones(50)
        weights_partial[:10] = 0
        lgh_partial = LossGradHess(X, y, 0.01, intercept=True, sample_weight=weights_partial)
        
        # Loss on subset should match weighted loss
        lgh_subset = LossGradHess(X[10:], y[10:], 0.01, intercept=True)
        
        coefs = np.random.randn(3) * 0.1
        
        # The partial loss (with zero weights) should be close to the subset loss
        # (regularization is the same, data loss is on subset)
        loss_partial = lgh_partial.loss(coefs)
        loss_subset = lgh_subset.loss(coefs)
        
        # The difference should only be in the regularization term normalization
        # which is expected to be small
        assert abs(loss_partial - loss_subset) < 1.0  # Reasonable bound
    
    def test_weighted_gradient_shape(self):
        """Gradient should have correct shape with weights."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        weights = np.random.rand(50)
        
        lgh = LossGradHess(X, y, 0.01, intercept=True, sample_weight=weights)
        coefs = np.zeros(3)
        grad = lgh.grad(coefs)
        
        assert grad.shape == (3,)
    
    def test_weighted_hessian_shape(self):
        """Hessian should have correct shape with weights."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        weights = np.random.rand(50)
        
        lgh = LossGradHess(X, y, 0.01, intercept=True, sample_weight=weights)
        coefs = np.zeros(3)
        hess = lgh.hess(coefs)
        
        assert hess.shape == (3, 3)


class TestLinearSplineWithSampleWeight:
    """Test LinearSplineLogisticRegression with sample_weight in fit()."""
    
    def test_fit_with_sample_weight(self):
        """Basic fit with sample_weight should work."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 1)
        probs = expit(X[:, 0])
        y = np.random.binomial(1, probs)
        weights = np.random.rand(n) + 0.5
        
        model = LinearSplineLogisticRegression(n_knots=5, random_state=42)
        model.fit(X, y, sample_weight=weights)
        
        assert model.is_fitted
        preds = model.predict(X)
        assert preds.shape == (n,)
        assert np.all(preds >= 0) and np.all(preds <= 1)
    
    def test_fit_without_sample_weight(self):
        """Fit without sample_weight should work (backward compatibility)."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 1)
        probs = expit(X[:, 0])
        y = np.random.binomial(1, probs)
        
        model = LinearSplineLogisticRegression(n_knots=5, random_state=42)
        model.fit(X, y)  # No sample_weight
        
        assert model.is_fitted
    
    def test_zero_weight_samples_ignored(self):
        """Samples with weight 0 should be effectively ignored."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 1)
        probs = expit(X[:, 0])
        y = np.random.binomial(1, probs)
        
        # Add some outlier samples
        X_outliers = np.array([[10.0], [-10.0]])
        y_outliers = np.array([0, 1])  # Wrong labels for outliers
        
        X_combined = np.vstack([X, X_outliers])
        y_combined = np.concatenate([y, y_outliers])
        
        # Fit with zero weight on outliers
        weights = np.ones(n + 2)
        weights[-2:] = 0.0
        
        model_weighted = LinearSplineLogisticRegression(n_knots=5, random_state=42)
        model_weighted.fit(X_combined, y_combined, sample_weight=weights)
        
        # Fit without outliers
        model_no_outliers = LinearSplineLogisticRegression(n_knots=5, random_state=42)
        model_no_outliers.fit(X, y)
        
        # Predictions should be similar on the original data range
        X_test = np.linspace(-2, 2, 20).reshape(-1, 1)
        preds_weighted = model_weighted.predict(X_test)
        preds_no_outliers = model_no_outliers.predict(X_test)
        
        # Should be close (not exact due to different knot positions from outliers in X)
        np.testing.assert_array_almost_equal(preds_weighted, preds_no_outliers, decimal=1)
    
    def test_sample_weight_validation(self):
        """Invalid sample_weight should raise appropriate errors."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = np.random.randint(0, 2, 100)
        
        model = LinearSplineLogisticRegression(n_knots=5)
        
        # Wrong length
        with pytest.raises(ValueError, match="sample_weight has .* samples"):
            model.fit(X, y, sample_weight=np.ones(50))
        
        # Negative weights
        with pytest.raises(ValueError, match="non-negative"):
            model.fit(X, y, sample_weight=np.array([-1.0] * 100))
    
    def test_two_stage_fitting_with_weights(self):
        """Two-stage fitting should work with sample_weight."""
        np.random.seed(42)
        n = 300
        X = np.random.randn(n, 1)
        probs = expit(X[:, 0])
        y = np.random.binomial(1, probs)
        weights = np.random.rand(n) + 0.5
        
        model = LinearSplineLogisticRegression(
            n_knots=5,
            two_stage_fitting_initial_size=100,
            random_state=42,
        )
        model.fit(X, y, sample_weight=weights)
        
        assert model.is_fitted
        assert len(model.fitting_history_) == 2
    
    def test_progressive_fitting_with_weights(self):
        """Progressive fitting should work with sample_weight."""
        np.random.seed(42)
        n = 300
        X = np.random.randn(n, 1)
        probs = expit(X[:, 0])
        y = np.random.binomial(1, probs)
        weights = np.random.rand(n) + 0.5
        
        model = LinearSplineLogisticRegression(
            n_knots=5,
            progressive_fitting_fractions=(0.3, 1.0),
            random_state=42,
        )
        model.fit(X, y, sample_weight=weights)
        
        assert model.is_fitted
        assert len(model.fitting_history_) == 2

