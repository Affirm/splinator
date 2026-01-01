"""Tests for the progressive fitting improvements (Issue #9)."""
from __future__ import absolute_import, division, print_function

import unittest
import numpy as np
from scipy.special import expit

from splinator.estimators import LinearSplineLogisticRegression, LossGradHess
from splinator.monotonic_spline import Monotonicity


class TestHessian(unittest.TestCase):
    """Test the Hessian computation."""

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 50
        self.n_features = 5
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.randint(0, 2, self.n_samples)
        self.alpha = 0.01

    def test_hessian_shape(self):
        """Test that Hessian has correct shape."""
        lgh = LossGradHess(self.X, self.y, self.alpha, intercept=True)
        coefs = np.zeros(self.n_features)
        H = lgh.hess(coefs)
        self.assertEqual(H.shape, (self.n_features, self.n_features))

    def test_hessian_symmetry(self):
        """Test that Hessian is symmetric."""
        lgh = LossGradHess(self.X, self.y, self.alpha, intercept=True)
        coefs = np.random.randn(self.n_features)
        H = lgh.hess(coefs)
        np.testing.assert_array_almost_equal(H, H.T)

    def test_hessian_positive_semidefinite(self):
        """Test that Hessian is positive semi-definite."""
        lgh = LossGradHess(self.X, self.y, self.alpha, intercept=True)
        coefs = np.random.randn(self.n_features)
        H = lgh.hess(coefs)
        eigenvalues = np.linalg.eigvalsh(H)
        self.assertTrue(np.all(eigenvalues >= -1e-10))

    def test_hessian_numerical_gradient(self):
        """Test Hessian against numerical differentiation of gradient."""
        lgh = LossGradHess(self.X, self.y, self.alpha, intercept=True)
        coefs = np.random.randn(self.n_features) * 0.1
        H_analytical = lgh.hess(coefs)
        
        # Numerical Hessian via finite differences on gradient
        eps = 1e-5
        H_numerical = np.zeros((self.n_features, self.n_features))
        for i in range(self.n_features):
            coefs_plus = coefs.copy()
            coefs_plus[i] += eps
            coefs_minus = coefs.copy()
            coefs_minus[i] -= eps
            H_numerical[:, i] = (lgh.grad(coefs_plus) - lgh.grad(coefs_minus)) / (2 * eps)
        
        np.testing.assert_array_almost_equal(H_analytical, H_numerical, decimal=4)


class TestProgressiveFitting(unittest.TestCase):
    """Test progressive fitting functionality."""

    def setUp(self):
        np.random.seed(123)
        self.n_samples = 500
        self.X = np.random.randn(self.n_samples, 1) * 2
        probs = expit(self.X[:, 0])
        self.y = np.random.binomial(1, probs)

    def test_progressive_fitting_runs(self):
        """Test that progressive fitting runs without errors."""
        model = LinearSplineLogisticRegression(
            n_knots=10,
            progressive_fitting_fractions=(0.2, 1.0),
            random_state=42,
        )
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted)
        self.assertEqual(len(model.fitting_history_), 2)

    def test_progressive_fitting_with_monotonicity(self):
        """Test progressive fitting with monotonicity constraints."""
        model = LinearSplineLogisticRegression(
            n_knots=10,
            monotonicity='increasing',
            progressive_fitting_fractions=(0.3, 1.0),
            random_state=42,
        )
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted)

    def test_progressive_fitting_history(self):
        """Test that fitting history is recorded correctly."""
        model = LinearSplineLogisticRegression(
            n_knots=10,
            progressive_fitting_fractions=(0.2, 0.5, 1.0),
            random_state=42,
        )
        model.fit(self.X, self.y)
        
        self.assertEqual(len(model.fitting_history_), 3)
        for i, history in enumerate(model.fitting_history_):
            self.assertIn('stage', history)
            self.assertIn('n_samples', history)
            self.assertIn('fraction', history)


class TestStratifiedSampling(unittest.TestCase):
    """Test stratified sampling functionality."""

    def setUp(self):
        np.random.seed(456)
        self.n_samples = 500
        # Bimodal distribution
        X1 = np.random.randn(self.n_samples // 2, 1) - 3
        X2 = np.random.randn(self.n_samples // 2, 1) + 3
        self.X = np.vstack([X1, X2])
        probs = expit(self.X[:, 0] * 0.5)
        self.y = np.random.binomial(1, probs)

    def test_stratified_sampling_method(self):
        """Test the _stratified_subsample method directly."""
        model = LinearSplineLogisticRegression(n_knots=10, random_state=42)
        model.random_state_ = np.random.RandomState(42)
        model.input_score_column_index = 0
        
        X_sub, y_sub, weight_sub, indices = model._stratified_subsample(
            self.X, self.y, n_samples=100, n_strata=10
        )
        
        self.assertEqual(len(X_sub), 100)
        self.assertEqual(len(y_sub), 100)
        self.assertIsNone(weight_sub)
        
        # Check coverage
        original_range = np.ptp(self.X[:, 0])
        subsample_range = np.ptp(X_sub[:, 0])
        coverage = subsample_range / original_range
        self.assertGreater(coverage, 0.8)


class TestEarlyStopping(unittest.TestCase):
    """Test early stopping functionality."""

    def test_convergence_check(self):
        """Test the _check_convergence method."""
        model = LinearSplineLogisticRegression(
            n_knots=10, 
            early_stopping_tol=1e-4,
            random_state=42
        )
        
        coefs_old = np.array([1.0, 2.0, 3.0])
        coefs_similar = np.array([1.0001, 2.0001, 3.0001])
        coefs_different = np.array([1.5, 2.5, 3.5])
        
        self.assertTrue(model._check_convergence(coefs_old, coefs_similar))
        self.assertFalse(model._check_convergence(coefs_old, coefs_different))
        self.assertFalse(model._check_convergence(None, coefs_similar))


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing API."""

    def setUp(self):
        np.random.seed(999)
        self.n_samples = 300
        self.X = np.random.randn(self.n_samples, 1)
        probs = expit(self.X[:, 0])
        self.y = np.random.binomial(1, probs)

    def test_legacy_two_stage_fitting(self):
        """Test that legacy two_stage_fitting_initial_size still works."""
        model = LinearSplineLogisticRegression(
            n_knots=10,
            two_stage_fitting_initial_size=100,
            random_state=42,
        )
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted)
        self.assertEqual(len(model.fitting_history_), 2)

    def test_direct_fitting(self):
        """Test that direct fitting (no progressive) still works."""
        model = LinearSplineLogisticRegression(
            n_knots=10,
            random_state=42,
        )
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted)
        self.assertEqual(len(model.fitting_history_), 1)


if __name__ == '__main__':
    unittest.main()
