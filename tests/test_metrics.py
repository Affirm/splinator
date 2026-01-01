from __future__ import absolute_import, division

import numpy as np
import pytest

from splinator.metrics import (
    expected_calibration_error,
    spiegelhalters_z_statistic,
    ts_refinement_loss,
    calibration_loss,
    logloss_decomposition,
)
from splinator.metric_wrappers import make_metric_wrapper
import unittest


class TestMetrics(unittest.TestCase):
    def test_spiegelhalters_z_statistic(self):
        labels = np.array([1, 0])

        scores_equal = np.array([0.2, 0.2])
        szs_equal = spiegelhalters_z_statistic(labels, scores_equal)
        self.assertAlmostEqual(1.06066, szs_equal, places=3)

        scores_diff = np.array([0.4, 0.5])
        szs_diff = spiegelhalters_z_statistic(labels, scores_diff)
        self.assertAlmostEqual(1.22474, szs_diff, places=3)

    def test_expected_calibration_error(self):
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0, 0, 0.1, 0.8, 0.2, 0.3, 0.7, 0.9, 0.9, 1])
        # The scores will be ranked and binned.
        # For each bin, we compute the absolute difference and compute the average.
        # 1st bin labels: [0, 0, 0, 0, 1] | scores: [0, 0, 0.1, 0.2, 0.3]
        # 1st bin absolute average diff = 0.08
        # 2nd bin labels: [0, 1, 1, 1, 1] | scores: [0.7, 0.8, 0.9, 0.9, 1]
        # 2nd bin absolute average diff = 0.06
        # ece should be 0.5*(0.08+0.06) = 0.07
        ece = expected_calibration_error(labels, scores, n_bins=2)
        self.assertAlmostEqual(0.07, ece, places=3)


class TestTSRefinementLoss:
    """Tests for ts_refinement_loss function."""
    
    def test_basic_calculation(self):
        """Test basic refinement loss calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.3, 0.7, 0.9])
        
        loss = ts_refinement_loss(y_true, y_pred)
        assert loss > 0
        assert np.isfinite(loss)
    
    def test_refinement_less_than_or_equal_total(self):
        """Refinement loss should be <= total loss."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.2, 0.4, 0.6, 0.8])
        
        from splinator.temperature_scaling import _weighted_cross_entropy
        total = _weighted_cross_entropy(y_true, y_pred)
        refinement = ts_refinement_loss(y_true, y_pred)
        
        assert refinement <= total + 1e-10  # Small tolerance for numerical errors
    
    def test_with_sample_weights(self):
        """Test refinement loss with sample weights."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.3, 0.7, 0.9])
        weights = np.array([1.0, 2.0, 1.0, 2.0])
        
        loss = ts_refinement_loss(y_true, y_pred, sample_weight=weights)
        assert loss > 0
        assert np.isfinite(loss)


class TestCalibrationLoss:
    """Tests for calibration_loss function."""
    
    def test_non_negative(self):
        """Calibration loss should be non-negative."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.3, 0.7, 0.9])
        
        loss = calibration_loss(y_true, y_pred)
        assert loss >= -1e-10  # Allow small numerical tolerance
    
    def test_well_calibrated_has_low_calibration_loss(self):
        """Well-calibrated predictions should have low calibration loss."""
        np.random.seed(42)
        n = 1000
        
        # Generate well-calibrated predictions
        y_pred = np.random.uniform(0.1, 0.9, n)
        y_true = (np.random.uniform(0, 1, n) < y_pred).astype(int)
        
        loss = calibration_loss(y_true, y_pred)
        # Should be relatively small for well-calibrated predictions
        assert loss < 0.1
    
    def test_miscalibrated_has_higher_calibration_loss(self):
        """Miscalibrated predictions should have higher calibration loss."""
        np.random.seed(42)
        n = 1000
        
        # Generate well-calibrated predictions
        true_prob = np.random.uniform(0.3, 0.7, n)
        y_true = (np.random.uniform(0, 1, n) < true_prob).astype(int)
        
        # Well-calibrated
        y_pred_good = true_prob
        
        # Overconfident (miscalibrated)
        y_pred_bad = np.where(true_prob > 0.5, 0.95, 0.05)
        
        loss_good = calibration_loss(y_true, y_pred_good)
        loss_bad = calibration_loss(y_true, y_pred_bad)
        
        assert loss_bad > loss_good


class TestLossDecomposition:
    """Tests for logloss_decomposition function."""
    
    def test_returns_all_keys(self):
        """Should return dict with all expected keys."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.3, 0.7, 0.9])
        
        result = logloss_decomposition(y_true, y_pred)
        
        assert 'total_loss' in result
        assert 'refinement_loss' in result
        assert 'calibration_loss' in result
        assert 'calibration_fraction' in result
        assert 'optimal_temperature' in result
    
    def test_decomposition_adds_up(self):
        """Total loss should equal refinement + calibration."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.2, 0.4, 0.6, 0.8])
        
        result = logloss_decomposition(y_true, y_pred)
        
        expected_total = result['refinement_loss'] + result['calibration_loss']
        np.testing.assert_almost_equal(
            result['total_loss'], expected_total, decimal=5
        )
    
    def test_calibration_fraction_in_valid_range(self):
        """Calibration fraction should be in [0, 1]."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.3, 0.7, 0.9])
        
        result = logloss_decomposition(y_true, y_pred)
        
        assert 0 <= result['calibration_fraction'] <= 1
    
    def test_optimal_temperature_positive(self):
        """Optimal temperature should be positive."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.3, 0.7, 0.9])
        
        result = logloss_decomposition(y_true, y_pred)
        
        assert result['optimal_temperature'] > 0


class TestMakeMetricWrapper:
    """Tests for make_metric_wrapper factory function."""
    
    def test_sklearn_wrapper(self):
        """Test sklearn scorer wrapper."""
        scorer = make_metric_wrapper(ts_refinement_loss, 'sklearn')
        
        # Should be a sklearn scorer object
        assert hasattr(scorer, '_score_func')
    
    def test_xgboost_wrapper(self):
        """Test XGBoost wrapper signature."""
        wrapper = make_metric_wrapper(ts_refinement_loss, 'xgboost', name='ts_ref')
        
        # Should be callable
        assert callable(wrapper)
        
        # Mock a DMatrix-like object
        class MockDMatrix:
            def get_label(self):
                return np.array([0, 0, 1, 1])
            def get_weight(self):
                return np.array([])
        
        # Should return (name, value) tuple
        from scipy.special import logit
        y_pred_logits = logit(np.array([0.1, 0.3, 0.7, 0.9]))
        result = wrapper(y_pred_logits, MockDMatrix())
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == 'ts_ref'
        assert isinstance(result[1], float)
    
    def test_lightgbm_wrapper(self):
        """Test LightGBM wrapper signature."""
        wrapper = make_metric_wrapper(
            ts_refinement_loss, 'lightgbm', name='ts_ref', higher_is_better=False
        )
        
        # Mock a Dataset-like object
        class MockDataset:
            def get_label(self):
                return np.array([0, 0, 1, 1])
            def get_weight(self):
                return None
        
        from scipy.special import logit
        y_pred_logits = logit(np.array([0.1, 0.3, 0.7, 0.9]))
        result = wrapper(y_pred_logits, MockDataset())
        
        # Should return (name, value, higher_is_better) tuple
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0] == 'ts_ref'
        assert isinstance(result[1], float)
        assert result[2] is False
    
    def test_pytorch_wrapper(self):
        """Test PyTorch wrapper auto-converts tensors."""
        wrapper = make_metric_wrapper(ts_refinement_loss, 'pytorch')
        
        # Test with numpy arrays (should work)
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.3, 0.7, 0.9])
        
        result = wrapper(y_true, y_pred)
        assert isinstance(result, float)
    
    def test_unknown_framework_raises(self):
        """Should raise for unknown framework."""
        with pytest.raises(ValueError, match="Unknown framework"):
            make_metric_wrapper(ts_refinement_loss, 'unknown_framework')
    
    def test_default_name_from_function(self):
        """Should use function name as default metric name."""
        wrapper = make_metric_wrapper(ts_refinement_loss, 'xgboost')
        
        class MockDMatrix:
            def get_label(self):
                return np.array([0, 0, 1, 1])
            def get_weight(self):
                return np.array([])
        
        from scipy.special import logit
        y_pred_logits = logit(np.array([0.1, 0.3, 0.7, 0.9]))
        result = wrapper(y_pred_logits, MockDMatrix())
        
        assert result[0] == 'ts_refinement_loss'
