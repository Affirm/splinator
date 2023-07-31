from __future__ import absolute_import, division

import numpy as np

from splinator.metrics import (
    expected_calibration_error,
    spiegelhalters_z_statistic,
)
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
