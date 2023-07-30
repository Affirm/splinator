from __future__ import absolute_import, division, print_function

import itertools
import unittest
import numpy as np
from splinator.monotonic_spline import Monotonicity
from splinator.estimators import LinearSplineLogisticRegression
from splinator.tests.common import (
    generate_piecewise_function,
    generate_training_data,
)
from splinator.tests.test_helpers import assert_allclose_absolute


class TestFitOnSyntheticData(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        options = dict(
            num_additional_inputs=[0, 1],
            num_knots=[5, 10],
            monotonicity=[
                Monotonicity.increasing.value,
                Monotonicity.decreasing.value,
                Monotonicity.none.value],
            intercept=[True],
            minimizer_options=[{'maxiter': 500}],
            method=['SLSQP', 'trust-constr']
        )
        keys, values = zip(*options.items())
        self.options_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    def test_fit_on_piecewise_function_with_exact_knots(self):
        # type: () -> None
        """
        If the input data are generated from a piecewise linear function, then the spline
        should fit the function exactly.
        """
        for permutation in self.options_dicts:
            fn = generate_piecewise_function(
                permutation['num_knots'],
                permutation['monotonicity'],
                permutation['num_additional_inputs'],
            )
            X_values, y_values, true_prob_values = generate_training_data(fn)

            exact_knots_model = LinearSplineLogisticRegression(
                intercept=True,
                monotonicity=permutation['monotonicity'],
                knots=fn.knots_x_values,
                n_knots=None,
                method=permutation['method'],
                minimizer_options=permutation['minimizer_options'],
            )

            exact_knots_model.fit(X_values, y_values)
            assert_allclose_absolute(
                exact_knots_model.predict(X_values), true_prob_values, atol=0.05, allowed_not_close_fraction=0.05
            )

    def test_fit_on_piecewise_function_with_only_num_knots(self):
        """
        If the input data are generated from a piecewise linear function, then the spline
        should fit the function exactly.
        """
        for permutation in self.options_dicts:
            fn = generate_piecewise_function(
                permutation['num_knots'],
                permutation['monotonicity'],
                permutation['num_additional_inputs'],
            )
            X_values, y_values, true_prob_values = generate_training_data(fn)

            # Model trained without specifying the knots might not be as accurate
            # we increase number of knots to better fit it
            fit_n_knots_model = LinearSplineLogisticRegression(
                intercept=True,
                monotonicity=permutation['monotonicity'],
                n_knots=30,
                knots=None,
                method=permutation['method'],
                minimizer_options=permutation['minimizer_options'],
            )
            fit_n_knots_model.fit(X_values, y_values)
            assert_allclose_absolute(
                fit_n_knots_model.predict(X_values), true_prob_values, atol=0.08, allowed_not_close_fraction=0.08
            )
