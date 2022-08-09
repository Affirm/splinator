from __future__ import absolute_import, division, print_function

from typing import TYPE_CHECKING
import itertools
from typing import NamedTuple
import unittest
import numpy as np
import pandas as pd

from splinator.monotonic_spline import Monotonicity
from splinator.esimators import LinearSplineLogisticRegression, MinimizationMethod

if TYPE_CHECKING:
    from typing import Optional, Iterable, List, Tuple


PiecewiseFunction = NamedTuple(
    'PiecewiseFunction',
    [
        ('knots', np.ndarray),
        ('knot_values', np.ndarray),
        ('slopes', np.ndarray),
        ('additional_inputs_coefs', np.ndarray),
    ],
)


def f_logit(p):
    return np.log(p / (1 - p))


def f_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_model_configs(additional_base_options=None, include_none_monotonicities=True):
    # type: (Optional[dict] ,bool) -> Iterable[LogisticSplineEstimator]

    base_options = {'input_column': 'x'}
    if additional_base_options:
        base_options.update(additional_base_options)

    monotonicity_args = [
        {'monotonicity': Monotonicity.increasing},
        {'monotonicity': Monotonicity.decreasing},
    ]
    if include_none_monotonicities:
        monotonicity_args.append({'monotonicity': Monotonicity.none})

    optionses = [
        [{'knots': 10}, {'knots': np.array([0.1, 0.4, 0.9])}],
        monotonicity_args,
        [{'intercept': True}, {'intercept': False}],
        [
            {'additional_inputs_columns': []},
            {'additional_inputs_columns': ['z']},
            {'additional_inputs_columns': ['z1', 'z2', 'z3', 'z4', 'z5']},
        ],
        [{'method': MinimizationMethod.slsqp}, {'method': MinimizationMethod.trust_constr}],
        [
            {'two_stage_fitting_enabled': True, 'two_stage_fitting_initial_size': 80},
            {'two_stage_fitting_enabled': False, 'two_stage_fitting_initial_size': None},
        ],
        [{'C': 1}, {'C': 10}],
    ]  # type: List[List[dict]]

    for options in itertools.product(*optionses):
        args_dict = base_options.copy()  # type: dict
        for option in options:
            args_dict.update(option)
        yield LogisticSplineEstimator(**args_dict)


def generate_sample_data(model, size=100):
    # type: (LogisticSplineEstimator, int) -> Tuple[pd.DataFrame, np.ndarray]
    column_names = [model.input_column] + model.additional_inputs_columns
    X = np.random.uniform(size=(size, len(column_names)))
    X_df = pd.DataFrame(data=X, columns=column_names)
    X_df['y'] = np.random.randint(2, size=size)
    return X_df, X_df['y']


def generate_piecewise_function(num_knots, monotonicity, num_additional_inputs):
    # type: (int, Monotonicity, int) -> PiecewiseFunction
    """
    Generates random knot values and slopes and additional inputs coefficients.
    Also picks nonzero slopes for the region left of the first knot and right of the last knot.
    """
    # Since we are fitting in the log odds space, we limit the values of piecewise functions to
    # be within a certain range to avoid extreme probability values.
    if num_additional_inputs == 0:
        knot_value_lower = -4.5  # ~ 0.01 in probability space
        knot_value_upper = 4.5  # ~ 0.99 in probability space
    else:
        # if we have additional input, we reserve some space for
        # additional inputs & coefs to affect the knot values
        knot_value_lower = -3.5  # ~ 0.03 in probability space
        knot_value_upper = 3.5  # ~ 0.97 in probability space

    # depending on monotonicity, we have knot values in different orders
    if monotonicity == Monotonicity.none:
        knot_values = np.random.uniform(knot_value_lower, knot_value_upper, size=num_knots)
    elif monotonicity == Monotonicity.increasing:
        knot_values = np.sort(np.random.uniform(knot_value_lower, knot_value_upper, size=num_knots))
    elif monotonicity == Monotonicity.decreasing:
        knot_values = np.sort(np.random.uniform(knot_value_lower, knot_value_upper, size=num_knots))[::-1]

    knot_values_diffs = np.diff(knot_values)

    # we want the slopes to be within (-5, 5)
    # slope = knot_values_diff / knots_diff => knots_diff = knot_values_diff / slope
    min_slope = 0.5
    max_slope = 1.5

    knots_diffs = np.random.uniform(
        low=np.abs(knot_values_diffs) / max_slope, high=np.abs(knot_values_diffs) / min_slope
    )

    knots_first = np.random.uniform(-8, -5)
    knots = np.concatenate(([knots_first], knots_diffs)).cumsum()

    def generate_slope_based_on_monotonicity():
        if monotonicity == Monotonicity.increasing:
            return np.random.uniform(0, 1)
        elif monotonicity == Monotonicity.decreasing:
            return np.random.uniform(-1, 0)
        elif monotonicity == Monotonicity.none:
            return np.random.uniform(-1, 1)

    slopes = np.concatenate(
        [
            [generate_slope_based_on_monotonicity()],
            knot_values_diffs / knots_diffs,
            [generate_slope_based_on_monotonicity()],
        ]
    )

    if num_additional_inputs > 0:
        additional_inputs_coefs = np.random.uniform(
            -1 / num_additional_inputs, 1 / num_additional_inputs, size=num_additional_inputs
        )
    else:
        additional_inputs_coefs = None

    return PiecewiseFunction(knots, knot_values, slopes, additional_inputs_coefs)


def generate_training_data(input_column, additional_inputs_columns, fn):
    # type: (str, List[str], PiecewiseFunction) -> pd.DataFrame
    """
    Samples X, additional_inputs, y data from a PiecewiseFunction.
    Samples points left of the first knot and right of the last knots with x values
    in the range (knots[0] - 5, knots[0]) and (knots[-1], knots[-1] + 5).
    """
    x_points = np.concatenate([[fn.knots[0] - 0.1], fn.knots, [fn.knots[-1] + 0.1]])

    log_odds_points = np.concatenate(
        [[fn.knot_values[0] - 0.1 * fn.slopes[0]], fn.knot_values, [fn.knot_values[-1] + 0.1 * fn.slopes[-1]]]
    )

    X_list = []  # type: List[np.ndarray]
    log_odds_list = []  # type: List[np.ndarray]
    num_of_points = 2000
    for x0, log_odds_0, x1, log_odds_1 in np.array(
        [x_points[:-1], log_odds_points[:-1], x_points[1:], log_odds_points[1:]]
    ).T:
        X_ = np.random.uniform(x0, x1, size=num_of_points)
        log_odds_ = np.interp(X_, [x0, x1], [log_odds_0, log_odds_1])
        X_list.append(X_)
        log_odds_list.append(log_odds_)

    X_data = np.concatenate(X_list)
    log_odds_data = np.concatenate(log_odds_list)
    assert X_data.shape == ((fn.knots.shape[0] + 1) * num_of_points,)
    assert log_odds_data.shape == ((fn.knots.shape[0] + 1) * num_of_points,)

    # Generate additional inputs and combine to get X and y
    if fn.additional_inputs_coefs is not None:
        additional_inputs = np.random.uniform(-1, 1, size=(X_data.shape[0], fn.additional_inputs_coefs.shape[0]))
        log_odds = log_odds_data + additional_inputs.dot(fn.additional_inputs_coefs)
    else:
        additional_inputs = None
        log_odds = log_odds_data

    X_df = pd.DataFrame(
        {input_column: X_data, 'y': np.random.binomial(n=1, p=f_sigmoid(log_odds)), 'prob': f_sigmoid(log_odds)}
    )

    if additional_inputs is not None:
        for col_name, additional_input_data in zip(additional_inputs_columns, additional_inputs.T):
            X_df[col_name] = additional_input_data

    return X_df


class TestFit(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_fit_on_piecewise_function(self):
        # type: () -> None
        """
        If the input data are generated from a piecewise linear function, then the spline
        should fit the function exactly.
        """
        for num_additional_inputs in [0, 1, 5]:
            additional_inputs_columns = ['z{}'.format(i) for i in range(num_additional_inputs)]
            for monotonicity in Monotonicity:
                num_knots = 5

                fn = generate_piecewise_function(num_knots, monotonicity, num_additional_inputs)
                X_df = generate_training_data('x', additional_inputs_columns, fn)

                # Model trained with exactly the right knot locations should be really accurate
                same_knots_model = LogisticSplineEstimator(
                    intercept=True,
                    monotonicity=monotonicity,
                    additional_inputs_columns=additional_inputs_columns,
                    input_column='x',
                    knots=fn.knots,
                )

                same_knots_model.fit(X_df, X_df['y'])
                np.testing.assert_allclose(
                    same_knots_model.predict(X_df), X_df['prob'], atol=0.05
                )
                # Model trained without specifying the knots might not be as accurate
                fit_knots_model = LinearSplineLogisticRegression(
                    intercept=True,
                    monotonicity=monotonicity,
                    additional_inputs_columns=additional_inputs_columns,
                    input_column='x',
                    knots=10,
                )
                fit_knots_model.fit(X_df, X_df['y'])
                np.testing.assert_allclose(
                    fit_knots_model.predict(X_df), X_df['prob'], atol=0.08
                )