from __future__ import absolute_import, division, print_function

from typing import TYPE_CHECKING, NamedTuple
import itertools
import numpy as np

from splinator.monotonic_spline import Monotonicity
from splinator.estimators import LinearSplineLogisticRegression, MinimizationMethod
from scipy.special import expit
if TYPE_CHECKING:
    from typing import Optional, Iterable, List, Tuple


PiecewiseFunction = NamedTuple(
    'PiecewiseFunction',
    [
        ('knots_x_values', np.ndarray),
        ('knots_y_values', np.ndarray),
        ('slopes', np.ndarray),
        ('additional_inputs_coefs', np.ndarray),
    ],
)

slsqp_options = {
    'maxiter': 100,
    'ftol': 1e-12,
    'iprint': 1,
    'disp': True,
    'eps': 1.4901161193847656e-08,
    'finite_diff_rel_step': None
}

trust_constr_options = {
    'maxiter': 2000,
    'ftol': 1e-12,
    'disp': True,
}


def generate_slope_based_on_monotonicity(monotonicity):
    if monotonicity == Monotonicity.increasing.value:
        return np.random.uniform(0, 1)
    elif monotonicity == Monotonicity.decreasing.value:
        return np.random.uniform(-1, 0)
    elif monotonicity == Monotonicity.none.value:
        return np.random.uniform(-1, 1)


def generate_piecewise_function(num_knots, monotonicity, num_additional_inputs):
    # type: (int, Monotonicity, int) -> PiecewiseFunction
    """
    Generates random knot values and slopes and additional inputs coefficients.
    Also picks nonzero slopes for the region left of the first knot and right of the last knot.
    """
    # Since we are fitting in the log odds space, we limit the values of piecewise functions to
    # be within a certain range to avoid extreme probability values.
    if num_additional_inputs == 0:
        knot_y_lower = -4.5  # ~ 0.01 in probability space
        knot_y_upper = 4.5  # ~ 0.99 in probability space
    else:
        # if we have additional input, we reserve some space for
        # additional inputs & coefs to affect the knot values
        knot_y_lower = -3.5  # ~ 0.03 in probability space
        knot_y_upper = 3.5  # ~ 0.97 in probability space

    # depending on monotonicity, we have knot values in different orders
    if monotonicity == Monotonicity.none.value:
        knots_y_values = np.random.uniform(knot_y_lower, knot_y_upper, size=num_knots)
    elif monotonicity == Monotonicity.increasing.value:
        knots_y_values = np.sort(np.random.uniform(knot_y_lower, knot_y_upper, size=num_knots))
    elif monotonicity == Monotonicity.decreasing.value:
        knots_y_values = np.sort(np.random.uniform(knot_y_lower, knot_y_upper, size=num_knots))[::-1]

    knot_values_diffs = np.diff(knots_y_values)

    # slope = knot_values_diff / knots_diff => knots_diff = knot_values_diff / slope
    min_slope = 0.5
    max_slope = 1.5

    knots_diffs = np.random.uniform(
        low=np.abs(knot_values_diffs) / max_slope, high=np.abs(knot_values_diffs) / min_slope
    )

    knots_first = np.random.uniform(-8, -5)
    knots_x_values = np.concatenate(([knots_first], knots_diffs)).cumsum()

    slopes = np.concatenate(
        [
            [generate_slope_based_on_monotonicity(monotonicity)],
            knot_values_diffs / knots_diffs,
            [generate_slope_based_on_monotonicity(monotonicity)],
        ]
    )

    if num_additional_inputs > 0:
        additional_inputs_coefs = np.random.uniform(
            -1 / num_additional_inputs, 1 / num_additional_inputs, size=num_additional_inputs
        )
    else:
        additional_inputs_coefs = None

    return PiecewiseFunction(knots_x_values, knots_y_values, slopes, additional_inputs_coefs)


def generate_training_data(fn, num_of_points=1000):
    # type: (str, List[str], PiecewiseFunction) -> pd.DataFrame
    """
    Samples X, additional_inputs, y data from a PiecewiseFunction.
    Samples points left of the first knot and right of the last knots with x values
    in the range (knots[0] - eps, knots[0]) and (knots[-1], knots[-1] + eps).
    """
    x_points = fn.knots_x_values

    log_odds_points = fn.knots_y_values

    input_value_list = []  # type: List[np.ndarray]
    log_odds_list = []  # type: List[np.ndarray]

    for x0, log_odds_0, x1, log_odds_1 in np.array(
        [x_points[:-1], log_odds_points[:-1], x_points[1:], log_odds_points[1:]]
    ).T:
        X_ = np.random.uniform(x0, x1, size=num_of_points)
        log_odds_ = np.interp(X_, [x0, x1], [log_odds_0, log_odds_1])
        input_value_list.append(X_)
        log_odds_list.append(log_odds_)

    input_values = np.concatenate(input_value_list).reshape(-1, 1)
    log_odds_data = np.concatenate(log_odds_list)
    assert input_values.shape == ((fn.knots_x_values.shape[0] - 1) * num_of_points, 1)
    assert log_odds_data.shape == ((fn.knots_x_values.shape[0] - 1) * num_of_points,)

    # Generate additional inputs and combine to get X and y
    if fn.additional_inputs_coefs is not None:
        additional_inputs = np.random.uniform(-1, 1, size=(input_values.shape[0], fn.additional_inputs_coefs.shape[0]))
        log_odds = log_odds_data + additional_inputs.dot(fn.additional_inputs_coefs)
        X_values = np.hstack([input_values, additional_inputs])
    else:
        log_odds = log_odds_data
        X_values = input_values

    y_values = np.random.binomial(n=1, p=expit(log_odds))
    true_prob_values = expit(log_odds)

    return X_values, y_values, true_prob_values


def generate_sample_data(model, size=100):
    # type: (LinearSplineLogisticRegression, int) -> Tuple[pd.DataFrame, np.ndarray]
    column_names = [model.input_column] + model.additional_inputs_columns
    X = np.random.uniform(size=(size, len(column_names)))
    X_df = pd.DataFrame(data=X, columns=column_names)
    X_df['y'] = np.random.randint(2, size=size)
    return X_df, X_df['y']


def generate_model_configs(additional_base_options=None, include_none_monotonicities=True):
    # type: (Optional[dict] ,bool) -> Iterable[LinearSplineLogisticRegression]

    base_options = {'input_score_column_index': 0}
    if additional_base_options:
        base_options.update(additional_base_options)

    monotonicity_args = [
        {'monotonicity': Monotonicity.increasing.value},
        {'monotonicity': Monotonicity.decreasing.value},
    ]
    if include_none_monotonicities:
        monotonicity_args.append({'monotonicity': Monotonicity.none.value})

    options = [
        [{'n_knots': 10, 'knots': None},
         {'n_knots': 100, 'knots': None},
         {'n_knots': None, 'knots': np.array([0.1, 0.4, 0.9])},
         {'n_knots': None, 'knots': np.linspace(0, 1, 10, endpoint=False)}],
        monotonicity_args,
        [{'intercept': True}, {'intercept': False}],
        [
            {'two_stage_fitting_initial_size': 1000},
            {'two_stage_fitting_initial_size': None},
        ],
        [{'C': 1}, {'C': 10}],
    ]  # type: List[List[dict]]

    for options in itertools.product(*options):
        args_dict = base_options.copy()  # type: dict
        for option in options:
            args_dict.update(option)
        yield LinearSplineLogisticRegression(**args_dict)
