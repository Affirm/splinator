# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import cvxopt
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.sparse

if TYPE_CHECKING:
    from typing import Callable, List, Optional, Tuple, Union, Any


# Turn off logging for quadratic program solver
cvxopt.solvers.options['show_progress'] = False


def _to_cvx(mat):
    # type: (np.ndarray) -> cvxopt.base.matrix
    return cvxopt.matrix(mat.astype(np.double), tc='d')


class Monotonicity(Enum):
    none = 'none'
    increasing = 'increasing'
    decreasing = 'decreasing'


def _get_monotonicity_constraint_matrices(
    monotonicity,  # type: Monotonicity
    num_constrained,  # type: int
    num_unconstrained,  # type: int
    min_absolute_slope,  # type: Union[float, np.ndarray, None]
):
    # type: (...) -> Tuple[np.ndarray, np.ndarray]
    """
    Gets constraints such that the returned spline will be monotonic.
    Assumes that the unconstrained variables are at the front and the constrained at the end.

    (Here "constrained" coefficient means a coefficient with a monotonicity constraint, i.e.
    either the coefficient for x or the coefficient for max(x-knots_i, 0). "Unconstrained" means the
    other coefficients, i.e. for now the intercept and additional inputs coefficients.
    So there should be (num_knots + 1) "constrained" coefficients.)

    :param monotonicity: A member of the Monotonicity enum (must be either increasing, decreasing).
    :param num_constrained: Number of constrained variables (assumed to be at the end).
    :param num_unconstrained: Number of unconstrained variables (assumed to be at the beginning).
    :return: matrices G and h to be used as constraints in the quadratic program (G * coefficients <= h)
    """
    assert monotonicity is not Monotonicity.none

    if monotonicity is Monotonicity.increasing:
        sign = -1
    else:
        sign = 1
    unconstrained_matrix = np.zeros((num_constrained, num_unconstrained))
    constrained_matrix = sign * np.tril(np.ones((num_constrained, num_constrained)))
    G = np.hstack([unconstrained_matrix, constrained_matrix])

    if min_absolute_slope is None:
        # Unless otherwise specified, just enforce that slope is >= 0 or <= 0,
        # depending on monotonicity
        min_absolute_slope = 0.0
    if isinstance(min_absolute_slope, float):
        h = np.repeat(-min_absolute_slope, G.shape[0])
    else:
        h = -min_absolute_slope

    assert G.shape == (num_constrained, num_constrained + num_unconstrained)
    assert h.shape == (num_constrained,)

    return G, h


def _get_bounds_constraint_matrices(
    monotonicity,  # type: Monotonicity
    left_bound,  # type: Optional[Tuple[float, float]]
    right_bound,  # type: Optional[Tuple[float, float]]
    knots,  # type: np.ndarray
    intercept,  # type: bool
    distance_from_bound=0.0,  # type: float
):
    # type: (...) -> Tuple[np.ndarray, np.ndarray]
    """
    Precondition:
    - monotonicity must be increasing or decreasing, not None
    - either left_bound or right_bound should be non-None
    :param distance_from_bound: How far the minimum/maximum values of the spline output
        should be from the maximum/minimum bound. This is needed because the quadratic
        program solver only allows soft (>= or <=) constraints, not hard (> or <)
        constraints, and we usually don't want the output values to equal the maximum or
        minimum value.
    """
    assert monotonicity is not Monotonicity.none
    assert left_bound is not None or right_bound is not None

    # Build the rows of the constraint matrices (in the form lhs * coefs <= rhs)
    # as if monotonicity is increasing
    lhs = []
    rhs = []

    if left_bound is not None:
        min_x_transformed = _get_design_matrix(
            np.array([left_bound[0]]),
            None,
            knots,
            intercept,
        )[0]

        # add the constraint min_x_transformed * coefs >= left_bound[1]
        lhs.append(-min_x_transformed)

        if monotonicity is Monotonicity.increasing:
            y_val = left_bound[1] + distance_from_bound
        else:
            y_val = left_bound[1] - distance_from_bound
        rhs.append(-y_val)

    if right_bound is not None:
        max_x_transformed = _get_design_matrix(
            np.array([right_bound[0]]),
            None,
            knots,
            intercept,
        )[0]

        # add the constraint max_x_transformed * coefs <= right_bound[1]
        lhs.append(max_x_transformed)

        if monotonicity is Monotonicity.increasing:
            y_val = right_bound[1] - distance_from_bound
        else:
            y_val = right_bound[1] + distance_from_bound
        rhs.append(y_val)

    # If monotonicity is decreasing, flip the direction of the inequalities by
    # flipping the sign of the matrices.
    if monotonicity is Monotonicity.increasing:
        return np.array(lhs), np.array(rhs)
    else:
        return -np.array(lhs), -np.array(rhs)


def _knot_transformation(X, knots):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    Transforms univariate data with knot transformations.
    If knots == [1, 3] and we pass in [1, 5, 9],
    will return three columns, [1, 5, 9], [0, 4, 8], and
    [0, 2, 6]
    """
    x_array = np.array(X)
    if len(x_array.shape) > 1:
        raise ValueError("X must be univariate")

    # These arrays will have the same number of rows as x_array, and
    # same number of columns as knots.
    knot_subtracted = x_array[:, None] - knots[None, :]
    clipped = np.clip(knot_subtracted, 0.0, None)

    return np.column_stack([x_array, clipped])


def _get_design_matrix(
    inputs,  # type: np.ndarray
    additional_inputs,  # type: Optional[np.ndarray]
    knots,  # type: np.ndarray
    intercept,  # type: bool
):
    # type: (...) -> np.ndarray
    """
    Takes in univariate data and returns a design matrix.
    First gets knot-transformed matrix and concatenates an intercept
    if need be.

    :param inputs: 1-D array of length N
    :param additional_inputs: matrix of shape N x num_additional_inputs
        (if no additional inputs, should be either None or a matrix of shape N x 0)
    :param knots: 1-D array
    """
    assert len(inputs.shape) == 1

    if additional_inputs is not None:
        assert len(additional_inputs.shape) == 2
        assert additional_inputs.shape[0] == inputs.shape[0]
    else:
        additional_inputs = np.empty(shape=(inputs.shape[0], 0))

    if intercept:
        intercept_col = np.ones((inputs.shape[0], 1))
    else:
        intercept_col = np.empty((inputs.shape[0], 0))

    return np.hstack(
        [
            intercept_col,
            additional_inputs,
            _knot_transformation(inputs, knots),
        ]
    ).astype(float)


def _fit_knots(X, num_knots):
    # type: (np.ndarray, int) -> np.ndarray
    """
    Generates knots by finding `num_knots` quantiles of the given input distribution
    """
    if len(X.shape) != 1:
        raise ValueError("X must be a vector; has shape {}".format(X.shape))
    if X.shape[0] < num_knots:
        raise ValueError(
            "num_knots must be smaller than number of data points; num_knots is {} "
            "but there are {} data points".format(num_knots, X.shape[0])
        )

    percentiles = np.linspace(0.2, 99.8, num_knots)

    return np.percentile(X, percentiles)


def _get_regularization_penalty_matrix(
    num_knots,  # type: int
    num_additional_inputs,  # type: int
    intercept,  # type: bool
    penalize_intercept,  # type: bool
    penalize_first_coef,  # type: bool
    l2_penalty_knots,  # type: float
    l2_penalty_additional_inputs,  # type: float
):
    # type: (...) -> np.ndarray
    """
    Helper for implementing l2 regularization of each model coefficient.

    Returns a diagonal matrix, reg_matrix, that can be incorporated into the loss function is:
       coefs^T * reg_matrix * coefs
    """

    penalties = []

    # Intercept
    if intercept:
        penalties.append(l2_penalty_knots if penalize_intercept else 0)

    # Additional inputs
    penalties.extend([l2_penalty_additional_inputs] * num_additional_inputs)

    # First coefficient (for x)
    penalties.append(l2_penalty_knots if penalize_first_coef else 0)

    # num_knots knot coefficents (for max(x-knots_i, 0))
    penalties.extend([l2_penalty_knots] * num_knots)

    return np.diag(penalties)
