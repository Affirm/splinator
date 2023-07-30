from __future__ import print_function
from __future__ import absolute_import
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from typing import Optional, Tuple, Union


class Monotonicity(Enum):
    none = 'none'
    increasing = 'increasing'
    decreasing = 'decreasing'


def _get_monotonicity_constraint_matrices(
    monotonicity,  # type: str
    num_constrained,  # type: int
    num_unconstrained,  # type: int
    min_absolute_slope,  # type: Union[float, np.ndarray, None]
):
    # type: (...) -> Tuple[np.ndarray, np.ndarray]
    """
    Gets constraints such that the returned spline will be monotonic.
    Assumes that the unconstrained variables are at the front and the constrained at the end (IMPORTANT!).

    (Here "constrained" coefficient means a coefficient with a monotonicity constraint, i.e.
    either the coefficient for x or the coefficient for max(x-knots_i, 0). "Unconstrained" means the
    other coefficients, i.e. for now the intercept and additional inputs coefficients.
    So there should be (num_knots + 1) "constrained" coefficients.)

    :param monotonicity: A member of the Monotonicity enum (must be either increasing, decreasing).
    :param num_constrained: Number of constrained variables (assumed to be at the end).
    :param num_unconstrained: Number of unconstrained variables (assumed to be at the beginning).
    :return: matrices G and h to be used as constraints in the quadratic program (G * coefficients <= h)
    """
    # no need to run this function if there is no monotonicity
    assert monotonicity is not Monotonicity.none.value

    if monotonicity is Monotonicity.increasing.value:
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
    if isinstance(min_absolute_slope, (int, float)):
        h = np.repeat(-min_absolute_slope, G.shape[0])
    else:
        h = -min_absolute_slope

    assert G.shape == (num_constrained, num_constrained + num_unconstrained)
    assert h.shape == (num_constrained,)

    return G, h


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

    percentiles = np.linspace(0, 100, num_knots, endpoint=False)[1:]

    return np.percentile(X, percentiles)
