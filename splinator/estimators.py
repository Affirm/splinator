import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.special import expit
from sklearn.utils.extmath import log_logistic
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_array, check_random_state, _check_sample_weight, check_consistent_length
from sklearn.exceptions import DataConversionWarning, NotFittedError
from warnings import warn
from splinator.monotonic_spline import (
    Monotonicity,
    _fit_knots,
    _get_design_matrix,
    _get_monotonicity_constraint_matrices,
)
from enum import Enum
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Iterable, Tuple


class MinimizationMethod(Enum):
    """These two constrained-optimization methods are supported by scipy.optimize.minimize"""
    # they are case-sensitive
    slsqp = 'SLSQP'
    trust_constr = 'trust-constr'


class LossGradHess:
    def __init__(self, X, y, alpha, intercept):
        # type: (np.ndarray, np.ndarray, float, bool) -> None
        """
        In the generation of design matrix, if intercept option is True, the first column of design matrix is of 1's,
        which means that the first coefficient corresponds to the intercept term. This setup is a little different
        from sklearn's logistic regression.
        """
        self.y = y.copy()
        # encodes y as 1, -1 make the log-loss form simpler to write
        self.y[y == 0] = -1
        self.X = X
        self.alpha = alpha
        self.intercept = intercept

    def loss(self, coefs):
        # type: (np.ndarray) -> np.ndarray
        yz = self.y * np.dot(self.X, coefs)
        # P(label= 1 or -1 |X) = 1 / (1+exp(-yz))
        # Log Likelihood = Sum over log ( 1/(1 + exp(-yz)) )
        loss_val = -np.sum(log_logistic(yz))
        if self.intercept:
            loss_val += 0.5 * self.alpha * np.dot(coefs[1:], coefs[1:])
        else:
            loss_val += 0.5 * self.alpha * np.dot(coefs, coefs)
        return loss_val

    def grad(self, coefs):
        # type: (np.ndarray) -> np.ndarray
        yz = self.y * np.dot(self.X, coefs)

        z = expit(yz)

        # if y = 1, we want z to be close to 1; if y = -1, we want z to be close to 0.
        z0 = (z - 1) * self.y

        grad = np.dot(self.X.T, z0)

        if self.intercept:
            grad[1:] += self.alpha * coefs[1:]
        else:
            grad += self.alpha * coefs

        return grad


class LinearSplineLogisticRegression(RegressorMixin, TransformerMixin, BaseEstimator):
    """ Piecewise Logistic Regression with Linear Splines

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    >>> from splinator.estimators import LinearSplineLogisticRegression
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = LinearSplineLogisticRegression()
    >>> estimator.fit(X, y)
    """

    def __init__(
            self,
            input_score_column_index: int = 0,
            n_knots: Optional[int] = 100,
            knots: Optional[Union[List[float], np.ndarray]] = None,
            monotonicity: str = Monotonicity.none.value,
            intercept: bool = True,
            method: str = MinimizationMethod.slsqp.value,
            minimizer_options: Dict[str, Any] = None,
            C: int = 100,
            two_stage_fitting_initial_size: int = None,
            random_state: int = 31,
            verbose=False,
    ):
        # type: (...) -> None
        """

        Parameters
        ----------
        input_score_column_index : int, default=0
        knots : array-like of float, default=None
        n_knots : int: default=100
            Only one of knots and n_knots are needed. If both are non-null, it will throw an exception in fitting
        monotonicity : str, default='none'
            Whether to enforce that the function is monotonically increasing or decreasing.
        intercept : bool, default=True
            If True, allows the function value at x=0 to be nonzero.
        method : str, default='slsqp'
             The method named passed to scipy minimize. We have tested two methods: SLSQP and trust-contr.
             For scipy minimize, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        minimizer_options : dict, default={}
            Some scipy minimizer methods have their special options.
            For example: {'disp': True} will display a termination report. {'ftol': 1e-10} sets the precision goal
            for the value of f in the stopping criterion for SLSQP.
            Visit scipy minimize manual for options:
                (1) https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
                (2) https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html
        C : int, default=100
            Inverse of regularization strength; must be a positive float. Like in support vector machines,
            smaller values specify stronger regularization.
        two_stage_fitting_initial_size : int, default=None
            subsample size of training data for first fitting.
            If two_stage_fitting is not used, this should be None.
        random_state : int, default=31
            random seed number, default is 31
        """
        self.input_score_column_index = input_score_column_index
        self.n_knots = n_knots
        self.knots = knots
        self.monotonicity = monotonicity
        self.intercept = intercept
        self.method = method
        self.minimizer_options = minimizer_options
        self.C = C
        self.two_stage_fitting_initial_size = two_stage_fitting_initial_size
        self.random_state = random_state
        self.verbose = verbose

    def _fit(self, X, y, initial_guess=None):
        # type: (pd.DataFrame, pd.Series, Optional[np.ndarray], bool) -> None
        constraint = []  # type: Union[Dict[str, Any], List, LinearConstraint]
        if self.monotonicity != Monotonicity.none.value:
            # This function returns G and h such that G * beta <= 0 is the constraint we want:
            # See https://docs.google.com/document/d/1xDPsnfKhxkUwNfKGyAzsvV3lEq8-3YYn0O7Uw7fgqsM/edit
            G, h = _get_monotonicity_constraint_matrices(
                self.monotonicity,
                num_constrained=self.knots_.shape[0] + 1,
                num_unconstrained=int(self.intercept) + self.n_features_in_ - 1,
                min_absolute_slope=None,
            )
            if self.method == MinimizationMethod.trust_constr.value:
                # We give the constraint that G * beta >= -infinity and G * beta <= 0.
                # (The -infinity is needed to specify that it's a one-sided constraint)
                constraint = LinearConstraint(
                    G,
                    -np.inf * np.ones(G.shape[0]),
                    np.zeros(G.shape[0]),
                )
            elif self.method == MinimizationMethod.slsqp.value:
                # the SLSQP solver expects a constraint of the form M * x >= 0,
                # so we pass in M = -G to enforce that G * beta <= 0.
                constraint = {'type': 'ineq', 'fun': lambda x: np.dot(-G, x)}

            else:
                raise ValueError("Only trust-constr and SLSQP are currently supported.")

        design_X = _get_design_matrix(
            inputs=self.get_input_scores(X),
            additional_inputs=self.get_additional_columns(X),
            knots=self.knots_,
            intercept=self.intercept,
        )
        if initial_guess is None:
            x0 = np.zeros(design_X.shape[1])
        else:
            x0 = initial_guess

        lgh = LossGradHess(design_X, y, 1 / self.C, self.intercept)

        result = minimize(
            fun=lgh.loss,
            x0=x0,
            jac=lgh.grad,
            method=self.method,
            constraints=constraint,
            options=self.minimizer_options,
        )
        self.result_ = result
        optimization_message = "The minimization failed with message: '{message}'".format(message=result.message)

        if not result.success:
            warn(optimization_message)

        self.coefficients_ = result.x

    def get_input_scores(self, X):
        if X.ndim > 1:
            input_scores = X[:, self.input_score_column_index]
        else:
            input_scores = X
        return input_scores

    def get_additional_columns(self, X):
        additional_columns = np.delete(X, self.input_score_column_index, axis=1)
        return additional_columns

    def fit(self, X, y):
        # type: (pd.DataFrame, Union[np.ndarray, pd.Series], Optional[np.ndarray]) -> None
        """
        When the dataset is too large, we choose to use a random subset of the data to do an initial fit;
        Then we take the coefficients as initial guess to fit again using the entire dataset. This will speed
        up training and avoid under-fitting.
        We use two_stage_fitting_size as the sampling size.
        """
        self.random_state_ = check_random_state(self.random_state)
        check_params = dict(accept_sparse=False, ensure_2d=False)

        X = check_array(X, dtype=[np.float64, np.float32], **check_params)
        self.n_features_in_ = 1 if X.ndim == 1 else X.shape[1]

        if y is None:
            raise ValueError("y should be a 1d array")
        y = check_array(y, dtype=X.dtype, **check_params)
        if y.ndim > 1:
            warn(
                "A column-vector y was passed when a 1d array was expected.",
                DataConversionWarning,
            )
            y = y[:, 0]

        check_consistent_length(X, y)

        if self.n_knots and self.knots is None:
            # only n_knots given so we create knots
            self.n_knots_ = min([self.n_knots, X.shape[0]])
            self.knots_ = _fit_knots(self.get_input_scores(X), self.n_knots_)
        elif self.knots is not None and self.n_knots is None:
            # knots are given so we just take them
            self.knots_ = np.array(self.knots)
        else:
            raise ValueError("knots and n_knots cannot be both null or non-null")

        if self.method not in ['SLSQP', 'trust-constr']:
            raise ValueError("optimization method can only be either 'SLSQP' or 'trust-contr'")

        if self.two_stage_fitting_initial_size is None:
            self._fit(X, y, initial_guess=None)
        else:
            if self.two_stage_fitting_initial_size > X.shape[0]:
                raise ValueError("two_stage_fitting_initial_size should be smaller than data size")

            # initial fitting without guess
            index = self.random_state_.choice(np.arange(len(X)), self.two_stage_fitting_initial_size, replace=False)
            if isinstance(X, pd.DataFrame):
                X_sub, y_sub = X.iloc[index], y[index]
            else:
                X_sub, y_sub = X[index, :], y[index]
            self._fit(X_sub, y_sub, initial_guess=None)

            # final fitting with coefs from initial run as guess
            self._fit(X, y, initial_guess=self.coefficients_)

        return self

    def transform(self, X):
        if not self.is_fitted:
            raise NotFittedError(
                "predict or transform is not available if the estimator was not fitted"
            )
        X = self._validate_data(X, reset=False)

        design_X = _get_design_matrix(
            inputs=self.get_input_scores(X),
            additional_inputs=self.get_additional_columns(X),
            knots=self.knots_,
            intercept=self.intercept,
        )
        return expit(np.dot(design_X, self.coefficients_))

    def predict(self, X):
        return self.transform(X)

    @property
    def is_fitted(self) -> bool:
        """
        Check if the model is fitted by checking if it has 'coefficients_' attribute

        Returns
        -------
        is_fitted : bool
        """
        return hasattr(self, 'coefficients_')

    def _more_tags(self) -> Dict[str, bool]:
        """
        Override default sklearn tags (sklearn.utils._DEFAULT_TAGS)

        Returns
        -------
        tags : dict
        """
        return {"poor_score": True, "binary_only": True, "requires_y": True}
