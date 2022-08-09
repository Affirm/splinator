"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.special import expit
from sklearn.utils.extmath import log_logistic
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from splinator.monotonic_spline import (
    Monotonicity,
    _fit_knots,
    _get_design_matrix,
    _get_monotonicity_constraint_matrices,
)
from enum import Enum


class MinimizationMethod(Enum):
    """These methods are supported by scipy.optimize.minimize"""

    slsqp = 'SLSQP'
    trust_constr = 'trust-constr'


class LossGradHess:
    """Much of this function is copied from sklearn logistic regression implementation -- presumably
    this is implemented this way for speed and to deal with float errors, but at the expense
    of readability / representing things as they are usually presented in literature.
    """

    def __init__(self, X, y, alpha, intercept):
        # type: (np.ndarray, np.ndarray, float, bool) -> None
        """
        In the generation of design matrix, if intercept option is True, the first column of design matrix is of 1's,
        which means that the first coefficient corresponds to the intercept term. This setup is a little different
        from sklearn.
        """
        self.y = y.copy()
        self.y[y == 0] = -1  # SKLearn encodes 1, -1
        self.X = X
        self.alpha = alpha
        self.intercept = intercept

    def loss(self, coefs):
        # type: (np.ndarray) -> np.ndarray
        yz = self.y * np.dot(self.X, coefs)
        # Take advantage of the fact that sigmoid(x) = 1 - sigmoid(-x)
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
        z0 = (z - 1) * self.y

        grad = np.dot(self.X.T, z0)
        if self.intercept:
            # do not add regularization term to intercept's gradient
            grad[1:] += self.alpha * coefs[1:]
        else:
            grad += self.alpha * coefs

        return grad

    def hess(self, coefs):
        # type: (np.ndarray) -> np.ndarray
        yz = self.y * np.dot(self.X, coefs)
        z = expit(yz)
        d = z * (1 - z)
        dX = d[:, np.newaxis] * self.X

        hessian = np.dot(self.X.T, dX)

        row_idx, col_idx = np.diag_indices_from(hessian)
        # add regularization term diagonally
        # if there is intercept, then we don't need to add regularization term to idx[0, 0]
        if self.intercept:
            hessian[row_idx[1:], col_idx[1:]] += self.alpha
        else:
            hessian[row_idx, col_idx] += self.alpha

        return hessian


class LinearSplineLogisticRegression(BaseEstimator):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    >>> from splinator.esimators import LinearSplineLogisticRegression
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = LinearSplineLogisticRegression()
    >>> estimator.fit(X, y)
    LinearSplineLogisticRegression()
    """

    def __init__(
            self,
            input_column,  # type: str
            coefficients=None,  # type: Optional[Union[List[float], np.ndarray]]
            knots=100,  # type: Union[int, List[float], np.ndarray]
            monotonicity=Monotonicity.none,  # type: Union[Monotonicity, str]
            intercept=True,  # type: bool
            additional_inputs_columns=None,  # type: Optional[List[Union[str, List[str], Tuple[str, ...]]]]
            method=MinimizationMethod.slsqp,  # type: Union[MinimizationMethod, str]
            minimizer_options={},  # type: Dict[str, Any]
            C=100,  # type: int
            two_stage_fitting_enabled=False,  # type: bool
            two_stage_fitting_initial_size=None,  # type: Optional[int]
            feature_names=None,  # type: Optional[Iterable[str]]
            random_seed=31,  # type: int
    ):
        # type: (...) -> None
        """
        For most uses, the defaults are reasonable and don't need to be changed, except
        for knots and monotonicity.

        :param input_column: The name of the column in the DataFrame passed into `fit`
            that will contain the (spline-transformed) input.
        :param coefficients: The coefficients of the underlying logistic regression
            model. We allow creating a logistic spline object with coefficients.
        :param knots: A list of knot x-values to use, or how many knots to include in
            the fitted function. (The function can only change slope at the knots.) If
            a number of knots is specified, chooses equally spaced quantiles in the
            x-values passed to fit.
        :param monotonicity: Whether to enforce that the function is monotonically
            increasing or decreasing.
        :param intercept: If True, allows the function value at x=0 to be nonzero.
        :param additional_inputs_columns: Names of the columns in the DataFrame passed into
            `fit` that will contain the additional (linear-transformed) inputs. The additional
            columns may be tuples, for example, if OneHotEncoder is used as feature transformer.
        :param method: The method named passed to scipy minimize. We have tested two methods:
            SLSQP and trust-contr. For scipy minimize, see
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        :param minimizer_options: Some scipy minimizer methods have their special options.
            For example: {'disp': True} will display a termination report. {'ftol': 1e-10} sets the precision goal
            for the value of f in the stopping criterion for SLSQP.
            Visit scipy minimize manual for options:
                (1) https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
                (2) https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html
        :param C: Inverse of regularization strength; must be a positive float. Like in
            support vector machines, smaller values specify stronger regularization.
            Due to the know error above (application of regularization to the intercept), we
             apply very small regularization (will fix later).
        :param two_stage_fitting_enabled: use two-stage fitting if True. When the dataset
            is too large, we choose a subsample for an initial fit to get estimates of coefficients.
            This set of coefficients will be used as initial guess for the second fit, which takes the
            entire dataset.
        :param two_stage_fitting_initial_size: subsample size of training data for first fitting.
            If two_stage_fitting_enabled is False, this should be None.
        :param feature_names: Name of the features.
        :param random_seed: random seed number, default is 31
        """

        if not input_column:
            raise ValueError('input_column cannot be none or empty')
        self.input_column = input_column

        if coefficients is not None:
            self.coefficients = np.array(coefficients)
            if self.coefficients.ndim != 1:
                raise ValueError("coefficients should be a 1-D array")
            if isinstance(knots, int):
                raise ValueError("knots should not be int if model is initialized with coefficients")
        else:
            self.coefficients = None

        if isinstance(knots, int):
            self.knots = knots
        elif isinstance(knots, list) or isinstance(knots, np.ndarray):
            self.knots = np.array(knots)
            assert self.knots.ndim == 1, "knots should be a 1-D array"  # type: ignore
        else:
            raise ValueError('"knots" argument must be an int or a list')
        self.fitted_knots = None

        self.monotonicity = Monotonicity(monotonicity)
        self.intercept = intercept
        self.method = method if isinstance(method, MinimizationMethod) else MinimizationMethod(method)
        self.feature_names = feature_names
        self.minimizer_options = minimizer_options

        if not C > 0:
            raise ValueError("C must be a positive non-zero value")
        self.C = C

        if two_stage_fitting_enabled:
            if two_stage_fitting_initial_size is None:
                raise ValueError("two_stage_fitting_initial_size is None but wo_stage_fitting is enabled")
            elif not two_stage_fitting_initial_size > 0:
                raise ValueError(
                    "two_stage_fitting_initial_size must be a positive non-zero value \
                                  when two_stage_fitting is enabled"
                )
        else:
            if two_stage_fitting_initial_size is not None:
                raise ValueError("two_stage_fitting is not enabled but two_stage_fitting_initial_size is not None")

        self.two_stage_fitting_enabled = two_stage_fitting_enabled
        self.two_stage_fitting_initial_size = two_stage_fitting_initial_size

        self.additional_inputs_columns = []

        if additional_inputs_columns is not None:
            # In model serialization, tuple columns were converted to list, let's convert them back.
            # Tuple columns are usually a result of transformers like OneHotEncoder.

            for column in additional_inputs_columns:
                if isinstance(column, tuple) or isinstance(column, list):
                    if len(column) != 2:
                        raise ValueError("only pair lists/tuples are supported for column names.")

            additional_inputs_columns = [tuple(x) if isinstance(x, list) else x for x in additional_inputs_columns]
            if len(set(additional_inputs_columns)) != len(additional_inputs_columns):
                raise ValueError("column names in additional_inputs_columns must be unique")
            if input_column in additional_inputs_columns:
                raise ValueError(
                    "\"{}\" cannot be used both as the main input column and one "
                    "of the additional input columns".format(input_column)
                )
            self.additional_inputs_columns = additional_inputs_columns

        if isinstance(self.knots, np.ndarray) and self.coefficients is not None:
            if self.knots.ndim != 1:
                raise ValueError('knots must be a 1d array')
            if self.coefficients.ndim != 1:
                raise ValueError('coefficients must be a 1d array')
            intercept_dim = 1 if self.intercept else 0
            if not self.coefficients.shape[0] == 1 + self.knots.shape[0] + intercept_dim + len(
                    self.additional_inputs_columns
            ):
                raise ValueError(
                    "dimension of coefficients should equal dim(knots) + int(intercept) + " "num_additional_inputs + 1"
                )
        self.random_seed = random_seed

    def _fit(self, X, y, initial_guess=None, final_run=False):
        # type: (pd.DataFrame, pd.Series, Optional[np.ndarray], bool) -> None
        if isinstance(self.knots, int):
            # knots being type int means that we want to re-fit this number of
            # knots every time fit is called.
            knots = _fit_knots(X[self.input_column].values, self.knots)
            if final_run:
                # If this is final run, we should save the fitted knots.
                # Otherwise, the computed knots will only be used for this fitting. In this way, knots property can
                # still be kept int, and we would be able to re-fit next time.
                self.knots = knots
        else:
            knots = self.knots

        constraint = []  # type: Union[Dict[str, Any], List, LinearConstraint]
        if self.monotonicity != Monotonicity.none:
            # This function returns G and h such that G * beta <= 0 is the constraint we want:
            # See https://docs.google.com/document/d/1xDPsnfKhxkUwNfKGyAzsvV3lEq8-3YYn0O7Uw7fgqsM/edit
            G, h = _get_monotonicity_constraint_matrices(
                self.monotonicity,
                num_constrained=knots.shape[0] + 1,
                num_unconstrained=int(self.intercept) + len(self.additional_inputs_columns),
                min_absolute_slope=None,
            )

            if self.method == MinimizationMethod.trust_constr:
                # We give the constraint that G * beta >= -infinity and G * beta <= 0.
                # (The -infinity is needed to specify that it's a one-sided constraint)
                constraint = LinearConstraint(
                    G,
                    -np.inf * np.ones(G.shape[0]),
                    np.zeros(G.shape[0]),
                )
            elif self.method == MinimizationMethod.slsqp:
                # the SLSQP solver expects a constraint of the form M * x >= 0,
                # so we pass in M = -G to enforce that G * beta <= 0.
                constraint = {'type': 'ineq', 'fun': lambda x: np.dot(-G, x)}

            else:
                raise ValueError("Only trust-constr and SLSQP are currently supported.")

        design_X = _get_design_matrix(
            inputs=X[self.input_column].values,
            additional_inputs=X[self.additional_inputs_columns].values,
            knots=knots,
            intercept=self.intercept,
        )

        if initial_guess is None:
            x0 = np.zeros(design_X.shape[1])
        else:
            x0 = initial_guess

        lgh = LossGradHess(design_X, y.values, 1 / self.C, self.intercept)

        result = minimize(
            fun=lgh.loss,
            x0=x0,
            jac=lgh.grad,
            hess=lgh.hess if self.method == MinimizationMethod.trust_constr else None,
            method=self.method.value,
            constraints=constraint,
            options=self.minimizer_options,
        )
        if not result.success:
            warnings.warn(f"The minimization failed with message: '{result.message}'")
        self.coefficients = result.x

    def fit(self, X, y, sample_weight=None):
        # type: (pd.DataFrame, Union[np.ndarray, pd.Series], Optional[np.ndarray]) -> None
        """
        When the dataset is too large, we choose to use a random subset of the data to do an initial fit;
        Then we take the coefficients as initial guess to fit again using the entire dataset. This will speed
        up training and avoid under-fitting.
        We use two_stage_fitting_size as the sampling size.
        """
        np.random.seed(self.random_seed)

        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        if self.two_stage_fitting_enabled:
            if self.two_stage_fitting_initial_size > X.shape[0]:
                raise ValueError("two_stage_fitting_initial_size should be smaller than data size")
            index = np.random.choice(np.arange(len(X)), self.two_stage_fitting_initial_size, replace=False)
            X_sub, y_sub = X.iloc[index], y.iloc[index]
            # initial fitting without guess
            self._fit(X_sub, y_sub, initial_guess=None, final_run=False)
            # final fitting with coefs from initial run as guess
            self._fit(X, y, initial_guess=self.coefficients, final_run=True)
        else:
            # one fitting that is final
            self._fit(X, y, initial_guess=None, final_run=True)

    def predict(self, X, **kwargs):
        # type: (pd.DataFrame, **object) -> np.ndarray
        if not self.is_fitted:
            raise ValueError("Cannot call predict on a model that has not been fitted")

        design_X = _get_design_matrix(
            inputs=X[self.input_column].values,
            additional_inputs=X[self.additional_inputs_columns].values,
            knots=self.knots,
            intercept=self.intercept,
        )

        return expit(np.dot(design_X, self.coefficients))

    @property
    def is_fitted(self):
        # type: () -> bool
        # NOTE: the model can be initialized with coefficients so this might not be a reliable test of fit
        return self.coefficients is not None

    def transform(self, X, **transformer_kwargs):
        # type: (pd.DataFrame, **Any) -> pd.DataFrame
        return X

    def state_to_dict(self):
        # type: () -> dict
        if not self.is_fitted:
            raise ValueError("cannot serialize unfit model")

        return {
            'input_column': self.input_column,
            'coefficients': self.coefficients,
            'knots': self.knots,
            'monotonicity': self.monotonicity.value,
            'intercept': self.intercept,
            'additional_inputs_columns': self.additional_inputs_columns,
            'method': self.method.value,
            'minimizer_options': self.minimizer_options,
            'C': self.C,
            'two_stage_fitting_enabled': self.two_stage_fitting_enabled,
            'two_stage_fitting_initial_size': self.two_stage_fitting_initial_size,
            'random_seed': self.random_seed,
        }