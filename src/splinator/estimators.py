import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.special import expit, log_expit
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_random_state, validate_data
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
    """Optimization methods supported by scipy.optimize.minimize"""
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
        loss_val = -np.sum(log_expit(yz))
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

    def hess(self, coefs):
        # type: (np.ndarray) -> np.ndarray
        """
        Compute the Hessian of the logistic loss.
        
        The Hessian of logistic regression is: H = X.T @ diag(w) @ X + alpha * I
        where w = p * (1 - p) and p = sigmoid(X @ coefs).
        
        This is used by trust-constr for faster convergence (Newton-like steps).
        """
        z = np.dot(self.X, coefs)
        p = expit(z)
        # Weights for the Hessian: p * (1 - p)
        # This is always positive, making the Hessian positive semi-definite
        weights = p * (1 - p)
        
        # H = X.T @ diag(weights) @ X
        # Efficient computation: (X.T * weights) @ X
        H = np.dot(self.X.T * weights, self.X)
        
        # Add regularization term
        if self.intercept:
            # Don't regularize the intercept
            reg_indices = np.arange(1, H.shape[0])
            H[reg_indices, reg_indices] += self.alpha
        else:
            H += self.alpha * np.eye(H.shape[0])
        
        return H


class LinearSplineLogisticRegression(RegressorMixin, TransformerMixin, BaseEstimator):
    """Piecewise Logistic Regression with Linear Splines.

    A scikit-learn compatible estimator that fits a piecewise linear function
    in log-odds space for probability calibration. Supports monotonicity
    constraints and L2 regularization.

    Parameters
    ----------
    input_score_column_index : int, default=0
        Index of the column in X to use as the primary input score for spline fitting.
    n_knots : int or None, default=100
        Number of knots to automatically place at quantiles of the input distribution.
        Only one of `knots` and `n_knots` should be provided.
    knots : array-like of float or None, default=None
        Explicit knot positions. Only one of `knots` and `n_knots` should be provided.
    monotonicity : str, default='none'
        Monotonicity constraint: 'none', 'increasing', or 'decreasing'.
    intercept : bool, default=True
        Whether to include an intercept term in the model.
    method : str, default='SLSQP'
        Optimization method for scipy.optimize.minimize. Supports 'SLSQP' or 'trust-constr'.
        SLSQP is recommended for problems with monotonicity constraints (fastest).
        trust-constr with Hessian is faster for unconstrained problems.
    minimizer_options : dict or None, default=None
        Additional options passed to the scipy minimizer.
    C : float, default=100
        Inverse of regularization strength (larger values = weaker regularization).
    two_stage_fitting_initial_size : int or None, default=None
        If provided, performs initial fit on a subsample of this size for faster convergence.
        Deprecated: prefer using `progressive_fitting_fractions` for better performance.
    progressive_fitting_fractions : tuple or None, default=None
        Tuple of fractions for progressive fitting (e.g., (0.1, 0.3, 1.0)).
        Each stage uses stratified sampling to maintain score distribution coverage.
        If provided, overrides `two_stage_fitting_initial_size`.
    stratified_sampling : bool, default=True
        If True, uses quantile-based stratified sampling to ensure subsamples cover
        the full score range. Only applies when using progressive/two-stage fitting.
    early_stopping_tol : float or None, default=1e-4
        If provided, stops progressive fitting early when coefficient change
        (relative L2 norm) falls below this threshold between stages.
    use_hessian : bool or 'auto', default='auto'
        Controls Hessian usage with trust-constr method.
        - 'auto': Enable Hessian when monotonicity='none' (3-4x faster)
        - True: Always use Hessian (can be slow with constraints)
        - False: Never use Hessian
    random_state : int, default=31
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to print verbose output during fitting.

    Attributes
    ----------
    coefficients_ : ndarray
        Fitted coefficients after training.
    knots_ : ndarray
        The knot positions used in the model.
    n_features_in_ : int
        Number of features seen during fit.
    fitting_history_ : list
        History of fitting stages when using progressive fitting.

    Examples
    --------
    >>> from splinator.estimators import LinearSplineLogisticRegression
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = LinearSplineLogisticRegression()
    >>> estimator.fit(X, y)
    
    For monotonic calibration (recommended):
    >>> estimator = LinearSplineLogisticRegression(
    ...     monotonicity='increasing',  # SLSQP is fastest for constraints
    ...     n_knots=50,
    ... )
    
    For unconstrained fitting on large data (uses Hessian automatically):
    >>> estimator = LinearSplineLogisticRegression(
    ...     monotonicity='none',
    ...     method='trust-constr',  # Auto-enables Hessian for 3-4x speedup
    ... )
    """

    def __init__(
            self,
            input_score_column_index: int = 0,
            n_knots: Optional[int] = 100,
            knots: Optional[Union[List[float], np.ndarray]] = None,
            monotonicity: str = Monotonicity.none.value,
            intercept: bool = True,
            method: str = MinimizationMethod.slsqp.value,
            minimizer_options: Optional[Dict[str, Any]] = None,
            C: int = 100,
            two_stage_fitting_initial_size: Optional[int] = None,
            progressive_fitting_fractions: Optional[Tuple[float, ...]] = None,
            stratified_sampling: bool = True,
            early_stopping_tol: Optional[float] = 1e-4,
            use_hessian: Union[bool, str] = 'auto',
            random_state: int = 31,
            verbose: bool = False,
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
            Deprecated: prefer using `progressive_fitting_fractions`.
        progressive_fitting_fractions : tuple, default=None
            Fractions of data for progressive fitting stages (e.g., (0.1, 0.3, 1.0)).
            Uses stratified sampling and warm-starts each stage with previous coefficients.
        stratified_sampling : bool, default=True
            Use quantile-based stratified sampling for progressive fitting.
        early_stopping_tol : float, default=1e-4
            Stop early if coefficient change between stages is below this threshold.
        use_hessian : bool or 'auto', default='auto'
            Use analytical Hessian with trust-constr. 'auto' enables for unconstrained problems.
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
        self.progressive_fitting_fractions = progressive_fitting_fractions
        self.stratified_sampling = stratified_sampling
        self.early_stopping_tol = early_stopping_tol
        self.use_hessian = use_hessian
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

        # Determine whether to use Hessian
        # - 'auto': use Hessian only for trust-constr with no monotonicity constraints (3-4x faster)
        # - True: always use Hessian with trust-constr
        # - False: never use Hessian
        if self.use_hessian == 'auto':
            use_hess = (
                self.method == MinimizationMethod.trust_constr.value
                and self.monotonicity == Monotonicity.none.value
            )
        else:
            use_hess = (
                self.use_hessian
                and self.method == MinimizationMethod.trust_constr.value
            )
        hess = lgh.hess if use_hess else None

        result = minimize(
            fun=lgh.loss,
            x0=x0,
            jac=lgh.grad,
            hess=hess,
            method=self.method,
            constraints=constraint,
            options=self.minimizer_options or {},
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

    def _stratified_subsample(self, X, y, n_samples, n_strata=10):
        # type: (np.ndarray, np.ndarray, int, int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        Create a stratified subsample based on quantiles of the input scores.
        
        This ensures the subsample covers the full range of scores, which is
        important for fitting splines accurately.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        n_samples : int
            Target number of samples in the subsample
        n_strata : int
            Number of strata (quantile bins) to use
            
        Returns
        -------
        X_sub : array-like
        y_sub : array-like  
        indices : array-like
            Indices of selected samples
        """
        input_scores = self.get_input_scores(X)
        n_total = len(X)
        
        # Calculate quantile boundaries for stratification
        percentiles = np.linspace(0, 100, n_strata + 1)
        boundaries = np.percentile(input_scores, percentiles)
        
        # Assign each sample to a stratum
        strata = np.digitize(input_scores, boundaries[1:-1])
        
        # Sample proportionally from each stratum
        selected_indices = []
        samples_per_stratum = max(1, n_samples // n_strata)
        
        for s in range(n_strata):
            stratum_indices = np.where(strata == s)[0]
            if len(stratum_indices) == 0:
                continue
            
            # Sample from this stratum
            n_to_sample = min(samples_per_stratum, len(stratum_indices))
            sampled = self.random_state_.choice(
                stratum_indices, n_to_sample, replace=False
            )
            selected_indices.extend(sampled)
        
        # If we need more samples to reach target, sample randomly from remainder
        selected_indices = np.array(selected_indices)
        if len(selected_indices) < n_samples:
            remaining = np.setdiff1d(np.arange(n_total), selected_indices)
            n_extra = min(n_samples - len(selected_indices), len(remaining))
            if n_extra > 0:
                extra = self.random_state_.choice(remaining, n_extra, replace=False)
                selected_indices = np.concatenate([selected_indices, extra])
        
        # Shuffle to avoid any ordering effects
        self.random_state_.shuffle(selected_indices)
        selected_indices = selected_indices[:n_samples]
        
        if isinstance(X, pd.DataFrame):
            X_sub = X.iloc[selected_indices]
        else:
            X_sub = X[selected_indices, :]
        y_sub = y[selected_indices]
        
        return X_sub, y_sub, selected_indices

    def _random_subsample(self, X, y, n_samples):
        # type: (np.ndarray, np.ndarray, int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
        """Simple random subsampling (original behavior)."""
        indices = self.random_state_.choice(
            np.arange(len(X)), n_samples, replace=False
        )
        if isinstance(X, pd.DataFrame):
            X_sub, y_sub = X.iloc[indices], y[indices]
        else:
            X_sub, y_sub = X[indices, :], y[indices]
        return X_sub, y_sub, indices

    def _check_convergence(self, coefs_old, coefs_new):
        # type: (np.ndarray, np.ndarray) -> bool
        """Check if coefficients have converged based on relative change."""
        if coefs_old is None or self.early_stopping_tol is None:
            return False
        
        # Relative L2 norm of change
        norm_old = np.linalg.norm(coefs_old)
        if norm_old < 1e-10:
            # If old coefficients are near zero, use absolute change
            change = np.linalg.norm(coefs_new - coefs_old)
        else:
            change = np.linalg.norm(coefs_new - coefs_old) / norm_old
        
        return change < self.early_stopping_tol

    def fit(self, X, y):
        # type: (pd.DataFrame, Union[np.ndarray, pd.Series], Optional[np.ndarray]) -> None
        """
        Fit the linear spline logistic regression model.
        
        Supports three fitting modes:
        1. Direct fitting (default): Fit on full data
        2. Two-stage fitting (legacy): Uses `two_stage_fitting_initial_size`
        3. Progressive fitting (recommended): Uses `progressive_fitting_fractions`
        
        Progressive fitting with stratified sampling is recommended for large datasets
        as it provides faster convergence while maintaining calibration quality.
        """
        self.random_state_ = check_random_state(self.random_state)
        self.fitting_history_ = []

        # Validate X and y, this sets n_features_in_ automatically
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            ensure_2d=True,
            dtype=[np.float64, np.float32],
            y_numeric=True,
            multi_output=False,
        )

        if y.ndim > 1:
            warn(
                "A column-vector y was passed when a 1d array was expected.",
                DataConversionWarning,
            )
            y = y[:, 0]

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
            raise ValueError("optimization method can only be either 'SLSQP' or 'trust-constr'")

        n_samples = X.shape[0]

        # Determine fitting mode
        if self.progressive_fitting_fractions is not None:
            # Mode 3: Progressive fitting (recommended)
            self._progressive_fit(X, y)
        elif self.two_stage_fitting_initial_size is not None:
            # Mode 2: Legacy two-stage fitting
            self._two_stage_fit(X, y)
        else:
            # Mode 1: Direct fitting on full data
            self._fit(X, y, initial_guess=None)
            self.fitting_history_.append({
                'stage': 0,
                'n_samples': n_samples,
                'fraction': 1.0,
                'iterations': getattr(self.result_, 'nit', None),
                'converged': self.result_.success,
            })

        return self

    def _two_stage_fit(self, X, y):
        # type: (np.ndarray, np.ndarray) -> None
        """Legacy two-stage fitting for backward compatibility."""
        n_samples = X.shape[0]
        
        if self.two_stage_fitting_initial_size > n_samples:
            raise ValueError("two_stage_fitting_initial_size should be smaller than data size")
        
        # Warn if subsample is too small relative to knots
        samples_per_knot = self.two_stage_fitting_initial_size / (self.knots_.shape[0] + 1)
        if samples_per_knot < 50:
            warn(
                f"Subsample size ({self.two_stage_fitting_initial_size}) may be too small "
                f"for {self.knots_.shape[0]} knots ({samples_per_knot:.0f} samples/knot). "
                f"Consider increasing subsample size or reducing knots for better warm-start."
            )

        # Stage 1: Fit on subsample
        if self.stratified_sampling:
            X_sub, y_sub, _ = self._stratified_subsample(
                X, y, self.two_stage_fitting_initial_size
            )
        else:
            X_sub, y_sub, _ = self._random_subsample(
                X, y, self.two_stage_fitting_initial_size
            )
        
        self._fit(X_sub, y_sub, initial_guess=None)
        self.fitting_history_.append({
            'stage': 0,
            'n_samples': self.two_stage_fitting_initial_size,
            'fraction': self.two_stage_fitting_initial_size / n_samples,
            'iterations': getattr(self.result_, 'nit', None),
            'converged': self.result_.success,
        })

        if self.verbose:
            print(f"Stage 1: {self.two_stage_fitting_initial_size} samples, "
                  f"{self.result_.nit} iterations")

        # Stage 2: Fit on full data with warm start
        coefs_stage1 = self.coefficients_.copy()
        self._fit(X, y, initial_guess=coefs_stage1)
        self.fitting_history_.append({
            'stage': 1,
            'n_samples': n_samples,
            'fraction': 1.0,
            'iterations': getattr(self.result_, 'nit', None),
            'converged': self.result_.success,
        })

        if self.verbose:
            print(f"Stage 2: {n_samples} samples, {self.result_.nit} iterations")

    def _progressive_fit(self, X, y):
        # type: (np.ndarray, np.ndarray) -> None
        """
        Progressive fitting with gradual sample increase.
        
        Uses stratified sampling to ensure each stage covers the full score range.
        Warm-starts each stage with coefficients from the previous stage.
        Supports early stopping if coefficients converge.
        """
        n_samples = X.shape[0]
        fractions = self.progressive_fitting_fractions
        
        # Validate fractions
        if not all(0 < f <= 1.0 for f in fractions):
            raise ValueError("All fractions must be in (0, 1]")
        if fractions[-1] != 1.0:
            # Ensure we always end with full data
            fractions = tuple(fractions) + (1.0,)
        
        prev_coefs = None
        
        for stage, frac in enumerate(fractions):
            n_stage_samples = int(n_samples * frac)
            n_stage_samples = max(n_stage_samples, self.knots_.shape[0] + 10)  # Ensure enough samples
            n_stage_samples = min(n_stage_samples, n_samples)
            
            if frac >= 1.0:
                # Use full data for final stage
                X_stage, y_stage = X, y
            else:
                # Warn if subsample too small for knots
                samples_per_knot = n_stage_samples / (self.knots_.shape[0] + 1)
                if samples_per_knot < 50 and self.verbose:
                    print(f"  Warning: Stage {stage} has {samples_per_knot:.0f} samples/knot (recommend >= 50)")
                
                # Subsample for intermediate stages
                if self.stratified_sampling:
                    X_stage, y_stage, _ = self._stratified_subsample(
                        X, y, n_stage_samples
                    )
                else:
                    X_stage, y_stage, _ = self._random_subsample(
                        X, y, n_stage_samples
                    )
            
            self._fit(X_stage, y_stage, initial_guess=prev_coefs)
            
            self.fitting_history_.append({
                'stage': stage,
                'n_samples': n_stage_samples,
                'fraction': frac,
                'iterations': getattr(self.result_, 'nit', None),
                'converged': self.result_.success,
            })
            
            if self.verbose:
                print(f"Stage {stage + 1}/{len(fractions)}: "
                      f"{n_stage_samples} samples ({frac:.1%}), "
                      f"{self.result_.nit} iterations")
            
            # Check for early stopping (but always complete final stage)
            if frac < 1.0 and self._check_convergence(prev_coefs, self.coefficients_):
                if self.verbose:
                    print(f"Early stopping: coefficients converged at stage {stage + 1}")
                # Still fit on full data for final refinement, but with fewer iterations expected
                prev_coefs = self.coefficients_.copy()
                self._fit(X, y, initial_guess=prev_coefs)
                self.fitting_history_.append({
                    'stage': stage + 1,
                    'n_samples': n_samples,
                    'fraction': 1.0,
                    'iterations': getattr(self.result_, 'nit', None),
                    'converged': self.result_.success,
                    'early_stopped': True,
                })
                if self.verbose:
                    print(f"Final stage: {n_samples} samples, {self.result_.nit} iterations")
                break
            
            prev_coefs = self.coefficients_.copy()

    def transform(self, X):
        if not self.is_fitted:
            raise NotFittedError(
                "predict or transform is not available if the estimator was not fitted"
            )

        # Validate X and check n_features_in_ consistency
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            ensure_2d=True,
            dtype=[np.float64, np.float32],
            reset=False,
        )

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

    def __sklearn_tags__(self):
        """
        Define sklearn tags for scikit-learn >= 1.6.

        Returns
        -------
        tags : sklearn.utils.Tags
        """
        from sklearn.utils import Tags, TargetTags, InputTags, RegressorTags

        tags = super().__sklearn_tags__()
        tags.target_tags = TargetTags(
            required=True,
            one_d_labels=True,
            two_d_labels=False,
            positive_only=False,
            multi_output=False,
            single_output=True,
        )
        tags.regressor_tags = RegressorTags(poor_score=True)
        return tags

    def _more_tags(self) -> Dict[str, bool]:
        """
        Override default sklearn tags for scikit-learn < 1.6.

        Returns
        -------
        tags : dict
        """
        return {"poor_score": True, "binary_only": True, "requires_y": True}
