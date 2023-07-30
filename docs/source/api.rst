.. py:class:: LinearSplineLogisticRegression

    Piecewise Logistic Regression with Linear Splines

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    :param demo_param: str, default='demo_param'
        A parameter used for demonstration of how to pass and store parameters.

    :ivar coef_: ndarray of shape (n_features,)
        Coefficients of the linear spline logistic regression model.

    :ivar intercept_: float
        Intercept of the linear spline logistic regression model.

    :ivar knots_: ndarray of shape (n_knots,)
        The knot locations used in the linear spline logistic regression.

    :ivar monotonicity_: str, default='none'
        Whether the function is monotonically increasing or decreasing.

    :ivar n_features_in_: int
        Number of input features.

    :ivar two_stage_fitting_: bool
        Whether two-stage fitting is used.

    :ivar verbose_: bool
        Verbosity flag.

    :param input_score_column_index: int, default=0
        The index of the column containing input scores.

    :param n_knots: int, optional, default=100
        Number of knots used in the linear spline logistic regression.

    :param knots: list of float or np.ndarray, optional, default=None
        The knot locations used in the linear spline logistic regression.

    :param monotonicity: str, default='none'
        Whether to enforce that the function is monotonically increasing or decreasing.
        Valid values are 'none', 'increasing', and 'decreasing'.

    :param intercept: bool, default=True
        If True, allows the function value at x=0 to be nonzero.

    :param method: str, default='slsqp'
        The method named passed to scipy minimize. Supported values are 'slsqp' and 'trust-constr'.
        For more details, see the documentation for `scipy.optimize.minimize`.

    :param minimizer_options: dict, optional, default=None
        Some scipy minimizer methods have their special options.
        For example: {'disp': True} will display a termination report.
        For options, see the documentation for `scipy.optimize.minimize`.

    :param C: int, default=100
        Inverse of regularization strength; must be a positive float.
        Smaller values specify stronger regularization.

    :param two_stage_fitting_initial_size: int, optional, default=None
        Subsample size of training data for first fitting.
        If two-stage fitting is not used, this should be None.

    :param random_state: int, default=31
        Random seed number.

    :raises ValueError:
        If both `knots` and `n_knots` are non-null during fitting.

    :raises ValueError:
        If `monotonicity` has an invalid value.

    :raises ValueError:
        If `input_score_column_index` is negative.

    :raises ValueError:
        If `method` has an invalid value.

    :raises ValueError:

