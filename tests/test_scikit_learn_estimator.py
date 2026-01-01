import pytest
from splinator.estimators import LinearSplineLogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks


@parametrize_with_checks(
    [LinearSplineLogisticRegression()],
    expected_failed_checks=lambda estimator: {
        # Optimization-based estimators may not achieve exact numerical equivalence
        # between weighted fitting and duplicate-data fitting due to optimizer tolerances.
        # Our test_sample_weight.py verifies sample_weight functionality more thoroughly.
        "check_sample_weight_equivalence_on_dense_data": (
            "Numerical optimization tolerances prevent exact equivalence"
        ),
    },
)
def test_all_estimators(estimator, check):
    check(estimator)
