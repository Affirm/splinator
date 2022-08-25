import pytest
from splinator.estimators import LinearSplineLogisticRegression
from sklearn.utils.estimator_checks import check_estimator


@pytest.mark.parametrize(
    "estimator",
    [LinearSplineLogisticRegression()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
