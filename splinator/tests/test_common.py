import pytest

from sklearn.utils.estimator_checks import check_estimator

from splinator import LinearSplineLogisticRegression


@pytest.mark.parametrize(
    "estimator",
    [LinearSplineLogisticRegression()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
