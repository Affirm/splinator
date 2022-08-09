import pytest

from sklearn.utils.estimator_checks import check_estimator

from splinator import TemplateEstimator
from splinator import TemplateClassifier
from splinator import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
