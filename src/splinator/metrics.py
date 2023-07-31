import numpy as np
from sklearn.calibration import calibration_curve


def spiegelhalters_z_statistic(
    labels,  # type: np.array
    preds,  # type: np.array
):
    # type: (...) -> float
    a = ((labels - preds) * (1 - 2 * preds)).sum()
    b = ((1 - 2 * preds) ** 2 * preds * (1 - preds)).sum()
    return float(a / b ** 0.5)


def expected_calibration_error(labels, preds, n_bins=10):
    # type: (np.array, np.array, int) -> float
    fop, mpv = calibration_curve(y_true=labels, y_prob=preds, n_bins=n_bins, strategy='quantile')
    diff = np.array(fop) - np.array(mpv)
    ece = sum([abs(delta) for delta in diff]) / float(n_bins)
    return ece
