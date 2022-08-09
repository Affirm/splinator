import numpy as np


def assert_allclose_absolute(a, b, atol, allowed_not_close_fraction=0.0):
    # type: (Union[np.ndarray, pd.Series], Union[np.ndarray, pd.Series], float, float) -> None
    """
    Like np.testing.assert_allclose, but allows some fraction of values
    in the array to be outside the error range (defined by atol and rtol).
    """
    isclose = np.isclose(a, b, atol=atol)
    fraction_not_close = 1 - isclose.mean()

    # Some representative not-close values to display
    a_not_close = np.array(a)[~isclose][:10]
    b_not_close = np.array(b)[~isclose][:10]

    if fraction_not_close > allowed_not_close_fraction:
        # Run this assert function so it throws a nice error message.
        err_msg = (
            "{0:.2%} of the values are not close enough; allowed fraction is {1:.2%}\n"
            "Here's a sample of some mismatching values:\n{2}\n{3}"
        )
        err_msg = err_msg.format(fraction_not_close, allowed_not_close_fraction, a_not_close, b_not_close)
        np.testing.assert_allclose(a, b, atol=atol, err_msg=err_msg, verbose=True)
    elif fraction_not_close > 0:
        err_msg = (
            "{0:.2%} of the values are not close enough; "
            "this is not an error because {1:.2%} discrepancies are allowed\n"
            "Here's a sample of some mismatching values:\n{2}\n{3}".format(
                fraction_not_close, allowed_not_close_fraction, a_not_close, b_not_close
            )
        )