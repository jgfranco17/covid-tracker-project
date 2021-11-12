import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer


def percent_diff(expected, actual) -> float:
    """
    Calculates the percent difference between 2 values

    Args:
        expected: Expected value
        actual: Real/acquired value

    Returns:
        Float value
    """
    sign = 1 if expected > actual else -1
    value = (abs(actual - expected) / ((actual + expected) / 2)) * 100
    return sign * round(value, 2)


def min_max_change(minimum, maximum, base_value) -> dict:
    """

    Args:
        minimum: Smaller value
        maximum: Larger value
        base_value: base number to compare to

    Returns:
        Dictionary of results
    """
    return {
        'min': percent_diff(minimum, base_value),
        'max': percent_diff(maximum, base_value)
    }


def logfunc(var, coeff_outer, coeff_inner, constant) -> float:
    """
    Return values from a general log function.
    """
    return coeff_outer * np.log(coeff_inner * var) + constant
