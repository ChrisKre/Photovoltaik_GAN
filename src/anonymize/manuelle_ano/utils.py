import numpy as np
from scipy import signal


def get_first_last(x: np.ndarray, y: np.ndarray):
    """
    Get sunrise and sunset hour
    -> first and last values != 0
    :param x:
    :param y:
    :return:
    """
    x_1 = np.argwhere(x > 0)[0][0]
    x_l = np.argwhere(x > 0)[-1][0]
    y_1 = np.argwhere(y > 0)[0][0]
    y_l = np.argwhere(y > 0)[-1][0]
    return (x_1, x_l), (y_1, y_l)


def interpolate_to_koethen(day: np.ndarray, norm_day: np.ndarray):
    """
    Adjust length of daylighthours from location to length of daylighthours from normlocation
    :param day:
    :param norm_day:
    :return:
    """
    # Get sunrise-sunset
    (m_1, m_l), (k_1, k_l) = get_first_last(day, norm_day)
    # Determine dalyight curve
    curve = day[m_1 : m_l + 1]
    norm_day_curve = (k_l - k_1) + 1
    # Adjust from location to normlocation
    curve_stretched = signal.resample(curve, norm_day_curve)
    padded_array = np.zeros((24, 1))
    # Insert new daylight curve into padded array
    padded_array[k_1 : k_l + 1] = curve_stretched
    return padded_array
