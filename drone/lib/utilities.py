
import numpy as np
import numba

EPS = np.finfo(float).eps


@numba.jit(nopython=True)
def clip(a: np.ndarray, a_min: float = None, a_max: float = None):

    if a_min is not None:
        a = np.maximum(a, a_min)

    if a_max is not None:
        a = np.minimum(a, a_max)

    return a


@numba.jit(nopython=True)
def cross(a: np.ndarray, b: np.ndarray):

    e = np.zeros_like(a)
    e[0] = a[1] * b[2] - a[2] * b[1]
    e[1] = a[2] * b[0] - a[0] * b[2]
    e[2] = a[0] * b[1] - a[1] * b[0]

    return e


@numba.jit(nopython=True)
def norm2(a: np.ndarray):

    return np.sqrt(np.sum(np.multiply(a, a)))


@numba.jit(nopython=True)
def normalize(a: np.ndarray):
    """
    Calculate normalized quaternion
    :param a: array
    :return: normalized array
    """

    a_norm = norm2(a)

    if a_norm < EPS:
        return a

    return a / a_norm


@numba.jit(nopython=True)
def scale(a: np.ndarray, a_min: float, a_max: float):

    return a_min + (a - min(a)) * (a_max - a_min) / (max(a) - min(a))


@numba.jit(nopython=True)
def quadrant(a: float):

    a = a % (2 * np.pi)
    if a > np.pi:
        a = a - 2 * np.pi

    return a


@numba.jit(nopython=True)
def smooth_symmetric_clip(a: np.ndarray, a_clip: float):

    a = a_clip * np.sign(a) * (1 - np.exp(-np.abs(a) / a_clip))

    return a


@numba.jit(nopython=True)
def signed_sqrt(a):

    return np.sign(a) * np.sqrt(np.abs(a))

