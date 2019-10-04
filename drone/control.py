import scipy
import numpy as np


def calc_lqr_gain(A, B, Q, R):

    # first, try to solve the riccati equation
    s = scipy.linalg.solve_continuous_are(A, B, Q, R)

    # compute the LQR gain
    k = np.dot(np.linalg.inv(R), (np.dot(B.T, x)))

    eigs = np.linalg.eigvals(A - np.dot(B, k))

    return k, s, eigs


def update(x: np.ndarray,
           t: np.ndarray,
           k: np.ndarray,
           ):
    """

    :param x: state vector
    :param t: target (x, y, z, az)
    :param k: optimal gain matrix (LQR)
    :return: input vector
    """

    xyz_world = x[:2]
    xyz_world_target = t[:2]
    az_world_target = t[3]



    x_target = np.concatenate((
        t[0:3],
        t[
        t[0]
    ])

    u = -k @ x

    return t - u
