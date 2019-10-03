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

    u = -k @ x

    return t - u
