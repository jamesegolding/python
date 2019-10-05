import scipy.linalg
import numpy as np


def calc_lqr_gain(A, B, Q, R):

    # first, try to solve the riccati equation
    s = scipy.linalg.solve_continuous_are(A, B, Q, R)

    # compute the LQR gain
    k = np.dot(np.linalg.inv(R), (np.dot(B.T, s)))

    eigs = np.linalg.eigvals(A - np.dot(B, k))

    return k


def update(s: np.ndarray,
           tgt: np.ndarray,
           k: np.ndarray,
           u_0: np.ndarray,
           ):

    x = np.array([s[2], s[9]]) - np.array([tgt[0], 0])

    u = u_0 - k @ x

    return u
