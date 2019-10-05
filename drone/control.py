import scipy.linalg
import numpy as np
import quaternion
from parameters import *
import numba


def lqr_gain(A, B, Q, R):

    # first, try to solve the riccati equation
    s = scipy.linalg.solve_continuous_are(A, B, Q, R)

    # compute the LQR gain
    k = np.dot(np.linalg.inv(R), (np.dot(B.T, s)))

    eigs = np.linalg.eigvals(A - np.dot(B, k))

    return k


@numba.jit(nopython=True)
def update(s: np.ndarray,
           tgt: np.ndarray,
           k_z: np.ndarray,
           motr_inv: np.ndarray,
           u_0: np.ndarray,
           ):

    u_attitude = attitude(s, motr_inv)
    u_vertical = vertical(s, tgt, k_z)

    if max(u_attitude) - min(u_attitude) > f_motor_max:
        return 0.5 * f_motor_max * np.ones(4) + u_attitude

    u = u_0 + u_vertical + u_attitude

    if max(u) > f_motor_max:
        u = u - (max(u) - f_motor_max)

    return u


@numba.jit(nopython=True)
def vertical(s: np.ndarray,
             tgt: np.ndarray,
             k: np.ndarray,
             ):

    e_z = np.array([s[2], s[9]]) - np.array([tgt[0], 0])

    return -k @ e_z


@numba.jit(nopython=True)
def attitude(s: np.ndarray, motr_inv: np.ndarray):

    q_r = np.array([1., 0., 0., 0.])

    q = s[3:7]   # quaternion (world to body)
    o = s[10:]   # quaternion rate (world to body)

    e_q = quaternion.product(q_r, quaternion.conjugate(q))[1:]

    w_body = quaternion.rate_matrix_body(q)
    e_w = quaternion.to_angular_velocity(w_body, o)

    P_Q = 2.
    P_W = 0.5

    m_target = P_Q * e_q - P_W * e_w
    u = motr_inv @ np.concatenate((np.array([0.]), m_target))

    return u




