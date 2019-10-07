import scipy.linalg
import numpy as np
import numba
import utilities as utils
import quaternion
from parameters import *

import logging
logger = logging.getLogger("control")
logger.setLevel(logging.INFO)


# attitude control parameters
K_P = 0.5
K_V = 1.0
K_Q = 2.
K_W = 0.5
THETA_MAX = np.pi / 8


def lqr_gain(A, B, Q, R):

    # first, try to solve the riccati equation
    s = scipy.linalg.solve_continuous_are(A, B, Q, R)

    # compute the LQR gain
    k = np.dot(np.linalg.inv(R), (np.dot(B.T, s)))

    return k


def update(s: np.ndarray,
           tgt: np.ndarray,
           k_z: np.ndarray,
           motor_inv: np.ndarray,
           u_0: np.ndarray,
           ):

    u_attitude = attitude(s, tgt[np.array([0, 1, 3])], motor_inv)
    u_vertical = vertical(s, tgt[2], k_z)

    if max(u_attitude) - min(u_attitude) > f_motor_max:
        u = 0.5 * f_motor_max * np.ones(4) + u_attitude
    else:
        u = u_0 + u_vertical + u_attitude
        if max(u) > f_motor_max:
            u = u - (max(u) - f_motor_max)

    u = utils.clip(u, a_min=0., a_max=f_motor_max)

    return u


@numba.jit(nopython=True)
def vertical(s: np.ndarray,
             tgt_z: np.ndarray,
             k: np.ndarray,
             ):

    e_z = np.array([s[2], s[9]]) - np.array([tgt_z, 0])

    u = -k @ e_z

    # scale by vertical component of orientation
    q_real = s[3]
    u = u / utils.clip(q_real, 0.3)

    return u


def attitude(s: np.ndarray,
             tgt_xy_psi: np.ndarray,
             motor_inv: np.ndarray,
             ):

    # position and yaw targets
    e_p = tgt_xy_psi[np.array([0, 1])] - s[np.array([0, 1])]
    e_v = np.array([0, 0]) - s[np.array([7, 8])]
    r_psi = tgt_xy_psi[2]

    # get individual contributions
    q_xy = planar_ctrl_law(e_p, e_v)
    q_yaw = yaw_ctrl_law(r_psi)

    # overall target quaternion
    q_r = quaternion.product(q_xy, q_yaw)

    q = s[3:7]   # quaternion (world to body)
    o = s[10:]   # quaternion rate (world to body)

    e_q = quaternion.product(q_r, quaternion.conjugate(q))[1:]

    w_body = quaternion.rate_matrix_body(q)
    e_w = quaternion.to_angular_velocity(w_body, o)

    m_target = K_Q * e_q - K_W * e_w
    u = motor_inv @ np.concatenate((np.array([0.]), m_target))

    return u


def planar_ctrl_law(e_p, e_v):

    norm_e_p = utils.norm2(e_p)
    norm_e_v = utils.norm2(e_v)

    if norm_e_p > utils.EPS:
        # calculate axis to rotate about
        axis_p = utils.cross(e_p / norm_e_p, np.array([0, 0, 1]))
        # calculate angle target
        theta_p = utils.clip(K_P * norm_e_p, a_min=-THETA_MAX, a_max=THETA_MAX)
        q_xy = np.concatenate((np.array([np.cos(theta_p / 2.)]),
                               np.sin(theta_p / 2.) * axis_p))

        # damping terms
        if norm_e_v > utils.EPS:
            # calculate axis to rotate about
            axis_v = utils.cross(e_v / norm_e_v, np.array([0, 0, 1]))
            # calculate angle target
            theta_v = utils.clip(K_V * norm_e_v, a_min=-THETA_MAX, a_max=THETA_MAX)
            q_v = np.concatenate((np.array([np.cos(theta_v / 2.)]),
                                  np.sin(theta_v / 2.) * axis_v))
            q_xy = utils.normalize(quaternion.product(q_xy, q_v))
    else:
        q_xy = np.array([1., 0., 0., 0.])

    return q_xy


def yaw_ctrl_law(psi_r):

    # yaw target
    q_yaw = np.concatenate((
        np.array([np.cos(psi_r / 2.)]),
        np.sin(psi_r / 2.) * np.array([0, 0, 1]),
    ))

    return q_yaw


