import scipy.linalg
import numpy as np
import numba

import drone
import utilities as utils
import quaternion
from parameters import *

import logging
logger = logging.getLogger("control")
logger.setLevel(logging.INFO)


# attitude control parameters
K_P = 0.1
K_V = 0.3
K_Q = 0.1
K_W = 0.05
THETA_MAX = np.pi / 8.


def lqr_gain(a, b, q, r):

    # first, try to solve the riccati equation
    s = scipy.linalg.solve_continuous_are(a, b, q, r)

    # compute the LQR gain
    k = np.dot(np.linalg.inv(r), (np.dot(b.T, s)))

    return k

# control derived parameters
MOTOR_INV = drone.torque_motor_inv()
A, B, Q, R, U_0 = drone.vertical_state_space()
K_Z = lqr_gain(A, B, Q, R)


def update(s: np.ndarray,
           tgt: np.ndarray,
           ):

    u_attitude = attitude(s, tgt[np.array([0, 1])])
    u_vertical = vertical(s, tgt[2])

    if max(u_attitude) - min(u_attitude) > f_motor_max:
        u = 0.5 * f_motor_max * np.ones(4) + u_attitude
    else:
        u = U_0 + u_vertical + u_attitude
        if max(u) > f_motor_max:
            u = u - (max(u) - f_motor_max)

    return u


@numba.jit(nopython=True)
def vertical(s: np.ndarray,
             tgt_z: np.ndarray,
             ):

    e_z = np.array([s[2], s[9], 0., 0., 0., 0.]) - np.array([tgt_z, 0., 0., 0., 0., 0.])

    u = -K_Z @ e_z

    # scale by vertical component of orientation
    q_real = s[3]
    u = u / utils.clip(q_real, 0.3)

    return u


def attitude(s: np.ndarray,
             tgt_xy_psi: np.ndarray,
             ):

    # get states
    q = s[3:7]   # quaternion (world to body)
    n_body = s[10:13]   # quaternion rate (world to body)

    # position targets
    e_p = tgt_xy_psi[np.array([0, 1])] - s[np.array([0, 1])]
    e_v = np.array([0, 0]) - s[np.array([7, 8])]

    q_r = np.array([1., 0., 0., 0.])
    #q_r = planar_ctrl_law(e_p, e_v)

    # overall target quaternion
    e_q = quaternion.product(q_r, quaternion.conjugate(q))[1:]

    m_xy = K_Q * e_q - K_W * n_body
    u = MOTOR_INV @ np.concatenate((np.array([0.]), m_xy))

    return u


def planar_ctrl_law(e_p, e_v):

    norm_e_p = utils.norm2(e_p)
    norm_e_v = utils.norm2(e_v)

    if norm_e_p > utils.EPS:
        # calculate axis to rotate about
        axis_p = utils.cross(e_p / norm_e_p, np.array([0., 0., 1.]))
        # calculate angle target
        theta_p = utils.clip(K_P * norm_e_p, a_min=-THETA_MAX, a_max=THETA_MAX)
        q_xy = np.concatenate((np.array([np.cos(theta_p / 2.)]),
                               np.sin(theta_p / 2.) * axis_p))

        # damping terms
        if norm_e_v > utils.EPS:
            # calculate axis to rotate about
            axis_v = utils.cross(e_v / norm_e_v, np.array([0., 0., 1.]))
            # calculate angle target
            theta_v = utils.clip(K_V * norm_e_v, a_min=-THETA_MAX, a_max=THETA_MAX)
            q_v = np.concatenate((np.array([np.cos(theta_v / 2.)]),
                                  np.sin(theta_v / 2.) * axis_v))
            q_xy = utils.normalize(quaternion.product(q_xy, q_v))
    else:
        q_xy = np.array([1., 0., 0., 0.])

    return q_xy


