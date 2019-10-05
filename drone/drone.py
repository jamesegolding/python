import numpy as np
from parameters import *
import quaternion
import numba

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def vertical_state_space():

    a = np.array([
        [0., 1.],
        [0., 0.],
    ])

    b = np.array([
        [0., 0., 0., 0.],
        [1 / m, 1 / m, 1 / m, 1 / m],
    ])

    u_0 = 0.25 * m * G * np.ones(4)

    q = 100 * np.eye(2, 2)
    r = 1 * np.eye(4, 4)

    return a, b, q, r, u_0


@numba.jit(nopython=True)
def calc_derivative(s: np.ndarray, u: np.ndarray):

    x = s[:3]    # translational position (world to body)
    q = s[3:7]   # quaternion (world to body)
    v = s[7:10]  # translational velocity (world to body)
    o = s[10:]   # quaternion rate (world to body)

    # motor performance limits
    u = custom_clip(u, 0., f_motor_max)

    # calculate rotation matrix
    r = quaternion.to_rot_mat(q)
    w_body = quaternion.rate_matrix_body(q)
    n_body = quaternion.to_angular_velocity(w_body, o)

    # calculate contributions to translational force
    f_grav_wrld = m * np.array([0., 0., -G])
    f_drag_body = -0.5 * rho * np.multiply(np.array([cd_xy, cd_xy, cd_z]), v)
    f_motr_body = np.array([0, 0, u.sum()])

    if x[2] < 0:
        f_grnd_wrld = np.array([0., 0., -k_ground * x[2] - d_ground * v[2]])
    else:
        f_grnd_wrld = np.array([0., 0., 0.])

    # calculate translational acceleration
    g = (1. / m) * (f_grav_wrld + f_grnd_wrld + r @ (f_drag_body + f_motr_body))

    # calculate contributions to torque
    t_motr = np.array([[0.,         l_xy_arm, -l_xy_arm,  0.],
                       [l_xy_arm,   0.,         0.,      -l_xy_arm],
                       [k_z_arm,   -k_z_arm,  -k_z_arm,   k_z_arm]]) @ u
    t_inert = np.array([(J_yy - J_zz) * n_body[1] * n_body[2],
                        (J_zz - J_xx) * n_body[2] * n_body[0],
                        (J_xx - J_yy) * n_body[0] * n_body[1]])

    # calculate angular acceleration (body)
    dn_body = np.multiply(np.array([1. / J_xx, 1. / J_yy, 1. / J_zz]), t_motr + t_inert)

    # convert angular acceleration to quaternion acceleration
    do = quaternion.from_angular_acceleration(w_body, dn_body)

    ds = np.concatenate((v, o, g, do))

    return ds


@numba.jit(nopython=True)
def custom_clip(a, a_min, a_max):

    for i in range(len(a)):
        if a[i] > a_max:
            a[i] = a_max
        elif a[i] < a_min:
            a[i] = a_min

    return a


def step(s_prev: np.ndarray, u: np.ndarray, dt: float):

    assert dt <= 0.01, f"dt must be less than 0.01, is {dt}"

    ds = calc_derivative(s_prev, u)

    # euler integration
    s = s_prev + ds * dt
    s[0:7] = s[0:7] + 0.5 * ds[7:] * dt ** 2

    # for safety, re-normalize q
    s[3:7] = quaternion.normalize(s[3:7])

    return s
