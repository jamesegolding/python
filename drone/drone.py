import numpy as np
import numba
import enum
import quaternion
from parameters import *
import utilities as utils


import logging
logger = logging.getLogger("drone")
logger.setLevel(logging.INFO)


class State(enum.Enum):
    x = 0
    y = 1
    z = 2
    q_0 = 3
    q_i = 4
    q_j = 5
    q_k = 6
    vx = 7
    vy = 8
    vz = 9
    o_0 = 10
    o_i = 11
    o_j = 12
    o_k = 13


motor_mat = np.array([
    [0.,       l_arm,   -l_arm,    0.],
    [l_arm,    0.,       0.,      -l_arm],
    [k_z_arm, -k_z_arm, -k_z_arm,  k_z_arm]
])


def vertical_state_space():

    a = np.array([
        [0., 1.],  # vz
        [0., -0.5 * rho * cd_z],  # gz
    ])

    b = np.array([
        [0., 0., 0., 0.],
        [1 / m, 1 / m, 1 / m, 1 / m],
    ])

    u_0 = 0.25 * m * G * np.ones(4)

    q = 100 * np.eye(2, 2)
    r = 1 * np.eye(4, 4)

    return a, b, q, r, u_0


def torque_motor_inv():

    return np.linalg.inv(np.vstack((np.ones((1, 4)), motor_mat)))


@numba.jit(nopython=True)
def calc_derivative(s: np.ndarray, u: np.ndarray, r_disturb: float = 0.):

    x = s[:3]    # translational position (world to body)
    q = s[3:7]   # quaternion (world to body)
    v = s[7:10]  # translational velocity (world to body)
    o = s[10:]   # quaternion rate (world to body)

    # motor performance limits
    u = utils.clip(u, a_min=0., a_max=f_motor_max)

    # calculate rotation matrix
    r = quaternion.to_rot_mat(q)
    w_body = quaternion.rate_matrix_body(q)
    n_body = quaternion.to_angular_velocity(w_body, o)

    # calculate contributions to translational force
    f_gravity_world = m * np.array([0., 0., -G])
    f_drag_body = -0.5 * rho * np.multiply(np.array([cd_xy, cd_xy, cd_z]), v)
    f_motor_body = np.array([0, 0, u.sum()])

    if x[2] < 0:
        f_ground_world = np.array([0., 0., -k_ground * x[2] - d_ground * v[2]])
    else:
        f_ground_world = np.array([0., 0., 0.])

    # calculate translational acceleration
    g = (1. / m) * (f_gravity_world + f_ground_world + r @ (f_drag_body + f_motor_body))

    # calculate contributions to torque
    t_motor = motor_mat @ u
    t_inert = np.array([(J_yy - J_zz) * n_body[1] * n_body[2],
                        (J_zz - J_xx) * n_body[2] * n_body[0],
                        (J_xx - J_yy) * n_body[0] * n_body[1]])
    t_drag = -0.5 * rho * np.multiply(np.array([cd_axy, cd_axy, cd_az]), n_body)

    # calculate angular acceleration (body)
    dn_body = np.multiply(np.array([1. / J_xx, 1. / J_yy, 1. / J_zz]), t_motor + t_inert + t_drag)

    # convert angular acceleration to quaternion acceleration
    do = quaternion.from_angular_acceleration(w_body, dn_body)

    ds = np.concatenate((v, o, g, do)) + calc_disturbance(r_disturb)

    return ds


@numba.jit(nopython=True)
def calc_disturbance(r_disturb: float) -> np.ndarray:

    d = np.concatenate((
        r_disturb * np.random.normal(0., std_v_dist, 7),
        r_disturb * np.random.normal(0., std_g_dist, 7),
    ))

    return d


@numba.jit(nopython=True)
def step(s_prev: np.ndarray, u: np.ndarray, dt: float, r_disturb: float = 0.):

    ds = calc_derivative(s_prev, u, r_disturb)

    # euler integration
    s = s_prev + ds * dt
    s[0:7] = s[0:7] + 0.5 * ds[7:] * dt ** 2

    # for safety, re-normalize q
    s[3:7] = utils.normalize(s[3:7])

    return s
