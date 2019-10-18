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
    nx = 10
    ny = 11
    nz = 12


MOTOR_MAT = np.array([
    [0.,       l_arm,   -l_arm,    0.],
    [-l_arm,    0.,       0.,      l_arm],
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

    return np.linalg.inv(np.vstack((np.ones((1, 4)), MOTOR_MAT)))


#@numba.jit(nopython=True)
def calc_translational_derivative(s: np.ndarray, u: np.ndarray):

    x = s[:3]    # translational position (world to body)
    q = s[3:7]   # quaternion (world to body)
    v = s[7:10]  # translational velocity (world)

    # convert v into body frame
    v_body = quaternion.transform_inv(v, q)

    # calculate contributions to translational force
    f_gravity_w = m * np.array([0., 0., -G])
    cd_vec = np.array([cd_xy, cd_xy, cd_z])
    f_drag_b = -0.5 * rho * np.multiply(cd_vec, np.multiply(v_body, np.abs(v_body)))
    f_motor_b = np.array([0., 0., u.sum()])

    if x[2] < 0:
        f_ground_w = np.array([0., 0., -k_ground * x[2] - d_ground * v[2]])
    else:
        f_ground_w = np.array([0., 0., 0.])

    # calculate translational acceleration
    g = (1. / m) * (f_gravity_w + f_ground_w + quaternion.transform(f_drag_b + f_motor_b, q))

    return g


@numba.jit(nopython=True)
def calc_rotational_derivative(s: np.ndarray, u: np.ndarray) -> np.ndarray:

    n_body = s[10:13]  # angular velocities (body)

    # calculate contributions to torque
    t_motor = MOTOR_MAT @ u
    t_inert = np.array([(J_yy - J_zz) * n_body[1] * n_body[2],
                        (J_zz - J_xx) * n_body[2] * n_body[0],
                        (J_xx - J_yy) * n_body[0] * n_body[1]])
    cd_vec = np.array([cd_axy, cd_axy, cd_az])
    t_drag = -0.5 * rho * np.multiply(cd_vec, np.multiply(n_body, np.abs(n_body)))

    # calculate angular acceleration (body)
    dn_body = np.multiply(np.array([1. / J_xx, 1. / J_yy, 1. / J_zz]), t_motor + t_inert + t_drag)

    return dn_body


#@numba.jit(nopython=True)
def calc_derivative(s: np.ndarray, u: np.ndarray):

    # motor performance limits
    u = utils.clip(u, a_min=0., a_max=f_motor_max)

    # rates
    v = s[7:10]
    n_body = s[10:13]

    # accelerations
    g = calc_translational_derivative(s, u) + np.random.normal(0., std_g_dist, 3)
    dn_body = calc_rotational_derivative(s, u) + np.random.normal(0., std_dn_dist, 3)

    return v, n_body, g, dn_body


#@numba.jit(nopython=True)
def step(s: np.ndarray, u: np.ndarray, dt: float):

    v, n_body, g, dn_body = calc_derivative(s, u)

    # euler integration
    s[:3]  = s[:3] + v * dt + 0.5 * g * dt ** 2
    s[3:7] = quaternion.integrate(s[3:7], n_body, dt)
    s[7:10] = s[7:10] + g * dt
    s[10:13] = s[10:13] + dn_body * dt

    # for safety, re-normalize q
    #s[3:7] = utils.normalize(s[3:7])

    return s, g, dn_body
