import numpy as np
import orientation as ori
from parameters import *
import quaternion
import numba

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# state space A (der x = v_x)
a_linear = np.vstack((
    np.hstack((np.zeros((6, 6)), np.eye(6, 6))),
    np.hstack((np.zeros((3, 3)), np.diag([G, G, 0.]), np.zeros((3, 6)))),
    np.zeros((3, 12))
))

# state space B (motor effect)
b_linear = np.vstack((
    np.zeros((8, 4)),
    np.array([
        [1 / m,             1 / m,            1 / m,            1 / m],
        [l_xy_arm / J_xx,  -l_xy_arm / J_xx,  l_xy_arm / J_xx, -l_xy_arm / J_xx],
        [-l_xy_arm / J_yy, -l_xy_arm / J_yy,  l_xy_arm / J_yy,  l_xy_arm / J_yy],
        [k_z_arm / J_zz,    -k_z_arm / J_zz,   -k_z_arm / J_zz,    k_z_arm / J_zz],
    ]),
))

# state space offset (gravity and noise)
d_linear = np.concatenate((np.zeros(8), -G * np.ones(1), np.zeros(3)))


def state_space(x: np.ndarray, u: np.ndarray):

    assert isinstance(x, np.ndarray)
    assert isinstance(u, np.ndarray)

    assert len(x) == 12, f"State vector must have 12 elements, has {len(x)}"
    assert len(u) == 4, f"Input vector must have 4 elements, has {len(u)}"

    # cap motor torque
    f_motor_max = F_motor_max * np.ones(4)
    u = np.maximum(-f_motor_max, np.minimum(f_motor_max, u))

    # airborne mode blending
    if x[2] >= 0:
        q = 1
    else:
        q = 0

    a = a_linear.copy()
    b = b_linear.copy()
    d = d_linear.copy()

    # non linear mods for gravity
    a[6:8, 3:5] = 0.
    d[8] = d[8] + G
    d[6:9] = d[6:9] + ori.euler_to_rot_mat(x[3:7]) @ np.array([0., 0., -G])

    # non linear mods for air resistance
    a[6, 6] = -q * 0.5 * rho * cd_xy * x[6]
    a[7, 7] = -q * 0.5 * rho * cd_xy * x[7]
    a[8, 8] = -q * 0.5 * rho * cd_z * x[8]
    a[9, 9] = -q * 0.5 * rho * cd_axy * x[9]
    a[10, 10] = -q * 0.5 * rho * cd_axy * x[10]
    a[11, 11] = -q * 0.5 * rho * cd_az * x[11]

    # grounding
    a[(6, 7, 8, 9, 10, 11), (0, 1, 2, 3, 4, 5)] = (1 - q) * -k_ground
    a[(6, 7, 8, 9, 10, 11), (6, 7, 8, 9, 10, 11)] = (1 - q) * -d_ground

    # noise
    d[6:] = d[6:] + q * np.array([
        np.random.normal(0, std_g_xy_dist),
        np.random.normal(0, std_g_xy_dist),
        np.random.normal(0, std_g_z_dist),
        np.random.normal(0, std_dn_axy_dist),
        np.random.normal(0, std_dn_axy_dist),
        np.random.normal(0, std_dn_az_dist),
    ])

    x_dot = np.matmul(a, x) + np.matmul(b, u) + d

    return x_dot


def discrete_state_space(x_prev: np.ndarray, u: np.ndarray, dt: float):

    assert dt <= 0.01, f"dt must be less than 0.01, is {dt}"

    # angles within -pi to pi
    scale_outside = (np.abs(x_prev[3:6]) > np.pi).astype(float)

    x_prev[3:6] = x_prev[3:6] - np.multiply(scale_outside, 2 * np.pi * np.sign(x_prev[3:6]))

    x_dot = state_space(x_prev, u)

    # euler integration
    x = x_prev + x_dot * dt
    x[0:6] = x[0:6] + 0.5 * x_dot[6:] * dt ** 2

    return x


@numba.jit(nopython=True)
def quadcopter_model(s: np.ndarray, u: np.ndarray):

    x = s[:3]    # translational position (world to body)
    q = s[3:7]   # quaternion (world to body)
    v = s[7:10]  # translational velocity (world to body)
    o = s[10:]   # quaternion rate (world to body)

    # calculate rotation matrix
    r = quaternion.to_rot_mat(q)
    w_body = quaternion.rate_matrix_body(q)
    n_body = quaternion.to_angular_velocity_body(w_body, o)

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
    t_motr = np.array([[l_xy_arm,  -l_xy_arm,  l_xy_arm, -l_xy_arm],
                       [-l_xy_arm, -l_xy_arm,  l_xy_arm,  l_xy_arm],
                       [k_z_arm,   -k_z_arm,  -k_z_arm,   k_z_arm]]) @ u
    t_inert = np.array([(J_yy - J_zz) * n_body[1] * n_body[2],
                        (J_zz - J_xx) * n_body[2] * n_body[0],
                        (J_xx - J_yy) * n_body[0] * n_body[1]])

    # calculate angular acceleration (body)
    dn_body = np.multiply(np.array([1. / J_xx, 1. / J_yy, 1. / J_zz]), t_motr + t_inert)

    # convert angular acceleration to quaternion acceleration
    do = quaternion.from_angular_acceleration_body(w_body, dn_body)

    ds = np.concatenate((v, o, g, do))

    return ds


def step_quadcopter(s_prev: np.ndarray, u: np.ndarray, dt: float):

    assert dt <= 0.01, f"dt must be less than 0.01, is {dt}"

    ds = quadcopter_model(s_prev, u)

    # euler integration
    s = s_prev + ds * dt
    s[0:7] = s[0:7] + 0.5 * ds[7:] * dt ** 2

    # for safety, re-normalize q
    s[3:7] = quaternion.normalize(s[3:7])

    return s
