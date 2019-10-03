import numpy as np
import orientation as ori
from parameters import *

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
        [l_xy_arm / J_xy,  -l_xy_arm / J_xy,  l_xy_arm / J_xy, -l_xy_arm / J_xy],
        [-l_xy_arm / J_xy, -l_xy_arm / J_xy,  l_xy_arm / J_xy,  l_xy_arm / J_xy],
        [k_z_arm / J_z,    -k_z_arm / J_z,   -k_z_arm / J_z,    k_z_arm / J_z],
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
