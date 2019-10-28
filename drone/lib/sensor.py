
import numpy as np
import numba
import collections

from lib import drone, quaternion, utilities as utils
from lib.parameters import *

# state definitions
Sensor = collections.namedtuple('Sensor', ['accelerometer', 'gyroscope', 'magnetometer'])
Filter = collections.namedtuple('Filter', ['s', 'p', 'r_madgwick_gain', 'g'])

ORIENTATION_FILTER = "Kalman"

THETA_MAG = np.pi / 3.  # magnetic field elevation
CM = np.cos(THETA_MAG)
SM = np.sin(THETA_MAG)

MOTOR_MAT = drone.MOTOR_MAT

V_CD_GAIN = 0.0  # correction factor for drag induced acceleration

@numba.jit(nopython=True)
def from_state(s: np.ndarray, g: np.ndarray, r_scale: float = 1.):

    q = s[3:7]
    n_body = s[10:13]

    # transform acceleration into body frame
    g_world = g + np.array([0., 0., G])
    g_body = quaternion.transform_inv(g_world, q)
    accelerometer = g_body + r_scale * np.random.normal(0., std_g_xyz_noise, 3)

    # convert quaternion acceleration to angular acceleration
    gyroscope = n_body + r_scale * np.random.normal(0., std_n_axyz_noise, 3)

    # calculate magnetometer heading
    e_compass_world = np.array([CM, 0., SM])
    e_compass_body = quaternion.transform_inv(e_compass_world, q)
    magnetometer = e_compass_body + r_scale * np.random.normal(0., std_e_mag_noise, 3)

    sensor_state = Sensor(accelerometer=accelerometer, gyroscope=gyroscope, magnetometer=magnetometer)

    return sensor_state


@numba.jit(nopython=True)
def to_state(sensor_state: Sensor,
             filter_state: Filter,
             u: np.ndarray,
             dt: float,
             ):
    s_filter = filter_state.s

    if ORIENTATION_FILTER == "Kalman":
        q, n_body, p = kalman_orientation(sensor_state, filter_state, u, dt)
    elif ORIENTATION_FILTER == "Madgwick":
        q, n_body = madgwick(sensor_state, filter_state, dt)
        p = np.zeros((11, 11))
    else:
        q, n_body, p = np.zeros(4), np.zeros(3), np.zeros((11, 11))

    x, v, g = filter_translational(sensor_state, filter_state, u, dt)

    # calculate motor force
    df_motor = drone.calc_motor_derivative(filter_state.s, u)

    # update state vector
    s_filter[:3] = x
    s_filter[3:7] = q
    s_filter[7:10] = v
    s_filter[10:13] = n_body
    s_filter[13:17] = utils.clip(s_filter[13:17] + df_motor * dt, a_min=0., a_max=f_motor_max)

    # update filter (can't _replace in numba)
    filter_state = Filter(s=s_filter,
                          p=p,
                          r_trans_sensor=filter_state.r_trans_sensor,
                          r_madgwick_gain=filter_state.r_madgwick_gain,
                          g=g,
                          )

    return filter_state


@numba.jit(nopython=True)
def filter_translational(sensor_state: Sensor,
                         filter_state: Filter,
                         u: np.ndarray,
                         dt: float,
                         ):

    # get working variables
    s_est = filter_state.s
    accel = sensor_state.accelerometer

    x = s_est[:3]
    v = s_est[7:10]
    q = s_est[3:7]

    # dead reckoning
    g = quaternion.transform(accel, q) - np.array([0., 0., G])

    # integrate
    x = x + v * dt + 0.5 * dt ** 2 * g
    v = v + g * dt

    # estimate translational velocity based on xy accel
    v_cd_xy_est = utils.signed_sqrt(-2 * m * accel[:2] / rho / cd_xy)
    v_cd_est = quaternion.transform_inv(np.array([v_cd_xy_est[0], v_cd_xy_est[1], 0.]), q)
    v[:2] = v[:2] + (v_cd_est[:2] - v[:2]) * V_CD_GAIN

    return x, v, g


@numba.jit(nopython=True)
def madgwick(sensor_state: Sensor,
             filter_state: Filter,
             dt: float
             ):

    # get working variables
    q = filter_state.s[3:7]
    gain = filter_state.r_madgwick_gain
    g_accel = sensor_state.accelerometer
    n_gyro = sensor_state.gyroscope
    e_magnet = sensor_state.magnetometer

    # find length of vectors
    g_norm = utils.norm2(g_accel)
    e_norm = utils.norm2(e_magnet)

    if (g_norm > 0.01) and (e_norm > 0.01):
        # normalize
        g_accel = g_accel / g_norm
        e_magnet = e_magnet / e_norm

        h = quaternion.transform(e_magnet, q)
        b = np.array([0., utils.norm2(h[0:2]), 0., h[2]])

        # gradient descent step
        f = np.array([
            2 * (q[1] * q[3] - q[0] * q[2]) - g_accel[0],
            2 * (q[0] * q[1] + q[2] * q[3]) - g_accel[1],
            2 * (0.5 - q[1] ** 2 - q[2] ** 2) - g_accel[2],
            2 * b[1] * (0.5 - q[2] ** 2 - q[3] ** 2) + 2 * b[3] * (q[1] * q[3] - q[0] * q[2]) - e_magnet[0],
            2 * b[1] * (q[1] * q[2] - q[0] * q[3]) + 2 * b[3] * (q[0] * q[1] + q[2] * q[3]) - e_magnet[1],
            2 * b[1] * (q[0] * q[2] + q[1] * q[3]) + 2 * b[3] * (0.5 - q[1] ** 2 - q[2] ** 2) - e_magnet[2],
        ])

        j = np.array([
            [-2*q[2],                   2*q[3],                  -2*q[0],                   2*q[1]],
            [2*q[1],                    2*q[0],                   2*q[3],                   2*q[2]],
            [0,                        -4*q[1],                  -4*q[2],                   0],
            [-2*b[3]*q[2],              2*b[3]*q[3],             -4*b[1]*q[2]-2*b[3]*q[0], -4*b[1]*q[3]+2*b[3]*q[1]],
            [-2*b[1]*q[3]+2*b[3]*q[1],  2*b[1]*q[2]+2*b[3]*q[0],  2*b[1]*q[1]+2*b[3]*q[3], -2*b[1]*q[0]+2*b[3]*q[2]],
            [2*b[1]*q[2],               2*b[1]*q[3]-4*b[3]*q[1],  2*b[1]*q[0]-4*b[3]*q[2],  2*b[1]*q[1]],
        ])

        # get step update from accelerometer and magnetometer
        o_step = -1 * gain * utils.normalize(j.T @ f)

        w_body = quaternion.rate_matrix_body(q)
        n_step = quaternion.to_angular_velocity(w_body, o_step)

    else:
        n_step = np.zeros(3)

    # integrate
    n_body = n_gyro + n_step
    q = quaternion.integrate(q, n_body, dt)

    return q, n_body


@numba.jit(nopython=True)
def kalman_orientation(sensor_state: Sensor,
                       filter_state: Filter,
                       u: np.ndarray,
                       dt: float,
                       ):

    s_prev = filter_state.s.copy()
    s_prior, g_prior, _ = drone.step(s_prev, u, dt, r_scale_dist=0.)
    x_pred = np.concatenate((s_prior[3:7], s_prior[10:17]))

    # get sensitivities
    f_mat = kalman_orientation_f(s_prev, dt)
    h_mat = kalman_orientation_h(s_prev)

    # get model disturbance expectation
    g = np.concatenate((
        0.5 * dt ** 2 * std_dn_dist * np.ones(4),
        dt * std_dn_dist * np.ones(3),
        dt * std_motor * np.ones(4)))
    v = g.reshape(-1, 1) @ g.reshape(1, -1)

    # get prior p matrix
    p_pred = f_mat @ filter_state.p @ f_mat.T + v

    sensor_state_pred = from_state(s_prior, g=g_prior, r_scale=0.)
    z = np.concatenate((sensor_state.accelerometer, sensor_state.gyroscope, sensor_state.magnetometer))
    z_pred = np.concatenate((sensor_state_pred.accelerometer, sensor_state_pred.gyroscope, sensor_state_pred.magnetometer))
    y = z - z_pred

    # get measurement noise expectation
    r = np.diag(np.array([
        std_g_xyz_noise, std_g_xyz_noise, std_g_xyz_noise,
        std_n_axyz_noise, std_n_axyz_noise, std_n_axyz_noise,
        std_e_mag_noise, std_e_mag_noise, std_e_mag_noise,
    ]))

    p_h = p_pred @ np.ascontiguousarray(h_mat.T)
    s_mat = h_mat @ p_h + r
    k = p_h @ np.linalg.solve(s_mat, np.eye(s_mat.shape[0]))

    x_update = x_pred + k @ y
    p_update = (np.eye(11) - k @ h_mat) @ p_pred

    q_update = x_update[:4]
    n_body_update = x_update[4:7]

    return q_update, n_body_update, p_update


@numba.jit(nopython=True)
def kalman_orientation_h(s: np.ndarray):
    """
    Get derivative of observation wrt state vector
    :param s: full state vector
    :return: H matrix
    """

    q = s[3:7]

    e_gravity_world = np.array([0., 0., G])
    e_compass_world = np.array([CM, 0., SM])

    dr_inv_dq0 = quaternion.rot_mat_der(q, 0, b_inverse=False)
    dr_inv_dq1 = quaternion.rot_mat_der(q, 1, b_inverse=False)
    dr_inv_dq2 = quaternion.rot_mat_der(q, 2, b_inverse=False)
    dr_inv_dq3 = quaternion.rot_mat_der(q, 3, b_inverse=False)

    d_accel_d_q = np.empty((3, 4))
    d_accel_d_q[:, 0] = dr_inv_dq0 @ e_gravity_world
    d_accel_d_q[:, 1] = dr_inv_dq1 @ e_gravity_world
    d_accel_d_q[:, 2] = dr_inv_dq2 @ e_gravity_world
    d_accel_d_q[:, 3] = dr_inv_dq3 @ e_gravity_world

    d_gyro_d_n_body = np.eye(3)

    d_mag_d_q = np.empty((3, 4))
    d_mag_d_q[:, 0] = dr_inv_dq0 @ e_compass_world
    d_mag_d_q[:, 1] = dr_inv_dq1 @ e_compass_world
    d_mag_d_q[:, 2] = dr_inv_dq2 @ e_compass_world
    d_mag_d_q[:, 3] = dr_inv_dq3 @ e_compass_world

    h = np.vstack((
        np.hstack((d_accel_d_q, np.zeros((3, 7)))),
        np.hstack((np.zeros((3, 4)), d_gyro_d_n_body, np.zeros((3, 4)))),
        np.hstack((d_mag_d_q, np.zeros((3, 7)))),
    ))

    return h


@numba.jit(nopython=True)
def kalman_orientation_f(s: np.ndarray, dt: float):
    """
    Get derivative of model step function wrt state vector
    :param s: full state vector
    :param dt: time step
    :return: F matrix
    """
    q = s[3:7]
    n_body = s[10:13]  # angular velocities (body)

    inertia_mat = np.diag(np.array([1. / J_xx, 1. / J_yy, 1. / J_zz]))

    # calculate contributions to torque
    d_motor = inertia_mat @ MOTOR_MAT
    d_inert = inertia_mat @ np.array([
        [0., (J_yy - J_zz) * n_body[2], (J_yy - J_zz) * n_body[1]],
        [(J_zz - J_xx) * n_body[2], 0., (J_zz - J_xx) * n_body[0]],
        [(J_xx - J_yy) * n_body[1], (J_xx - J_yy) * n_body[0], 0.],
    ])

    n_body_abs = np.abs(n_body)
    d_drag = -rho * inertia_mat @ np.diag(np.array([cd_axy * n_body_abs[0], cd_axy * n_body_abs[1], cd_az * n_body_abs[2]]))

    q_der_q, q_der_n = quaternion.integrate_der(q, n_body, dt)

    return np.vstack((
        np.hstack((q_der_q, q_der_n, np.zeros((4, 4)))),
        np.hstack((np.zeros((3, 4)), np.eye(3) + dt * (d_inert + d_drag), dt * d_motor)),
        np.hstack((np.zeros((4, 7)), (1 - dt / tau_motor) * np.eye(4))),
    ))


def kalman_orientation_f_finite_difference(s: np.ndarray, u: np.ndarray, dt: float):

    # compute a finite difference H matrix
    eps = 0.00001
    f_num = np.zeros((11, 11))
    i_states = [3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16]
    for i_s, i_reduce in zip(i_states, range(11)):
        ds = np.zeros(17)
        ds[i_s] = eps
        s_pos, _, _ = drone.step(s + ds, u, dt, r_scale_dist=0.)
        s_neg, _, _ = drone.step(s - ds, u, dt, r_scale_dist=0.)
        f_num[:, i_reduce] = (s_pos[i_states] - s_neg[i_states]) / 2 / eps

    return f_num


def kalman_orientation_h_finite_difference(s: np.ndarray):
    # compute a finite difference H matrix
    eps = 0.00001
    h_num = np.zeros((9, 11))
    i_states = [3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16]
    for i_s, i_reduce in zip(i_states, range(7)):
        ds = np.zeros(17)
        ds[i_s] = eps
        sensor_state_p = from_state(s + ds, g=np.zeros(3), r_scale=0.)
        sensor_state_n = from_state(s - ds, g=np.zeros(3), r_scale=0.)
        h_num[:, i_reduce] = np.concatenate((
            (sensor_state_p.accelerometer - sensor_state_n.accelerometer) / 2 / eps,
            (sensor_state_p.gyroscope - sensor_state_n.gyroscope) / 2 / eps,
            (sensor_state_p.magnetometer - sensor_state_n.magnetometer) / 2 / eps,
        ))

    return h_num