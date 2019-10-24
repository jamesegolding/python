
import numpy as np
import numba
import collections

from lib import drone, quaternion, utilities as utils
from lib.parameters import *

# state definitions
Sensor = collections.namedtuple('Sensor', ['accelerometer', 'gyroscope', 'magnetometer'])
Filter = collections.namedtuple('Filter', ['s', 'p', 'r_trans_sensor', 'r_madgwick_gain'])

THETA_MAG = np.pi / 3.  # magnetic field elevation
CM = np.cos(THETA_MAG)
SM = np.sin(THETA_MAG)

V_CD_GAIN = 0.05  # correction factor for drag induced acceleration

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

    s = filter_state.s

    # calculate orientation
    df_motor = drone.calc_motor_derivative(filter_state.s, u)
    s[13:17] = utils.clip(s[13:17] + df_motor * dt, a_min=0., a_max=f_motor_max)

    filter_state = Filter(s=s,
                          p=filter_state.p,
                          r_trans_sensor=filter_state.r_trans_sensor,
                          r_madgwick_gain=filter_state.r_madgwick_gain,
                          )

    q, n_body = madgwick(sensor_state, filter_state, dt)
    x, v = filter_translational(sensor_state, filter_state, u, dt)

    # update state vector
    s[:3] = x
    s[3:7] = q
    s[7:10] = v
    s[10:13] = n_body

    # update filter (can't _replace in numba)
    filter_state = Filter(s=s,
                          p=filter_state.p,
                          r_trans_sensor=filter_state.r_trans_sensor,
                          r_madgwick_gain=filter_state.r_madgwick_gain,
                          )

    return filter_state


@numba.jit(nopython=True)
def filter_translational(sensor_state: Sensor,
                         filter_state: Filter,
                         u: np.ndarray,
                         dt: float,
                         ):

    # get working variables
    s = filter_state.s
    r_sensor = filter_state.r_trans_sensor
    accel = sensor_state.accelerometer

    x = s[:3]
    v = s[7:10]
    q = s[3:7]

    g_model = drone.calc_translational_derivative(s, u)
    g_sensor = quaternion.transform(accel, q) - np.array([0., 0., G])

    # merge
    g = (1 - r_sensor) * g_model + r_sensor * g_sensor
    x = x + v * dt + 0.5 * dt ** 2 * g
    v = v + g * dt

    # estimate translational velocity based on xy accel
    v_cd_xy_est = utils.signed_sqrt(-2 * m * accel[:2] / rho / cd_xy)
    v_cd_est = quaternion.transform(np.array([v_cd_xy_est[0], v_cd_xy_est[1], 0.]), q)
    v[:2] = v[:2] + (v_cd_est[:2] - v[:2]) * V_CD_GAIN

    return x, v


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


def kalman_orientation_h(s: np.ndarray):
    """
    Get derivative of observation wrt state vector
    :param s: reduced orientation state vector [q, n_body]
    :return: H
    """

    q = s[3:7]

    d_accel_d_q = G * np.array([
        [2 * q[2],   2 * q[3],  2 * q[0],  2 * q[1]],
        [-2 * q[1], -2 * q[0],  2 * q[3],  2 * q[2]],
        [2 * q[0],  -2 * q[1], -2 * q[2],  2 * q[3]],
    ])

    d_gyro_d_n_body = np.eye(3)

    d_mag_d_q = np.array([
        [q[0] * CM + q[2] * SM,  q[1] * CM + q[3] * SM,  q[2] * CM + q[0] * SM,  q[3] * CM + q[1] * SM],
        [q[3] * CM - q[1] * SM,  q[2] * CM - q[0] * SM,  q[1] * CM + q[3] * SM,  q[0] * CM + q[2] * SM],
        [-q[2] * CM + q[0] * SM, q[3] * CM - q[1] * SM, -q[0] * CM - q[2] * SM,  q[1] * CM + q[3] * SM],
    ])

    h = np.vstack((
        np.hstack((d_accel_d_q, np.zeros((3, 3)))),
        np.hstack((np.zeros((3, 4)), d_gyro_d_n_body)),
        np.hstack((d_mag_d_q, np.zeros((3, 3)))),
    ))

    return h
