
import numpy as np
import numba
import collections

import drone
import quaternion
import utilities as utils
from parameters import *

# state definitions
Sensor = collections.namedtuple('Sensor', ['accelerometer', 'gyroscope', 'magnetometer'])
Filter = collections.namedtuple('Filter', ['s', 'p', 'r_trans_sensor', 'r_madgwick_gain'])


@numba.jit(nopython=True)
def from_state(s: np.ndarray, g: np.ndarray):

    q = s[3:7]
    n_body = s[10:13]

    # transform acceleration into body frame
    g_body = quaternion.transform_inv(g + np.array([0., 0., G]), q)
    accelerometer = g_body + np.random.normal(0., std_g_xyz_noise, 3)

    # convert quaternion acceleration to angular acceleration
    gyroscope = n_body + np.random.normal(0., std_n_axyz_noise, 3)

    # calculate magnetometer heading
    magnetometer = quaternion.to_rot_mat(q)[:, 0] + np.random.normal(0., std_e_mag_noise, 3)

    sensor_state = Sensor(accelerometer=accelerometer, gyroscope=gyroscope, magnetometer=magnetometer)

    return sensor_state


#@numba.jit(nopython=True)
def to_state(sensor_state: Sensor,
             filter_state: Filter,
             u: np.ndarray,
             dt: float,
             s: np.ndarray,
             ):

    # calculate orientation
    q, n_body = madgwick(sensor_state, filter_state, dt)
    x, v = filter_translational(sensor_state, filter_state, u, dt, s)

    # update state vector
    s = filter_state.s
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


def kalman_fo_real(sensor_state: Sensor,
                   filter_state: Filter,
                   u: np.ndarray,
                   dt: float,
                   ):

    # get working variables
    s = filter_state.s
    p = filter_state.p
    accel = sensor_state.accelerometer

    x = s[:3]
    v = s[7:10]
    q = s[3:7]

    u = utils.clip(u, a_min=0., a_max=f_motor_max)
    g_model = drone.calc_translational_derivative(s, u)
    g_sensor = quaternion.transform_inv(accel, q) - np.array([0., 0., G])

    # useful bits
    zeros = np.zeros((3, 3))
    ones = np.eye(3)

    # drag model
    r = quaternion.to_rot_mat(q)
    r_inv = quaternion.to_rot_mat(quaternion.conjugate(q))
    cd_vec = np.array([cd_xy, cd_xy, cd_z])
    drag = -0.5 * rho / m * r @ np.diag(np.multiply(cd_vec, v))

    # system discrete state derivative
    f = np.vstack((
        np.hstack((ones, dt * ones, 0.5 * dt ** 2 * ones)),
        np.hstack((zeros, ones, dt * ones)),
        np.hstack((zeros, drag, zeros)),
    ))

    # system noise derivative
    g = np.vstack((
        0.5 * dt ** 2 * ones,
        dt * ones,
        ones,
    ))
    q = g @ g.T * std_g_dist ** 2
    r = std_g_xyz_noise ** 2 * ones

    # sensor state matrix (h = dzdx)
    h = np.hstack((zeros, zeros, r_inv))

    # assemble priori
    x_prior = np.concatenate((
        x + v * dt + 0.5 * dt ** 2 * g_model,
        v + dt * g_model,
        g_model,
    ))

    p_prior = f @ p @ f.T + q
    s = h @ p @ h.T + r

    k = p @ h.T @ np.linalg.inv(s)

    # update
    x_post = x_prior + k * (g_sensor - g_model)
    p = (np.eye(3) - k @ h) @ p

    return x_post[:3], x_post[3:6], p


#@numba.jit(nopython=True)
def filter_translational(sensor_state: Sensor,
                         filter_state: Filter,
                         u: np.ndarray,
                         dt: float,
                         s_actual: np.ndarray,
                         ):

    # get working variables
    s = filter_state.s
    r_sensor = filter_state.r_trans_sensor
    accel = sensor_state.accelerometer

    x = s[:3]
    v = s[7:10]
    q = s[3:7]

    u = utils.clip(u, a_min=0., a_max=f_motor_max)
    g_model = drone.calc_translational_derivative(s, u)
    g_sensor = quaternion.transform(accel, q) - np.array([0., 0., G])

    #print(g_model)
    #print(g_sensor)

    # merge
    g = (1 - r_sensor) * g_model + r_sensor * g_sensor

    x = x + v * dt + 0.5 * dt ** 2 * g
    v = v + g * dt

    return x, v


#@numba.jit(nopython=True)
def madgwick(sensor_state: Sensor,
             filter_state: Filter,
             dt: float
             ):

    # get working variables
    q = filter_state.s[3:7]
    gain = filter_state.r_madgwick_gain
    accel = sensor_state.accelerometer
    gyro = sensor_state.gyroscope
    magnet = sensor_state.magnetometer

    # normalize accelerometer
    a_norm = utils.norm2(accel)
    m_norm = utils.norm2(magnet)

    if (a_norm > 0.) and (m_norm > 0.):

        h = quaternion.transform(magnet, q)
        b = np.array([0., utils.norm2(h[0:2]), 0., h[2]])

        # gradient descent step
        f = np.array([
            2 * (q[1] * q[3] - q[0] * q[2]) - accel[0],
            2 * (q[0] * q[1] + q[2] * q[3]) - accel[1],
            2 * (0.5 - q[1] ** 2 - q[2] ** 2) - accel[2],
            2 * b[1] * (0.5 - q[2] ** 2 - q[3] ** 2) + 2 * b[3] * (q[1] * q[3] - q[0] * q[2]) - magnet[0],
            2 * b[1] * (q[1] * q[2] - q[0] * q[3]) + 2 * b[3] * (q[0] * q[1] + q[2] * q[3]) - magnet[1],
            2 * b[1] * (q[0] * q[2] + q[1] * q[3]) + 2 * b[3] * (0.5 - q[1] ** 2 - q[2] ** 2) - magnet[2],
        ])

        j = np.array([
            [-2*q[2],                   2*q[3],                  -2*q[0],                   2*q[1]],
            [2*q[1],                    2*q[0],                   2*q[3],                   2*q[2]],
            [0,                        -4*q[1],                  -4*q[2],                   0],
            [-2*b[3]*q[2],              2*b[3]*q[3],             -4*b[1]*q[2]-2*b[3]*q[0], -4*b[1]*q[3]+2*b[3]*q[1]],
            [-2*b[1]*q[3]+2*b[3]*q[1],  2*b[1]*q[2]+2*b[3]*q[0],  2*b[1]*q[1]+2*b[3]*q[3], -2*b[1]*q[0]+2*b[3]*q[2]],
            [2*b[1]*q[2],               2*b[1]*q[3]-4*b[3]*q[1],  2*b[1]*q[0]-4*b[3]*q[2],  2*b[1]*q[1]],
        ])

        step = utils.normalize(j.T @ f)

        # rate of change of q
        o_step = -gain * step

        w_body = quaternion.rate_matrix_body(q)
        n_step = quaternion.to_angular_velocity(w_body, o_step)

        n_body = gyro + n_step

    else:
        n_body = np.zeros(3)

    # integrate
    q = quaternion.integrate(q, n_body, dt)

    return q, n_body