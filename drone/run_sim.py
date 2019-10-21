import numpy as np
import time

from lib import control
from lib import drone
from lib import quaternion
from lib import sensor
from lib import utilities as utils
from lib.parameters import *
import post_process

import logging
logger = logging.getLogger("run_sim")
logger.setLevel(logging.INFO)


def run_sim(t: np.ndarray, target: np.ndarray,
            s0: np.ndarray,
            ):

    # initialize vectors
    s         = np.nan * np.ones((t.shape[0], 17))
    s_est     = np.nan * np.ones((t.shape[0], 17))
    g         = np.nan * np.ones((t.shape[0], 3))
    dn_body   = np.nan * np.ones((t.shape[0], 3))
    e         = np.nan * np.ones((t.shape[0], 3))
    e_est     = np.nan * np.ones((t.shape[0], 3))
    u         = np.nan * np.ones((t.shape[0], 4))
    g_sensor  = np.nan * np.ones((t.shape[0], 3))
    n_sensor  = np.nan * np.ones((t.shape[0], 3))
    en_sensor = np.nan * np.ones((t.shape[0], 3))

    # define initial conditions
    s[0, :] = s0
    s_est[0, :] = s0

    p = np.zeros((9, 9))
    p[6:, 6:] = std_g_xyz_noise ** 2 * np.eye(3)

    # initialize filter
    filter_state = sensor.Filter(s=s0, p=p, r_trans_sensor=0., r_madgwick_gain=0.02)

    print("Starting simulation...")

    t_start = time.time()
    # run sim
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]

        # controller update
        u[i, :] = control.update(s_est[i-1, :], target[i, :])

        # step model
        s[i, :], g[i, :], dn_body[i, :] = drone.step(s[i-1, :], u[i, :], dt)
        e[i, :] = quaternion.to_euler(s[i, 3:7])

        # sensor acquisition
        sensor_state = sensor.from_state(s[i, :], g[i, :])
        g_sensor[i, :] = sensor_state.accelerometer
        n_sensor[i, :] = sensor_state.gyroscope
        en_sensor[i, :] = sensor_state.magnetometer

        filter_state = sensor.to_state(sensor_state, filter_state, u[i, :], dt)
        s_est[i, :] = filter_state.s
        e_est[i, :] = quaternion.to_euler(s_est[i, 3:7])

        if i == 1:
            t_first = time.time()

    print(f"Simulation complete")
    print(f"First time step: {t_first - t_start:.2f} s")
    print(f"Remaining time steps: {1e3 * (time.time() - t_first) / (len(t) - 1):.2f} ms")

    summary = dict(
        t=t, s=s, s_est=s_est, g=g, dn_body=dn_body, u=u, e=e, e_est=e_est,
        g_sensor=g_sensor, n_sensor=n_sensor, en_sensor=en_sensor,
    )

    return summary


if __name__ == '__main__':

    # length and time step of simulations
    T_MAX = 10.
    DT = 0.002
    MODE = "Roll"

    t = np.arange(0, T_MAX, DT)

    # target trajectory
    tgt = np.zeros((len(t), 4))
    tgt[:, 0] = 0. * np.ones_like(t)
    tgt[:, 1] = 0. * np.ones_like(t)
    tgt[:, 2] = 5. * np.ones_like(t)

    psi = 1. * np.pi / 4.

    if MODE == "Pitch":
        axis = utils.normalize(np.array([0., 1., 0.]))
    elif MODE == "Roll":
        axis = utils.normalize(np.array([1., 0., 0.]))
    elif MODE == "Yaw":
        axis = utils.normalize(np.array([0., 0., 1.]))
    else:
        raise Exception("Unrecognised mode")

    # initial conditions
    s0 = np.zeros(17)
    s0[drone.State.x.value] = 0.
    s0[drone.State.y.value] = 0.
    s0[drone.State.z.value] = 5.
    s0[drone.State.q_0.value] = np.cos(psi / 2.)
    s0[drone.State.q_i.value] = np.sin(psi / 2.) * axis[0]
    s0[drone.State.q_j.value] = np.sin(psi / 2.) * axis[1]
    s0[drone.State.q_k.value] = np.sin(psi / 2.) * axis[2]
    s0[drone.State.vx.value] = 0.
    s0[drone.State.vy.value] = 0.
    s0[drone.State.vz.value] = 0.

    # run
    result = run_sim(t, tgt, s0)
    post_process.plot_result(result)

