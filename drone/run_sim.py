import numpy as np
import plotly
import time

import drone
import sensor
import quaternion
import control
from parameters import *
import visuals
import utilities as utils

import logging
logger = logging.getLogger("run_sim")
logger.setLevel(logging.INFO)


def run_sim(t: np.ndarray, target: np.ndarray,
            s0: np.ndarray,
            ):

    # initialize vectors
    s = np.empty((t.shape[0], 13));          s[:] = np.nan
    s_est = np.empty((t.shape[0], 13));      s_est[:] = np.nan
    g = np.empty((t.shape[0], 3));           g[:] = np.nan
    dn_body = np.empty((t.shape[0], 3));     dn_body[:] = np.nan
    e = np.empty((t.shape[0], 3));           e[:] = np.nan
    e_est = np.empty((t.shape[0], 3));       e_est[:] = np.nan
    u = np.empty((t.shape[0], 4));           u[:] = np.nan
    g_sensor = np.empty((t.shape[0], 3));    g_sensor[:] = np.nan
    n_sensor = np.empty((t.shape[0], 3));    n_sensor[:] = np.nan
    en_sensor = np.empty((t.shape[0], 3));   en_sensor[:] = np.nan

    # define initial conditions
    s[0, :] = s0
    s_est[0, :] = s0

    p = np.zeros((9, 9))
    p[6:, 6:] = std_g_xyz_noise ** 2 * np.eye(3)

    # initialize filter
    filter_state = sensor.Filter(s=s0, p=p, r_trans_sensor=0., r_madgwick_gain=0.05)

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

        filter_state = sensor.to_state(sensor_state, filter_state, u[i, :], dt, s[i, :])
        s_est[i, :] = filter_state.s
        e_est[i, :] = quaternion.to_euler(s_est[i, 3:7])

        if i == 1:
            t_first = time.time()

    print(f"Simulation complete")
    print(f"First time step: {t_first - t_start:.2f} seconds")
    print(f"Remaining time steps: {(time.time() - t_first) / (N_SAMPLES - 1):.5f} seconds")


    summary = dict(
        t=t, s=s, s_est=s_est, g=g, dn_body=dn_body, u=u, e=e, e_est=e_est,
        g_sensor=g_sensor, n_sensor=n_sensor, en_sensor=en_sensor,
    )

    return summary


def plot_result(result):

    # Summary
    data = dict()
    layout = dict()
    data['Position'] = [
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 0].squeeze(), name="X"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 1].squeeze(), name="Y"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 2].squeeze(), name="Z"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s_est"][:, 0].squeeze(), name="X est", line={'dash': 'dash'}),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s_est"][:, 1].squeeze(), name="Y est", line={'dash': 'dash'}),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s_est"][:, 2].squeeze(), name="Z est", line={'dash': 'dash'}),
    ]
    layout['Position'] = dict(xaxis=dict(title="Time [s]"), yaxis=dict(title="Position [m]"))

    data['Euler Angle'] = [
        plotly.graph_objs.Scatter(x=result["t"], y=(180/np.pi) * result["e"][:, 0].squeeze(), name="Roll"),
        plotly.graph_objs.Scatter(x=result["t"], y=(180/np.pi) * result["e"][:, 1].squeeze(), name="Pitch"),
        plotly.graph_objs.Scatter(x=result["t"], y=(180/np.pi) * result["e"][:, 2].squeeze(), name="Yaw"),
        plotly.graph_objs.Scatter(x=result["t"], y=(180 / np.pi) * result["e_est"][:, 0].squeeze(), name="Roll est", line={'dash': 'dash'}),
        plotly.graph_objs.Scatter(x=result["t"], y=(180 / np.pi) * result["e_est"][:, 1].squeeze(), name="Pitch est", line={'dash': 'dash'}),
        plotly.graph_objs.Scatter(x=result["t"], y=(180 / np.pi) * result["e_est"][:, 2].squeeze(), name="Yaw est", line={'dash': 'dash'}),
    ]
    layout['Euler Angle'] = dict(xaxis=dict(title="Time [s]"), yaxis=dict(title="Angle [deg]"))

    data['Quaternion'] = [
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 3].squeeze(), name="Q0"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 4].squeeze(), name="Qi"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 5].squeeze(), name="Qj"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 6].squeeze(), name="Qk"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s_est"][:, 3].squeeze(), name="Q0 est", line={'dash': 'dash'}),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s_est"][:, 4].squeeze(), name="Qi est", line={'dash': 'dash'}),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s_est"][:, 5].squeeze(), name="Qj est", line={'dash': 'dash'}),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s_est"][:, 6].squeeze(), name="Qk est", line={'dash': 'dash'}),
    ]
    layout['Quaternion'] = dict(xaxis=dict(title="Time [s]"), yaxis=dict(title="Q [-]"))

    data['Velocity'] = [
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 7].squeeze(), name="vx"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 8].squeeze(), name="vy"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 9].squeeze(), name="vz"),
    ]
    layout['Velocity'] = dict(xaxis=dict(title="Time [s]"), yaxis=dict(title="Velocity [m/s]"))

    data['Angular Velocity'] = [
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 10].squeeze(), name="nx"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 11].squeeze(), name="ny"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 12].squeeze(), name="nz"),
    ]
    layout['Angular Velocity'] = dict(xaxis=dict(title="Time [s]"), yaxis=dict(title="Angular Velocity [rad/s]"))

    data['Motors'] = [
        plotly.graph_objs.Scatter(x=result["t"], y=result["u"][:, 0].squeeze(), name="Front"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["u"][:, 1].squeeze(), name="Left"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["u"][:, 2].squeeze(), name="Right"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["u"][:, 3].squeeze(), name="Rear"),
    ]
    layout['Motors'] = dict(xaxis=dict(title="Time [s]"), yaxis=dict(title="Force [N]"))

    visuals.plot(data, layout, 'Summary')

    # Sensors
    data = dict()
    layout = dict()
    data['Accelerometer'] = [
        plotly.graph_objs.Scatter(x=result["t"], y=result["g_sensor"][:, 0].squeeze(), name="gx"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["g_sensor"][:, 1].squeeze(), name="gy"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["g_sensor"][:, 2].squeeze(), name="gz"),
    ]
    layout['Accelerometer'] = dict(xaxis=dict(title="Time [s]"), yaxis=dict(title="Acceleration [m/s/s]"))

    data['Gyroscope'] = [
        plotly.graph_objs.Scatter(x=result["t"], y=result["n_sensor"][:, 0].squeeze(), name="nx"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["n_sensor"][:, 1].squeeze(), name="ny"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["n_sensor"][:, 2].squeeze(), name="nz"),
    ]
    layout['Gyroscope'] = dict(xaxis=dict(title="Time [s]"), yaxis=dict(title="Angular Velocity [rad/s]"))

    data['Magnetometer'] = [
        plotly.graph_objs.Scatter(x=result["t"], y=result["en_sensor"][:, 0].squeeze(), name="enx"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["en_sensor"][:, 1].squeeze(), name="eny"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["en_sensor"][:, 2].squeeze(), name="enz"),
    ]
    layout['Magnetometer'] = dict(xaxis=dict(title="Time [s]"), yaxis=dict(title="Vector North [-]"))

    data['Quaternion'] = [
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 3].squeeze(), name="Q0"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 4].squeeze(), name="Qi"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 5].squeeze(), name="Qj"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 6].squeeze(), name="Qk"),
    ]
    layout['Quaternion'] = dict(xaxis=dict(title="Time [s]"), yaxis=dict(title="Q [-]"))

    data['Map'] = [
        plotly.graph_objs.Scatter(x=result["s"][:, 0].squeeze(),
                                  y=result["s"][:, 1].squeeze(),
                                  mode='markers',
                                  name="Trajectory",
                                  marker=dict(size=5.,
                                              colorscale='Viridis',
                                              color=utils.scale(result["s"][:, 2].squeeze(), 0., 1.))),
    ]
    layout['Map'] = dict(xaxis=dict(title="X [m]"), yaxis=dict(title="Y [m]"))

    visuals.plot(data, layout, 'Sensors')


if __name__ == '__main__':

    # length and time step of simulations
    N_SAMPLES = 2000
    DT = 0.01
    MODE = "Roll"

    t = np.arange(0, N_SAMPLES * DT, DT)

    # z target
    tgt = np.zeros((N_SAMPLES, 4))
    tgt[:, 0] = 0. * np.ones(N_SAMPLES)
    tgt[:, 1] = 0. * np.ones(N_SAMPLES)
    tgt[:, 2] = 5. * np.ones(N_SAMPLES)

    psi = 1. * np.pi / 4.

    if MODE == "Pitch":
        axis = utils.normalize(np.array([0., 1., 0.]))
    elif MODE == "Roll":
        axis = utils.normalize(np.array([1., 0., 0.]))
    elif MODE == "Yaw":
        axis = utils.normalize(np.array([0., 0., 1.]))
    else:
        raise Exception("Unrecognised mode")

    s0 = np.zeros(13)
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

    # define input vector
    result = run_sim(t, tgt, s0)
    plot_result(result)

