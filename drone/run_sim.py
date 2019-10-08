import numpy as np
import drone
import plotly
import quaternion
import control
import utilities as utils

import logging
logger = logging.getLogger("run_sim")
logger.setLevel(logging.INFO)


def run_sim(target: np.ndarray,
            dt: float,
            s0: np.ndarray,
            r_disturb: float = 0.,
            ):

    # initialize vectors
    s = np.empty((target.shape[0], 14))
    s[:] = np.nan
    e = np.empty((target.shape[0], 3))
    e[:] = np.nan
    u = np.empty((target.shape[0], 4))
    u[:] = np.nan

    # define initial conditions
    s[0, :] = s0

    motor_inv = drone.torque_motor_inv()
    a, b, q, r, u_0 = drone.vertical_state_space()
    k = control.lqr_gain(a, b, q, r)

    # run sim
    for i in range(1, target.shape[0]):
        u[i, :] = control.update(s[ i -1, :], target[i, :], k, motor_inv, u_0)
        s[i, :] = drone.step(s[ i -1, :], u[i, :], dt, r_disturb)
        e[i, :] = quaternion.to_euler(s[i, 3:7])

    return s, e, u


def plot_result(t, s, e, u, title):

    data_1 = [
        plotly.graph_objs.Scatter(x=t, y=s[:, 0].squeeze(), name="X"),
        plotly.graph_objs.Scatter(x=t, y=s[:, 1].squeeze(), name="Y"),
        plotly.graph_objs.Scatter(x=t, y=s[:, 2].squeeze(), name="Z"),
    ]

    data_2 = [
        plotly.graph_objs.Scatter(x=t, y=e[:, 0].squeeze(), name="Roll"),
        plotly.graph_objs.Scatter(x=t, y=e[:, 1].squeeze(), name="Pitch"),
        plotly.graph_objs.Scatter(x=t, y=e[:, 2].squeeze(), name="Yaw"),
    ]

    data_3 = [
        plotly.graph_objs.Scatter(x=t, y=s[:, 3].squeeze(), name="Q0"),
        plotly.graph_objs.Scatter(x=t, y=s[:, 4].squeeze(), name="Qi"),
        plotly.graph_objs.Scatter(x=t, y=s[:, 5].squeeze(), name="Qj"),
        plotly.graph_objs.Scatter(x=t, y=s[:, 6].squeeze(), name="Qk"),
    ]

    data_4 = [
        plotly.graph_objs.Scatter(x=t, y=s[:, 7].squeeze(), name="vx"),
        plotly.graph_objs.Scatter(x=t, y=s[:, 8].squeeze(), name="vy"),
        plotly.graph_objs.Scatter(x=t, y=s[:, 9].squeeze(), name="vz"),
    ]

    data_5 = [
        plotly.graph_objs.Scatter(x=t, y=u[:, 0].squeeze(), name="Front"),
        plotly.graph_objs.Scatter(x=t, y=u[:, 1].squeeze(), name="Left"),
        plotly.graph_objs.Scatter(x=t, y=u[:, 2].squeeze(), name="Right"),
        plotly.graph_objs.Scatter(x=t, y=u[:, 3].squeeze(), name="Rear"),
    ]

    data_6 = [
        plotly.graph_objs.Scatter(x=s[:, 0].squeeze(),
                                  y=s[:, 1].squeeze(),
                                  mode='markers',
                                  name="Trajectory",
                                  marker=dict(size=5.,
                                              colorscale='Viridis',
                                              color=utils.scale(s[:, 2].squeeze(), 0., 1.))),
    ]

    fig = plotly.subplots.make_subplots(
        rows=2,
        cols=3,
        subplot_titles=("Position", "Rotation", "Quaternions", "Velocity", "Motors", "Map"),
    )

    for d in data_1:
        fig.add_trace(d, row=1, col=1)
        fig.update_xaxes(title_text="Time [s]", row=1, col=1)
        fig.update_yaxes(title_text="Position [m]", row=1, col=1)
    for d in data_2:
        fig.add_trace(d, row=1, col=2)
        fig.update_xaxes(title_text="Time [s]", row=1, col=2)
        fig.update_yaxes(title_text="Angle [rad]", row=1, col=2)
    for d in data_3:
        fig.add_trace(d, row=1, col=3)
        fig.update_xaxes(title_text="Time [s]", row=1, col=3)
        fig.update_yaxes(title_text="Quaternion [-]", row=1, col=3)
    for d in data_4:
        fig.add_trace(d, row=2, col=1)
        fig.update_xaxes(title_text="Time [s]", row=2, col=1)
        fig.update_yaxes(title_text="Speed [m/s]", row=2, col=1)
    for d in data_5:
        fig.add_trace(d, row=2, col=2)
        fig.update_xaxes(title_text="Time [s]", row=2, col=2)
        fig.update_yaxes(title_text="Force [N]", row=2, col=2)
    for d in data_6:
        fig.add_trace(d, row=2, col=3)
        fig.update_xaxes(title_text="Position [m]", row=2, col=3)
        fig.update_yaxes(title_text="Position [m]", row=2, col=3)

    plotly.offline.plot(fig, filename=title + ".html")


if __name__ == '__main__':

    # length and time step of simulations
    N_SAMPLES = 2000
    DT = 0.01

    t = np.arange(0, N_SAMPLES * DT, DT)

    # z target
    tgt = np.zeros((N_SAMPLES, 4))

    tgt[:, 0] = 0. * np.ones(N_SAMPLES)
    tgt[:, 1] = 0. * np.ones(N_SAMPLES)
    tgt[:, 2] = 5. * np.ones(N_SAMPLES)

    s0 = np.zeros(14)
    s0[drone.State.x.value] = 10.
    s0[drone.State.y.value] = 3.
    s0[drone.State.z.value] = 5.
    s0[drone.State.q_0.value] = np.cos(np.pi / 2. / 2.)
    s0[drone.State.q_i.value] = 0.
    s0[drone.State.q_j.value] = 0.
    s0[drone.State.q_k.value] = np.sin(np.pi / 2. / 2.)
    s0[drone.State.vx.value] = 7.
    s0[drone.State.vy.value] = -4.
    s0[drone.State.vz.value] = 0.

    # define input vector
    s_height, e_height, u_height = run_sim(tgt, DT, s0, r_disturb=0.8)
    plot_result(t, s_height, e_height, u_height, "height_test")

