import numpy as np
import drone
import plotly
import quaternion
import control

import logging
logger = logging.getLogger("run_sim")
logger.setLevel(logging.INFO)


def run_sim(target: np.ndarray,
            dt: float,
            z0: float,
            v_x0: float = 0.,
            v_y0: float = 0.,
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
    s[0, :] = np.zeros(14)
    s[0, 2] = z0
    s[0, 3] = 1.
    s[0, 7] = v_x0
    s[0, 8] = v_y0

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
        plotly.graph_objs.Scatter(x=t, y=s[:, 3].squeeze(), name="Q_r"),
        plotly.graph_objs.Scatter(x=t, y=s[:, 4].squeeze(), name="Q_i"),
        plotly.graph_objs.Scatter(x=t, y=s[:, 5].squeeze(), name="Q_j"),
        plotly.graph_objs.Scatter(x=t, y=s[:, 6].squeeze(), name="Q_k"),
    ]

    data_4 = [
        plotly.graph_objs.Scatter(x=t, y=s[:, 7].squeeze(), name="v_x"),
        plotly.graph_objs.Scatter(x=t, y=s[:, 8].squeeze(), name="v_y"),
        plotly.graph_objs.Scatter(x=t, y=s[:, 9].squeeze(), name="v_z"),
    ]

    data_5 = [
        plotly.graph_objs.Scatter(x=t, y=u[:, 0].squeeze(), name="F_Front"),
        plotly.graph_objs.Scatter(x=t, y=u[:, 1].squeeze(), name="F_Left"),
        plotly.graph_objs.Scatter(x=t, y=u[:, 2].squeeze(), name="F_Right"),
        plotly.graph_objs.Scatter(x=t, y=u[:, 3].squeeze(), name="F_Rear"),
    ]

    fig = plotly.subplots.make_subplots(
        rows=2,
        cols=3,
        subplot_titles=("Position", "Rotation", "Quaternions", "Velocity", "Motors"),
    )

    for d in data_1:
        fig.add_trace(d, row=1, col=1)
        fig.update_xaxes(title_text="Time [s]", row=1, col=1)
    for d in data_2:
        fig.add_trace(d, row=1, col=2)
        fig.update_xaxes(title_text="Time [s]", row=1, col=2)
    for d in data_3:
        fig.add_trace(d, row=1, col=3)
        fig.update_xaxes(title_text="Time [s]", row=1, col=3)
    for d in data_4:
        fig.add_trace(d, row=2, col=1)
        fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    for d in data_5:
        fig.add_trace(d, row=2, col=2)
        fig.update_xaxes(title_text="Time [s]", row=2, col=2)

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
    tgt[:, 3] = 0. * np.ones(N_SAMPLES)

    # define input vector
    s_height, e_height, u_height = run_sim(tgt, DT, v_x0=10., z0=5., r_disturb=1.0)
    plot_result(t, s_height, e_height, u_height, "height_test")

