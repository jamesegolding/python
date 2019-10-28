import numpy as np
import plotly

from lib import visuals
from lib import utilities as utils
from lib.parameters import *

import logging
logger = logging.getLogger("post_proc")
logger.setLevel(logging.INFO)


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
        plotly.graph_objs.Scatter(x=result["t"], y=result["s_est"][:, 7].squeeze(), name="vx est", line={'dash': 'dash'}),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s_est"][:, 8].squeeze(), name="vy est", line={'dash': 'dash'}),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s_est"][:, 9].squeeze(), name="vz est ", line={'dash': 'dash'}),
    ]
    layout['Velocity'] = dict(xaxis=dict(title="Time [s]"), yaxis=dict(title="Velocity [m/s]"))

    data['Angular Velocity'] = [
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 10].squeeze(), name="nx"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 11].squeeze(), name="ny"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 12].squeeze(), name="nz"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s_est"][:, 10].squeeze(), name="nx est", line={'dash': 'dash'}),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s_est"][:, 11].squeeze(), name="ny est", line={'dash': 'dash'}),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s_est"][:, 12].squeeze(), name="nz est ", line={'dash': 'dash'}),
    ]
    layout['Angular Velocity'] = dict(xaxis=dict(title="Time [s]"), yaxis=dict(title="Angular Velocity [rad/s]"))

    data['Motors'] = [
        plotly.graph_objs.Scatter(x=result["t"], y=result["u"][:, 0].squeeze(), name="Front Tgt"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["u"][:, 1].squeeze(), name="Left Tgt"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["u"][:, 2].squeeze(), name="Right Tgt"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["u"][:, 3].squeeze(), name="Rear Tgt"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 13].squeeze(), name="Front"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 14].squeeze(), name="Left"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 15].squeeze(), name="Right"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s"][:, 16].squeeze(), name="Rear"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s_est"][:, 13].squeeze(), name="Front est", line={'dash': 'dash'}),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s_est"][:, 14].squeeze(), name="Left est", line={'dash': 'dash'}),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s_est"][:, 15].squeeze(), name="Right est", line={'dash': 'dash'}),
        plotly.graph_objs.Scatter(x=result["t"], y=result["s_est"][:, 16].squeeze(), name="Rear est", line={'dash': 'dash'}),
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

    data['Acceleration World'] = [
        plotly.graph_objs.Scatter(x=result["t"], y=result["g"][:, 0].squeeze(), name="gx"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["g"][:, 1].squeeze(), name="gy"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["g"][:, 2].squeeze() + G, name="gz"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["g_est"][:, 0].squeeze(), name="gx est", line={'dash': 'dash'}),
        plotly.graph_objs.Scatter(x=result["t"], y=result["g_est"][:, 1].squeeze(), name="gy est", line={'dash': 'dash'}),
        plotly.graph_objs.Scatter(x=result["t"], y=result["g_est"][:, 2].squeeze() + G, name="gz est", line={'dash': 'dash'}),
    ]
    layout['Acceleration World'] = dict(xaxis=dict(title="Time [s]"), yaxis=dict(title="Acceleration [m/s/s]"))

    data['Angular Acceleration'] = [
        plotly.graph_objs.Scatter(x=result["t"], y=result["dn_body"][:, 0].squeeze(), name="dnx"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["dn_body"][:, 1].squeeze(), name="dny"),
        plotly.graph_objs.Scatter(x=result["t"], y=result["dn_body"][:, 2].squeeze(), name="dnz"),
    ]
    layout['Angular Acceleration'] = dict(xaxis=dict(title="Time [s]"), yaxis=dict(title="Angular Accel [rad/s/s]"))

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
