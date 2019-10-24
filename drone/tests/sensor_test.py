
import numpy as np
from lib import sensor
from lib import quaternion
from lib import utilities as utils


def test_kalman_h_matrix(self):

    a_0 = quaternion.euler_fix_quadrant(2 * np.pi * np.random.rand())
    e_0 = np.random.rand(3)
    n_0 = np.random.rand(3)

    s_reduce = np.concatenate((quaternion.from_axis_angle(e_0, a_0), n_0))
    s = np.concatenate((
        np.zeros(3), s_reduce[:4], np.zeros(3), s_reduce[4:], np.zeros(4),
    ))

    h = sensor.kalman_orientation_h(s)

    eps = 0.001
    h_numeric = np.zeros((9, 7))
    for i_s, i_reduce in zip([3, 4, 5, 6, 10, 11, 12], range(7)):
        ds = np.zeros(17)
        ds[i_s] = eps
        sensor_state_p = sensor.from_state(s + ds, g=np.zeros(3), r_scale=0.)
        sensor_state_n = sensor.from_state(s - ds, g=np.zeros(3), r_scale=0.)
        h_numeric[:, i_reduce] = np.concatenate((
            (sensor_state_p.accelerometer - sensor_state_n.accelerometer) / 2 / eps,
            (sensor_state_p.gyroscope - sensor_state_n.gyroscope) / 2 / eps,
            (sensor_state_p.magnetometer - sensor_state_n.magnetometer) / 2 / eps,
        ))

    with np.printoptions(precision=4, suppress=True):
        print(h)
        print("---")
        print(h_numeric)

    return h, h_numeric

