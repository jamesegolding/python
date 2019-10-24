
import unittest
import numpy as np
from lib import sensor
from lib import quaternion

class SensorTest(unittest.TestCase):

    def test_kalman_h_matrix(self):
        """
        Sensor:      Check Orientation Kalman H matrix derivative matches the numeric version
        """
        a_0 = quaternion.euler_fix_quadrant(2 * np.pi * np.random.rand())
        e_0 = np.random.rand(3)
        n_0 = np.random.rand(3)

        s_reduce = np.concatenate((quaternion.from_axis_angle(e_0, a_0), n_0))
        s = np.concatenate((
            np.zeros(3), s_reduce[:4], np.zeros(3), s_reduce[4:], np.zeros(4),
        ))

        # compute algebraic h matrix
        h_alg = sensor.kalman_orientation_h(s)

        # compute a finite difference H matrix
        eps = 0.0001
        h_num = np.zeros((9, 7))
        for i_s, i_reduce in zip([3, 4, 5, 6, 10, 11, 12], range(7)):
            ds = np.zeros(17)
            ds[i_s] = eps
            sensor_state_p = sensor.from_state(s + ds, g=np.zeros(3), r_scale=0.)
            sensor_state_n = sensor.from_state(s - ds, g=np.zeros(3), r_scale=0.)
            h_num[:, i_reduce] = np.concatenate((
                (sensor_state_p.accelerometer - sensor_state_n.accelerometer) / 2 / eps,
                (sensor_state_p.gyroscope - sensor_state_n.gyroscope) / 2 / eps,
                (sensor_state_p.magnetometer - sensor_state_n.magnetometer) / 2 / eps,
            ))

        self.assertTrue(np.isclose(h_alg, h_num, rtol=1e-3).all(), f"Error in h derivative\n{h_num}\n{h_alg}")


