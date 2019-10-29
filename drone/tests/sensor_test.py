
import unittest
import numpy as np
from lib import sensor
from lib import quaternion
from lib import drone

class SensorTest(unittest.TestCase):

    def test_kalman_h_matrix(self):
        """
        Sensor:      Check Orientation Kalman H matrix derivative matches the numeric version
        """
        a_theta = quaternion.euler_fix_quadrant(2 * np.pi * np.random.rand())
        e_axis = np.random.rand(3)
        n_body = np.random.rand(3)
        q = quaternion.from_axis_angle(e_axis, a_theta)

        s = np.concatenate((
            np.zeros(3), q, np.zeros(3), n_body, np.zeros(4),
        ))

        h_alg = sensor.kalman_orientation_h(s)
        h_num = sensor.kalman_orientation_h_finite_difference(s)

        self.assertTrue(np.isclose(h_alg, h_num, rtol=1e-3).all(), f"Error in h derivative\n{h_num}\n{h_alg}")

    def test_kalman_f_matrix(self):
        """
        Sensor:      Check Orientation Kalman F matrix derivative matches the numeric version
        """
        for i in range(10):
            a_theta = quaternion.euler_fix_quadrant(2 * np.pi * np.random.rand())
            e_axis = np.random.rand(3)
            n_body = np.random.rand(3)
            q = quaternion.from_axis_angle(e_axis, a_theta)
            u = 5 * np.random.rand(4)
            dt = 0.001

            s = np.concatenate((
                np.zeros(3), q, np.zeros(3), n_body, np.ones(4),
            ))

            f_alg = sensor.kalman_orientation_f(s, dt)
            f_num = sensor.kalman_orientation_f_finite_difference(s, u, dt)

            self.assertTrue(np.isclose(f_alg, f_num, atol=5e-3).all(), f"Error in f derivative\n{f_num}\n{f_alg}")
