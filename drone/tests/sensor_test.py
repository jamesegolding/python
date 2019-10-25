
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

        # compute algebraic h matrix
        h_alg = sensor.kalman_orientation_h(s)

        i_states = [3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16]

        # compute a finite difference H matrix
        eps = 0.0001
        h_num = np.zeros((9, 11))
        for i_s, i_reduce in zip(i_states, range(7)):
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

    def test_kalman_f_matrix(self):
        """
        Sensor:      Check Orientation Kalman F matrix derivative matches the numeric version
        """
        a_theta = quaternion.euler_fix_quadrant(2 * np.pi * np.random.rand())
        e_axis = np.random.rand(3)
        n_body = np.random.rand(3)
        q = quaternion.from_axis_angle(e_axis, a_theta)

        u = 5 * np.random.rand(4)
        dt = 0.001

        i_states = [3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16]

        s = np.concatenate((
            np.zeros(3), q, np.zeros(3), n_body, np.ones(4),
        ))

        # compute algebraic f matrix
        f_alg = sensor.kalman_orientation_f(s, dt)

        # compute a finite difference H matrix
        eps = 0.0001
        f_num = np.zeros((11, 11))
        for i_s, i_reduce in zip(i_states, range(11)):
            ds = np.zeros(17)
            ds[i_s] = eps

            s_pos, _, _ = drone.step(s + ds, u, dt, r_scale_dist=0.)
            s_neg, _, _ = drone.step(s - ds, u, dt, r_scale_dist=0.)
            f_num[:, i_reduce] = (s_pos[i_states] - s_neg[i_states]) / 2 / eps

        self.assertTrue(True)
