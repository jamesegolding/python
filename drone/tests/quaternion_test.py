
import unittest
import numpy as np
from lib import quaternion


class QuaternionTest(unittest.TestCase):

    def test_baseline(self):
        """
        Quaternions: Compare transformations to baseline
        """
        x = np.array([0., 1., 0.])
        angles = np.array([np.pi / 2., np.pi / 2., np.pi / 2., np.pi / 8., np.pi / 8., np.pi / 8.])
        i_axis = np.array([0, 1, 2, 2, 1, 0])

        x_baseline = np.array([
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
            [-0.382683, 0.923880, 0.],
            [-0.353553, 0.923880, 0.146447],
            [-0.353553, 0.797511, 0.488852],
        ])

        for i, a, i_axis in zip(range(0, len(angles)), angles, i_axis):
            # rotate by angle a about axis i
            q = np.concatenate((np.array([np.cos(a / 2.)]), np.sin(a / 2.) * np.eye(3)[i_axis]))
            x = quaternion.transform(x, q)
            self.assertTrue(np.isclose(x, x_baseline[i, :]).all(), f"Transform vs baseline\n{x}\n{x_baseline[i, :]}")

    def test_inverses(self):
        """
        Quaternions: Check inverse rotation matrix conjugate
        """
        for i in range(10):
            a_0 = quaternion.euler_fix_quadrant(2 * np.pi * np.random.rand())
            e_0 = np.random.rand(3)

            # make a non unit quaternion
            q = quaternion.from_axis_angle(e_0, a_0) + np.array([0., 0., 0.5, 0.])

            q_conj = quaternion.conjugate(q)
            r = quaternion.to_rot_mat(q)

            r_inv = np.linalg.inv(r)
            r_conj = quaternion.to_rot_mat(q_conj)
            self.assertTrue(np.isclose(r_conj, r_inv).all(), f"Conjugate and inverse mismatch\n{r_inv}\n{r_conj}")

    def test_consistency(self):
        """
        Quaternions: Convert to and from quaternions and check consistency
        """
        for i in range(100):
            a_0 = quaternion.euler_fix_quadrant(2 * np.pi * np.random.rand(3))
            n_0 = np.random.rand(3)
            dn_0 = np.random.rand(3)
            v_0 = np.random.rand(3)

            # check that rotation matrix conversion is internally consistent
            r_1 = quaternion.euler_to_rot_mat(a_0)
            q_1 = quaternion.from_rot_mat(r_1)
            r_2 = quaternion.to_rot_mat(q_1)
            self.assertTrue(np.isclose(r_1, r_2).all(), f"Rotation matrices mismatch\n{r_1}\n{r_2}")

            # check that transform is correct
            v_0_t = quaternion.transform(v_0, q_1)
            v_1 = quaternion.transform_inv(v_0_t, q_1)
            v_2 = quaternion.transform(v_0_t, quaternion.conjugate(q_1))
            self.assertTrue(np.isclose(v_0, v_1).all(), f"Transform mismatch\n{v_0}\n{v_1}")
            self.assertTrue(np.isclose(v_0, v_2).all(), f"Conjugate transform mismatch\n{v_0}\n{v_2}")

            w_b_1 = quaternion.rate_matrix_body(q_1)
            o_b_1 = quaternion.from_angular_velocity(w_b_1, n_0)
            n_1 = quaternion.to_angular_velocity(w_b_1, o_b_1)
            self.assertTrue(np.isclose(n_0, n_1).all(), f"Angular velocity mismatch\n{n_0}\n{n_1}")

            w_w_1 = quaternion.rate_matrix_world(q_1)
            o_w_1 = quaternion.from_angular_velocity(w_w_1, n_0)
            n_2 = quaternion.to_angular_velocity(w_w_1, o_w_1)
            self.assertTrue(np.isclose(n_0, n_2).all(), f"Angular velocity mismatch\n{n_0}\n{n_2}")

            l_b_1 = quaternion.from_angular_acceleration(w_b_1, dn_0)
            dn_1 = quaternion.to_angular_acceleration(w_b_1, l_b_1)
            self.assertTrue(np.isclose(dn_0, dn_1).all(), f"Angular acceleration mismatch\n{dn_0}\n{dn_1}")

            l_w_1 = quaternion.from_angular_acceleration(w_w_1, dn_0)
            dn_2 = quaternion.to_angular_acceleration(w_w_1, l_w_1)
            self.assertTrue(np.isclose(dn_0, dn_2).all(), f"Angular acceleration mismatch\n{dn_0}\n{dn_2}")

    def test_rot_mat_der(self):
        """
        Quaternions: Check the rotation matrix derivative matches the numeric version
        """

        for i in range(10):
            a_0 = quaternion.euler_fix_quadrant(2 * np.pi * np.random.rand())
            e_0 = np.random.rand(3)

            # make a non unit quaternion
            q = quaternion.from_axis_angle(e_0, a_0)

            eps = 0.0001
            for i_q in range(4):
                dq = np.zeros(4)
                dq[i_q] = eps
                r_pos = quaternion.to_rot_mat(quaternion.conjugate(q + dq))
                r_neg = quaternion.to_rot_mat(quaternion.conjugate(q - dq))
                r_num = (r_pos - r_neg) / 2 / eps

                # get algebraic solution
                r_alg = quaternion.rot_mat_der(q, i_q, b_inverse=True)
                self.assertTrue(np.isclose(r_num, r_alg, rtol=1e-3).all(), f"Error in rot mat derivative\n{r_num}\n{r_alg}")

    def test_integrate_der(self):
        """
        Quaternions: Check the integration derivative matches the numeric version
        """

        for i in range(10):
            a_0 = quaternion.euler_fix_quadrant(2 * np.pi * np.random.rand())
            e_0 = np.random.rand(3)
            n_body = np.random.rand(3)
            q = quaternion.from_axis_angle(e_0, a_0)
            dt = 0.001

            # compute analytical derivative
            q_der_q_alg, q_der_n_alg = quaternion.integrate_der(q, n_body, dt)

            eps = 0.00001
            q_der_q_num = np.zeros((4, 4))
            for i_q in range(4):
                dq = np.zeros(4)
                dq[i_q] = eps
                q_p = quaternion.integrate(q + dq, n_body, dt)
                q_n = quaternion.integrate(q - dq, n_body, dt)
                q_der_q_num[:, i_q] = (q_p - q_n) / 2 / eps

            q_der_n_num = np.zeros((4, 3))
            for i_n in range(3):
                dn = np.zeros(3)
                dn[i_n] = eps
                q_p = quaternion.integrate(q, n_body + dn, dt)
                q_n = quaternion.integrate(q, n_body - dn, dt)
                q_der_n_num[:, i_n] = (q_p - q_n) / 2 / eps

            self.assertTrue(np.isclose(q_der_q_num, q_der_q_alg, atol=5e-3).all(), f"Error in integral derivative wrt q"
                                                                                   f"\n{q_der_q_num}\n{q_der_q_alg}")
            self.assertTrue(np.isclose(q_der_n_num, q_der_n_alg, atol=5e-3).all(), f"Error in integral derivative wrt n"
                                                                                   f"\n{q_der_n_num}\n{q_der_n_alg}")
