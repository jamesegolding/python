import numpy as np
import numba
import utilities as utils

"""
Quaternion Definitions and Transformations based on:
https://www.astro.rug.nl/software/kapteyn/_downloads/attitude.pdf
"""

@numba.jit(nopython=True)
def conjugate(q: np.ndarray):
    """
    Calculate quaternion conjugate
    :param q: quaternion
    :return: quaternion conjugate
    """

    return utils.normalize(np.multiply(q, np.array([1, -1, -1, -1])))


@numba.jit(nopython=True)
def product(a: np.ndarray, b: np.ndarray):
    """
    Calculate quaternion product
    :param a: quaternion
    :param b: quaternion
    :return: product of inputs a * b
    """

    ab = np.zeros(4)
    ab[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    ab[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    ab[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    ab[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]

    return ab


@numba.jit(nopython=True)
def rate_matrix_body(q: np.ndarray):
    """
    Calculate quaternion rate matrix (body) from quaternion
    :param q: quaternion
    :return: quaternion rate matrix in body frame
    """

    return np.array([
        [-q[1],  q[0],  q[3], -q[2]],
        [-q[2], -q[3],  q[0],  q[1]],
        [-q[3],  q[2], -q[1],  q[0]],
    ])


@numba.jit(nopython=True)
def rate_matrix_world(q: np.ndarray):
    """
    Calculate quaternion rate matrix (world) from quaternion
    :param q: quaternion
    :return: quaternion rate matrix in world frame
    """

    return np.array([
        [-q[1],  q[0], -q[3],  q[2]],
        [-q[2],  q[3],  q[0], -q[1]],
        [-q[3], -q[2],  q[1],  q[0]],
    ])


@numba.jit(nopython=True)
def to_angular_velocity(w: np.ndarray, q_dot: np.ndarray):
    """
    Calculate angular velocity (body or world) from quaternion rate
    :param w: quaternion rate matrix (body or world)
    :param q_dot: quaternion rate
    :return: angular velocity in body or world frame
    """

    return 2 * w @ q_dot


@numba.jit(nopython=True)
def from_angular_velocity(w: np.ndarray, n: np.ndarray):
    """
    Calculate quaternion rate from angular velocity (body or world)
    :param w: quaternion rate matrix (body or world)
    :param n: angular velocity in body or world frame
    :return: quaternion rate
    """

    return 0.5 * w.T @ n


@numba.jit(nopython=True)
def to_angular_acceleration(w: np.ndarray, q_dot_dot: np.ndarray):
    """
    Calculate angular acceleration (body or world) from quaternion rate
    :param w: quaternion rate matrix (body or world)
    :param q_dot_dot: quaternion rate of rate
    :return: angular acceleration in body or world frame
    """

    return 2 * w @ q_dot_dot


@numba.jit(nopython=True)
def from_angular_acceleration(w: np.ndarray, dn: np.ndarray):
    """
    Calculate quaternion rate of rate from angular velocity (body or world)
    :param w: quaternion rate matrix (body or world)
    :param dn: angular acceleration in body or world frame
    :return: quaternion rate of rate
    """

    return 0.5 * w.T @ dn


@numba.jit(nopython=True)
def to_euler(q: np.ndarray):
    """
    Calculate euler angles from quaternion
    :param q: quaternion
    :return: euler ZYX angles
    """

    r = to_rot_mat(q)

    return rot_mat_to_euler(r)


@numba.jit(nopython=True)
def to_rot_mat(q: np.ndarray):
    """
    Calculate rotation matrix from quaternion
    :param q: quaternion
    :return: rotation matrix
    """

    return np.array([
        [q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*(q[1]*q[2] + q[0]*q[3]), 2*(q[1]*q[3] - q[0]*q[2])],
        [2*(q[2]*q[1] - q[0]*q[3]), q[0]**2 + q[2]**2 - q[1]**2 - q[3]**2, 2*(q[2]*q[3] + q[0]*q[1])],
        [2*(q[3]*q[1] + q[0]*q[2]), 2*(q[3]*q[2] - q[0]*q[1]), q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2],
    ])


@numba.jit(nopython=True)
def from_rot_mat(r: np.ndarray):
    """
    Calculate quaternion from rotation matrix (ref sec 6.5)
    :param r: rotation matrix
    :return: quaternion
    """

    if (r[1, 1] > -r[2, 2]) and (r[0, 0] > -r[1, 1]) and (r[0, 0] > -r[2, 2]):
        return 0.5 * np.array([
            np.sqrt(1 + r[0, 0] + r[1, 1] + r[2, 2]),
            (r[1, 2] - r[2, 1]) / np.sqrt(1 + r[0, 0] + r[1, 1] + r[2, 2]),
            (r[2, 0] - r[0, 2]) / np.sqrt(1 + r[0, 0] + r[1, 1] + r[2, 2]),
            (r[0, 1] - r[1, 0]) / np.sqrt(1 + r[0, 0] + r[1, 1] + r[2, 2]),
        ])
    elif (r[1, 1] < -r[2, 2]) and (r[0, 0] > r[1, 1]) and (r[0, 0] > r[2, 2]):
        return 0.5 * np.array([
            (r[1, 2] - r[2, 1]) / np.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2]),
            np.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2]),
            (r[0, 1] + r[1, 0]) / np.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2]),
            (r[2, 0] + r[0, 2]) / np.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2]),
        ])
    elif (r[1, 1] > r[2, 2]) and (r[0, 0] < r[1, 1]) and (r[0, 0] < -r[2, 2]):
        return 0.5 * np.array([
            (r[2, 0] - r[0, 2]) / np.sqrt(1 - r[0, 0] + r[1, 1] - r[2, 2]),
            (r[0, 1] + r[1, 0]) / np.sqrt(1 - r[0, 0] + r[1, 1] - r[2, 2]),
            np.sqrt(1 - r[0, 0] + r[1, 1] - r[2, 2]),
            (r[1, 2] + r[2, 1]) / np.sqrt(1 - r[0, 0] + r[1, 1] - r[2, 2]),
        ])
    elif (r[1, 1] < r[2, 2]) and (r[0, 0] < -r[1, 1]) and (r[0, 0] < r[2, 2]):
        return 0.5 * np.array([
            (r[0, 1] - r[1, 0]) / np.sqrt(1 - r[0, 0] - r[1, 1] + r[2, 2]),
            (r[2, 0] + r[0, 2]) / np.sqrt(1 - r[0, 0] - r[1, 1] + r[2, 2]),
            (r[1, 2] + r[2, 1]) / np.sqrt(1 - r[0, 0] - r[1, 1] + r[2, 2]),
            np.sqrt(1 - r[0, 0] - r[1, 1] + r[2, 2]),
        ])
    else:
        raise Exception(f"Unexpected rotation matrix form\n{r}")


@numba.jit(nopython=True)
def rot_mat_to_euler(r: np.ndarray):
    """
    Calculate euler angles from rotation matrix
    :param r: rotation matrix
    :return: euler ZYX angles
    """

    phi = np.arctan2(r[2, 1], r[2, 2])
    theta = np.arctan2(-r[2, 0], np.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))
    psi = np.arctan2(r[1, 0], r[0, 0])

    return np.array([phi, theta, psi])


@numba.jit(nopython=True)
def euler_to_rot_mat(e: np.ndarray):
    """
    Calculate rotation matrix from euler angles
    :param e: euler ZYX angles
    :return: rotation matrix
    """

    c0 = np.cos(e[0])
    c1 = np.cos(e[1])
    c2 = np.cos(e[2])
    s0 = np.sin(e[0])
    s1 = np.sin(e[1])
    s2 = np.sin(e[2])

    return np.array([
        [c2 * c1,  c2 * s1 * s0 - s2 * c0,  c2 * s1 * c0 + s2 * s0],
        [s2 * c1,  s2 * s1 * s0 + c2 * c0,  s2 * s1 * c0 - c2 * s0],
        [-s1,      c1 * s0,                 c1 * c0],
    ])


@numba.jit(nopython=True)
def euler_fix_quadrant(e: np.ndarray):
    """
    Convert angle into -pi to +pi quadrant
    :param e: euler ZYX angles
    :return: euler ZYX angles within range -pi to pi
    """

    e_fix = e % (2 * np.pi)
    gt_pi = 0.5 * (1 + np.sign(e_fix - np.pi))
    e_fix = e_fix - 2 * np.pi * gt_pi

    return e_fix


def test():

    for i in range(100):

        e_0 = euler_fix_quadrant(2 * np.pi * np.random.rand(3))
        n_0 = np.random.rand(3)
        dn_0 = np.random.rand(3)

        r_1 = euler_to_rot_mat(e_0)
        e_1 = rot_mat_to_euler(r_1)

        assert np.isclose(e_0, e_1).all(), f"({i}) Euler angles to not match\n{e_0}\n{e_1}"

        q_1 = from_rot_mat(r_1)
        r_2 = to_rot_mat(q_1)

        assert np.isclose(r_1, r_2).all(), f"({i}) Rotation matrices to not match\n{r_1}\n{r_2}"

        w_b_1 = rate_matrix_body(q_1)
        o_b_1 = from_angular_velocity(w_b_1, n_0)
        n_1 = to_angular_velocity(w_b_1, o_b_1)

        assert np.isclose(n_0, n_1).all(), f"({i}) Angular velocities (body conversion) to not match\n{n_0}\n{n_1}"

        w_w_1 = rate_matrix_world(q_1)
        o_w_1 = from_angular_velocity(w_w_1, n_0)
        n_2 = to_angular_velocity(w_w_1, o_w_1)

        assert np.isclose(n_0, n_2).all(), f"({i}) Angular velocities (world conversion) to not match\n{n_0}\n{n_2}"

        l_b_1 = from_angular_acceleration(w_b_1, dn_0)
        dn_1 = to_angular_acceleration(w_b_1, l_b_1)

        assert np.isclose(dn_0, dn_1).all(), f"({i}) Angular accelerations (body conversion) to not match\n{dn_0}\n{dn_1}"

        l_w_1 = from_angular_acceleration(w_w_1, dn_0)
        dn_2 = to_angular_acceleration(w_w_1, l_w_1)

        assert np.isclose(dn_0, dn_2).all(), f"({i}) Angular accelerations (world conversion) to not match\n{dn_0}\n{dn_2}"


