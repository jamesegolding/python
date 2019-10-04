import numpy as np
import numba

"""
Quaternion Definitions and Transformations based on:
https://www.astro.rug.nl/software/kapteyn/_downloads/attitude.pdf
"""

EPS = np.finfo(float).eps


@numba.jit(nopython=True)
def normalize(q: np.ndarray):
    """
    Calculate normalized quaternion
    :param q: quaternion
    :return: normalized quaternion
    """

    q_norm = np.sqrt(np.sum(np.multiply(q, q)))

    if q_norm < EPS:
        return q

    return q / q_norm


@numba.jit(nopython=True)
def conjugate(q: np.ndarray):
    """
    Calculate quaternion conjugate
    :param q: quaternion
    :return: quaternion conjugate
    """

    return np.multiply(q, np.array([1, -1, -1, -1]))


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
def to_angular_velocity_body(w_body: np.ndarray, q_dot: np.ndarray):
    """
    Calculate angular velocity (body) from quaternion rate
    :param w_body: quaternion rate matrix (body)
    :param q_dot: quaternion rate
    :return: angular velocity in body frame
    """

    return 2 * w_body @ q_dot


@numba.jit(nopython=True)
def to_angular_velocity_world(w_world: np.ndarray, q_dot: np.ndarray):
    """
    Calculate angular velocity (world) from quaternion rate
    :param w_world: quaternion rate matrix (world)
    :param q_dot: quaternion rate
    :return: angular velocity in world frame
    """

    return 2 * w_world @ q_dot


@numba.jit(nopython=True)
def from_angular_velocity_body(w_body: np.ndarray, n_body: np.ndarray):
    """
    Calculate quaternion rate from angular velocity (body)
    :param w_body: quaternion rate matrix (body)
    :param n_body: angular velocity in body frame
    :return: quaternion rate
    """

    return 0.5 * w_body.T @ n_body


@numba.jit(nopython=True)
def from_angular_velocity_world(w_world: np.ndarray, n_world: np.ndarray):
    """
    Calculate quaternion rate from angular velocity (world)
    :param w_world: quaternion rate matrix (world)
    :param n_world: angular velocity in world frame
    :return: quaternion rate
    """

    return 0.5 * w_world.T @ n_world


@numba.jit(nopython=True)
def to_angular_acceleration_body(w_body: np.ndarray, q_dot_dot: np.ndarray):
    """
    Calculate angular acceleration (body) from quaternion rate
    :param w_body: quaternion rate matrix (body)
    :param q_dot_dot: quaternion rate of rate
    :return: angular acceleration in body frame
    """

    return 2 * w_body @ q_dot_dot


@numba.jit(nopython=True)
def to_angular_acceleration_world(w_world: np.ndarray, q_dot_dot: np.ndarray):
    """
    Calculate angular acceleration (world) from quaternion rate
    :param w_world: quaternion rate matrix (world)
    :param q_dot_dot: quaternion rate of rate
    :return: angular acceleration in world frame
    """

    return 2 * w_world @ q_dot_dot


@numba.jit(nopython=True)
def from_angular_acceleration_body(w_body: np.ndarray, dn_body: np.ndarray):
    """
    Calculate quaternion rate of rate from angular velocity (body)
    :param w_body: quaternion rate matrix (body)
    :param dn_body: angular acceleration in body frame
    :return: quaternion rate of rate
    """

    return 0.5 * w_body.T @ dn_body


@numba.jit(nopython=True)
def from_angular_acceleration_world(w_world: np.ndarray, dn_world: np.ndarray):
    """
    Calculate quaternion rate of rate from angular velocity (world)
    :param w_world: quaternion rate matrix (world) transposed
    :param dn_world: angular acceleration in world frame
    :return: quaternion rate of rate
    """

    return 0.5 * w_world.T @ dn_world


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

    phi = np.arctan(r[2, 1] / r[2, 2])
    theta = -np.arctan2(r[2, 0], np.sqrt(1 - r[2, 0] ** 2))
    psi = np.arctan(r[1, 0] / r[0, 0])

    e = np.array([phi, theta, psi])

    return e
