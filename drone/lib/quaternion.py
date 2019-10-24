import numpy as np
import numba
from lib import utilities as utils

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
def matrix(q: np.ndarray):
    """
    Calculate quaternion matrix from quaternion
    :param q: quaternion
    :return: quaternion matrix Q
    """

    return np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1],  q[0],  q[3], -q[2]],
        [q[2], -q[3],  q[0],  q[1]],
        [q[3],  q[2], -q[1],  q[0]],
    ])


@numba.jit(nopython=True)
def conjugate_matrix(q: np.ndarray):
    """
    Calculate quaternion matrix from quaternion
    :param q: quaternion
    :return: conjugate quaternion matrix Q_hat
    """

    return np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1],  q[0], -q[3],  q[2]],
        [q[2],  q[3],  q[0], -q[1]],
        [q[3], -q[2],  q[1],  q[0]],
    ])


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
def transform(v: np.ndarray, q: np.ndarray, q_conj: np.ndarray = None):
    """
    Calculate quaternion conjugate
    :param v: vector
    :param q: quaternion
    :param q_conj: optional conjugate quaternion
    :return: vector transformed by quaternion
    """
    q = utils.normalize(q)

    if q_conj is None:
        q_conj = conjugate(q)

    v_pad = np.concatenate((np.array([0.]), v))

    return product(q, product(v_pad, q_conj))[1:]


@numba.jit(nopython=True)
def transform_inv(v: np.ndarray, q: np.ndarray):
    """
    Calculate quaternion conjugate
    :param v: vector
    :param q: quaternion
    :return: vector inverse-transformed by quaternion
    """
    q = utils.normalize(q)
    q_conj = conjugate(q)

    return transform(v, q_conj, q)


@numba.jit(nopython=True)
def integrate(q: np.ndarray, n_body: np.ndarray, dt: float):
    """
    Calculate integrate single time step unit quaternion
    :param q: quaternion
    :param n_body: rotational velocity in body frame
    :param dt: time step
    :return: quaternion at t+1
    """

    q = utils.normalize(q)
    n_body_norm = utils.norm2(n_body)

    q_hat = np.concatenate((
        np.array([np.cos(0.5 * n_body_norm * dt)]),
        np.sin(0.5 * n_body_norm * dt) * utils.normalize(n_body),
    ))

    q = product(q_hat, q)

    return q


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
def to_yaw(q: np.ndarray):

    r10 = 2*(q[2]*q[1] - q[0]*q[3])
    r00 = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    psi = np.arctan2(r10, r00)

    return psi


@numba.jit(nopython=True)
def to_rot_mat(q: np.ndarray):
    """
    Calculate rotation matrix from quaternion
    :param q: quaternion
    :return: rotation matrix
    """
    q = utils.normalize(q)

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
        return np.nan * np.ones(4)



@numba.jit(nopython=True)
def to_axis_angle(q: np.ndarray):
    """
    Calculate axis rotation by angle from quaternion
    :param q: quaternion
    :return: axis
    :return: angle
    """

    q = utils.normalize(q)
    theta = 2 * np.arccos(q[0])
    axis = utils.normalize(q[1:])

    return axis, theta


@numba.jit(nopython=True)
def from_axis_angle(axis: np.ndarray, theta: float):
    """
    Calculate quaternion from rotation by angle about axis
    :param axis: axis
    :param theta: angle
    :return: quaternion
    """

    return utils.normalize(np.concatenate((np.array([np.cos(theta / 2.)]),
                                           np.sin(theta / 2.) * axis)))


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


@numba.jit(nopython=True)
def rot_mat_der(q: np.ndarray, i_q: int = 0, b_inverse: bool = False):
    """
    Return the derivative of a rotation matrix wrt given element
    :param q: quaternion
    :param i_q: derivative with respect to element i_q
    :param b_inverse: boolean true to return derivative of inverse rotation matrix
    :return: r_der_qi derivative of rotation matrix wrt i_q
    """

    q = utils.normalize(q)
    u_dv_dt = -2 * q[i_q] * to_rot_mat(q)

    if i_q == 0:
        v_du_dt = 2 * np.array([
            [q[0],  q[3], -q[2]],
            [-q[3], q[0],  q[1]],
            [q[2], -q[1],  q[0]],
        ])
    elif i_q == 1:
        v_du_dt = 2 * np.array([
            [q[1],  q[2],  q[3]],
            [q[2], -q[1],  q[0]],
            [q[3], -q[0], -q[1]],
        ])
    elif i_q == 2:
        v_du_dt = 2 * np.array([
            [-q[2], q[1], -q[0]],
            [q[1],  q[2],  q[3]],
            [q[0],  q[3], -q[2]],
        ])
    elif i_q == 3:
        v_du_dt = 2 * np.array([
            [-q[3],  q[0], q[1]],
            [-q[0], -q[3], q[2]],
            [q[1],   q[2], q[3]],
        ])
    else:
        return np.nan * np.ones((3, 3))

    dr_dqi = u_dv_dt + v_du_dt
    if b_inverse:
        return dr_dqi.T
    else:
        return dr_dqi
