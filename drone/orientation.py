import numpy as np
import numba


@numba.jit(nopython=True)
def quaternion_prod(a: np.ndarray, b: np.ndarray):
    """
    quaternion product
    :param a: numpy array quaternion (4, )
    :param b: numpy array quaternion (4, )
    :return: a*b                      (4, )
    """

    ab = np.zeros(4)
    ab[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    ab[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    ab[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    ab[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]

    return ab


@numba.jit(nopython=True)
def quaternion_conj(q: np.ndarray):
    """
    return quaternion conjugate
    :param q: numpy array quaternion (4, )
    :return: conjugate of q           (4, )
    """

    q_conj = np.multiply(q, np.array([1, -1, -1, -1]))

    return q_conj


@numba.jit(nopython=True)
def quaternion_to_rot_mat(q: np.ndarray):
    """
    convert quaternion to rotation matrix
    :param q: numpy array quaternion    (4, )
    :return: numpy array rotation matrix (3, 3)
    """

    r = np.zeros((3, 3))
    r[0, 0] = 2 * q[0] ** 2 - 1 + 2 * q[1] ** 2
    r[0, 1] = 2 * (q[1] * q[2] + q[0] * q[3])
    r[0, 2] = 2 * (q[1] * q[3] - q[0] * q[2])
    r[1, 0] = 2 * (q[2] * q[1] - q[0] * q[3])
    r[1, 1] = 2 * q[0] ** 2 - 1 + 2 * q[2] ** 2
    r[1, 2] = 2 * (q[2] * q[3] + q[0] * q[1])
    r[2, 0] = 2 * (q[3] * q[1] + q[0] * q[2])
    r[2, 1] = 2 * (q[3] * q[2] - q[0] * q[1])
    r[2, 2] = 2 * q[0] ** 2 - 1 + 2 * q[3] ** 2

    return r


@numba.jit(nopython=True)
def quaternion_to_euler(q: np.ndarray):
    """
    convert quaternion to euler angles
    :param q: numpy array quaternion     (4, )
    :return: numpy array euler ZYX angles (3, )
    """
    r = np.zeros((3, 3))
    r[0, 0] = 2 * q[0] ** 2 - 1 + 2 * q[1] ** 2
    r[1, 0] = 2 * (q[1] * q[2] - q[0] * q[3])
    r[2, 0] = 2 * (q[1] * q[3] + q[0] * q[2])
    r[2, 1] = 2 * (q[2] * q[3] - q[0] * q[1])
    r[2, 2] = 2 * q[0] ** 2 - 1 + 2 * q[3] ** 2

    e = rot_mat_to_euler(r)

    return e


@numba.jit(nopython=True)
def axis_angle_to_quaternion(axis: np.ndarray, angle: float):
    """
    convert rotation about axis to quaternion
    :param axis: numpy array axis vector          (3, )
    :param angle: float angle rotation about axis
    :return: numpy array quaternion              (4, )
    """

    q_0 = np.cos(angle / 2)
    q_1 = -axis[0] * np.sin(angle / 2)
    q_2 = -axis[1] * np.sin(angle / 2)
    q_3 = -axis[2] * np.sin(angle / 2)

    return np.array([q_0, q_1, q_2, q_3])


@numba.jit(nopython=True)
def rot_mat_to_quaternion(r: np.ndarray):
    """
    convert rotation matrix to quaternion
    :param r: numpy array rotation matrix (3, 3)
    :return: numpy array quaternion      (4, )
    """

    k = np.zeros((4, 4))
    k[0, 0] = (1. / 3.) * (r[0, 0] - r[1, 1] - r[2, 2])
    k[0, 1] = (1. / 3.) * (r[1, 0] + r[0, 1])
    k[0, 2] = (1. / 3.) * (r[2, 0] + r[0, 2])
    k[0, 3] = (1. / 3.) * (r[1, 2] - r[2, 1])
    k[1, 0] = (1. / 3.) * (r[1, 0] + r[0, 1])
    k[1, 1] = (1. / 3.) * (r[1, 1] - r[0, 0] - r[2, 2])
    k[1, 2] = (1. / 3.) * (r[2, 1] + r[1, 2])
    k[1, 3] = (1. / 3.) * (r[2, 0] - r[0, 2])
    k[2, 0] = (1. / 3.) * (r[2, 0] + r[0, 2])
    k[2, 1] = (1. / 3.) * (r[2, 1] + r[1, 2])
    k[2, 2] = (1. / 3.) * (r[2, 2] - r[0, 0] - r[1, 1])
    k[2, 3] = (1. / 3.) * (r[0, 1] - r[1, 0])
    k[3, 0] = (1. / 3.) * (r[1, 2] - r[2, 1])
    k[3, 1] = (1. / 3.) * (r[2, 0] - r[0, 2])
    k[3, 2] = (1. / 3.) * (r[0, 1] - r[1, 0])
    k[3, 3] = (1. / 3.) * (r[0, 0] + r[1, 1] + r[2, 2])

    w, v = np.linalg.eig(k)
    q = v[:, np.argmax(w)]
    q = -1 * q[np.array([3, 0, 1, 2])]

    return q


@numba.jit(nopython=True)
def axis_angle_to_rot_mat(axis: np.ndarray, angle: float):
    """
    convert rotation about axis to rotation matrix
    :param axis: numpy array axis vector          (3, )
    :param angle: float angle rotation about axis
    :return: numpy array rotation matrix          (3, 3)
    """

    kx = axis[0]
    ky = axis[1]
    kz = axis[2]
    c_t = np.cos(angle)
    s_t = np.sin(angle)
    v_t = 1 - np.cos(angle)

    r = np.zeros((3, 3))
    r[0, 0] = kx * kx * v_t + c_t
    r[0, 1] = kx * ky * v_t - kz * s_t
    r[0, 2] = kx * kz * v_t + ky * s_t
    r[1, 0] = kx * ky * v_t + kz * s_t
    r[1, 1] = ky * ky * v_t + c_t
    r[1, 2] = ky * kz * v_t - kx * s_t
    r[2, 0] = kx * kz * v_t - ky * s_t
    r[2, 1] = ky * kz * v_t + kx * s_t
    r[2, 2] = kz * kz * v_t + c_t

    return r
    

@numba.jit(nopython=True)
def euler_to_rot_mat(x: np.ndarray):
    """
    convert euler angles to rotation matrix
    :param x: numpy array of euler ZYX angles (3, )
    :return: numpy array rotation matrix      (3, 3)
    """

    r = np.array([
        [np.cos(x[2]) * np.cos(x[1]),
         -np.sin(x[2]) * np.cos(x[0]) + np.cos(x[2]) * np.sin(x[1]) * np.sin(x[0]),
         np.sin(x[2]) * np.sin(x[0]) + np.cos(x[2]) * np.sin(x[1]) * np.cos(x[0])],
        [np.sin(x[2]) * np.cos(x[1]),
         np.cos(x[2]) * np.cos(x[0]) + np.sin(x[2]) * np.sin(x[1]) * np.sin(x[0]),
         -np.cos(x[2]) * np.sin(x[0]) + np.sin(x[2]) * np.sin(x[1]) * np.cos(x[0])],
        [-np.sin(x[1]),
         np.cos(x[1]) * np.sin(x[0]),
         np.cos(x[1]) * np.cos(x[0])]]
    )

    return r


@numba.jit(nopython=True)
def rot_mat_to_euler(r: np.ndarray):
    """
    convert rotation matrix to euler angles
    :param r: numpy array rotation matrix (3, 3)
    :return: numpy array euler ZYX angles (3, )
    """

    phi = np.arctan(r[2, 1] / r[2, 2])
    theta = -np.arctan2(r[2, 0], np.sqrt(1 - r[2, 0] ** 2))
    psi = np.arctan(r[1, 0] / r[0, 0])

    e = np.array([phi, theta, psi])

    return e


def test():

    # simple single axis rotation about range of angles
    axis = np.array([0., 1., 0.])
    for angle in np.linspace(-2 * np.pi, 2 * np.pi, 10):
        r_exp = np.array([
            [np.cos(angle), 0., np.sin(angle)],
            [0., 1., 0.],
            [-np.sin(angle), 0., np.cos(angle)],
        ])
        _single_test(axis, angle, r_exp)

    # random example
    axis = np.array([1., 2., 3.])
    angle = np.pi / 2
    r_exp = np.array([
        [0.07142857, -0.65892658, 0.7488082],
        [0.94464087,  0.28571429, 0.16131019],
        [-0.32023677,  0.69583267, 0.64285714],
    ])
    _single_test(axis, angle, r_exp)

    # really random examples, no expected rot mat
    for i in range(20):
        axis = np.random.rand(3)
        angle = np.pi * np.random.rand()
        _single_test(axis, angle)


def _single_test(axis: np.ndarray,
                 angle: np.ndarray,
                 r_exp: np.ndarray = None
                 ):

    is_passed = True
    axis = axis / np.linalg.norm(axis)

    print(f"Axis {axis},  Angle {angle}")

    # convert to rotation matrix
    r_1 = axis_angle_to_rot_mat(axis, angle)
    if r_exp is not None:
        if not np.isclose(r_exp, r_1).all():
            print(f"\tRotation matrix from axis angle does not match expected \n{r_1} vs \n{r_exp}")
            is_passed = False

    # compare quaternion from rotation matrix and axis angle
    q_1 = axis_angle_to_quaternion(axis, angle)
    q_2 = rot_mat_to_quaternion(r_1)
    if not np.isclose(q_1, q_2).all():
        print(f"\tquaternion from axis angle and rotation matrix do not match \n{q_1} vs \n{q_2}")
        is_passed = False

    # convert back to rotation matrix
    r_2 = quaternion_to_rot_mat(q_1)
    if not np.isclose(r_1, r_2).all():
        print(f"\tRotation matrix from axis angle and quaternion do not match \n{r_1} vs \n{r_2}")
        is_passed = False

    # compare euler angles from rotation matrix and quaternion
    e_1 = rot_mat_to_euler(r_1)
    e_2 = quaternion_to_euler(q_1)
    if not np.isclose(e_1, e_2).all():
        print(f"\tEuler angles from rotation matrix and quaternion do not match \n{e_1} vs \n{e_2}")
        is_passed = False

    # convert back to rotation matrix
    r_3 = euler_to_rot_mat(e_1)
    if not np.isclose(r_1, r_3).all():
        print(f"\tRotation matrix from axis angle and euler do not match \n{r_1} vs \n{r_3}")
        is_passed = False

    if is_passed:
        print("\tTest passed")
