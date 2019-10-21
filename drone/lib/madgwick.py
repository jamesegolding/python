import numpy as np
import orientation as ori
import numba


class Madgwick:

    def __init__(self,
                 t_sample:float = 1./256.,
                 q: np.ndarray = np.array([1., 0., 0., 0.]),
                 gain: float = 0.1
                 ):
        self.t_sample = t_sample
        self.q = np.array(q)
        self.gain = gain

    def update(self,
               gyroscope,
               accelerometer,
               magnetometer,
               ):

        q, e = update(self.q, self.gain, self.t_sample, gyroscope, accelerometer, magnetometer)

        self.q = q

        return q, e


@numba.jit(nopython=True)
def update(q: np.ndarray,
           gain: float,
           t_sample: float,
           g: np.ndarray,
           a: np.ndarray,
           m: np.ndarray,
           ):

    # normalize accelerometer
    a_norm = np.linalg.norm(a)
    if a_norm == 0.:
        return q, np.zeros(3)

    # normalize magnetometer
    m_norm = np.linalg.norm(m)
    if m_norm == 0.:
        return q, np.zeros(3)

    h = ori.quaternion_prod(
        q, ori.quaternion_prod(np.array([0., m[0], m[1], m[2]]), ori.quaternion_conj(q)),
    )

    b = np.array([0., np.linalg.norm(h[1:3]), 0., h[3]])

    # gradient descent step
    F = np.array([
        2 * (q[1] * q[3] - q[0] * q[2]) - a[0],
        2 * (q[0] * q[1] + q[2] * q[3]) - a[1],
        2 * (0.5 - q[1] ** 2 - q[2] ** 2) - a[2],
        2 * b[1] * (0.5 - q[2] ** 2 - q[3] ** 2) + 2 * b[3] * (q[1] * q[3] - q[0] * q[2]) - m[0],
        2 * b[1] * (q[1] * q[2] - q[0] * q[3]) + 2 * b[3] * (q[0] * q[1] + q[2] * q[3]) - m[1],
        2 * b[1] * (q[0] * q[2] + q[1] * q[3]) + 2 * b[3] * (0.5 - q[1] ** 2 - q[2] ** 2) - m[2],
    ])

    J = np.array([
        [-2*q[2],                   2*q[3],                  -2*q[0],                   2*q[1]],
        [2*q[1],                    2*q[0],                   2*q[3],                   2*q[2]],
        [0,                        -4*q[1],                  -4*q[2],                   0],
        [-2*b[3]*q[2],              2*b[3]*q[3],             -4*b[1]*q[2]-2*b[3]*q[0], -4*b[1]*q[3]+2*b[3]*q[1]],
        [-2*b[1]*q[3]+2*b[3]*q[1],  2*b[1]*q[2]+2*b[3]*q[0],  2*b[1]*q[1]+2*b[3]*q[3], -2*b[1]*q[0]+2*b[3]*q[2]],
        [2*b[1]*q[2],               2*b[1]*q[3]-4*b[3]*q[1],  2*b[1]*q[0]-4*b[3]*q[2],  2*b[1]*q[1]],
    ])

    # calculate step
    step = J.T @ F
    step = step / np.linalg.norm(step)

    # rate of change of q
    q_dot = 0.5 * ori.quaternion_prod(
        q,
        np.array([0., g[0], g[1], g[2]]),
    ) - gain * step

    q = q + q_dot * t_sample
    q = q / np.linalg.norm(q)

    # euler
    e = ori.quaternion_to_euler(ori.quaternion_conj(q))

    return q, e
