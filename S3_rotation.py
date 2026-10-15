import numpy as np
import tolerances
from lie_utils import vec_hat, coeff_A, coeff_B, coeff_C, coeff_D


r'''
tau = theta * u (angle-axis representation) is the cartesian space element, where theta \in R is the angle in radians and u \in R^3 is the unit vector along the axis of rotation
q = [cos(theta/2); sin(theta/2) * u] is the group element (unit quaternion, Hamilton convention, [w, x, y, z])
tau_hat = tau / 2 = (theta/2) * u is the algebra element (a pure quaternion, stored as its 3-vector part)
'''
tol = tolerances.small_case_tol  # kept for backward compatibility (no longer needed for correctness)

def group_element(tau):
    tau = np.asarray(tau, dtype=float).reshape(3)
    theta = np.linalg.norm(tau)
    # sin(theta/2) * u = sin(theta/2)/theta * tau = 0.5 * sinc(theta/2) * tau  (smooth at theta = 0)
    return np.concatenate([[np.cos(theta / 2.)], 0.5 * np.sinc(theta / (2. * np.pi)) * tau])

def group_identity():
    return np.array([1., 0., 0., 0.])

def group_composition(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    q = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                  w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                  w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                  w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
    return q

def group_inverse(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([w, -x, -y, -z])

def group_action(q, x):
    return group_composition(group_composition(q, np.concatenate([[0.], x])), group_inverse(q))[1:]

def algebra_element(tau):
    return np.asarray(tau, dtype=float) / 2.

def compose_cartesian_element(theta, u):
    return theta * u

def decompose_cartesian_element(tau):
    tau = np.asarray(tau, dtype=float).reshape(3)
    theta = np.linalg.norm(tau)
    if theta < np.finfo(float).tiny:
        return 0., np.array([0., 0., 1.])
    return theta, tau / theta

def hat(tau):
    return np.asarray(tau, dtype=float) / 2.

def vee(tau_hat):
    return 2. * np.asarray(tau_hat, dtype=float)

def exp(tau_hat):
    return group_element(2. * np.asarray(tau_hat, dtype=float))

def log(q):
    return Log(q) / 2.

def Exp(tau):
    return group_element(tau)

def Log(q):
    """Unit quaternion -> rotation vector, theta in [0, pi].

    q and -q encode the same rotation, so q is flipped to w >= 0 first (otherwise theta > pi is returned).
    """
    q = np.asarray(q, dtype=float)
    if q[0] < 0.:
        q = -q
    w, v = q[0], q[1:]
    v_norm = np.linalg.norm(v)
    h = np.arctan2(v_norm, w)                        # theta / 2
    return 2. * v / (np.sinc(h / np.pi) * np.hypot(w, v_norm))

def plus_right(q, tau):
    return group_composition(q, Exp(tau))

def plus_left(q, tau):
    return group_composition(Exp(tau), q)

def minus_right(q1, q2):
    return Log(group_composition(group_inverse(q2), q1))

def minus_left(q1, q2):
    return Log(group_composition(q1, group_inverse(q2)))

def adjoint(q):
    return Rq_mat(q)

def jacobian_inverse(q):
    return -Rq_mat(q)

def jacobian_composition_1(q1, q2):
    return Rq_mat(group_inverse(q2))

def jacobian_composition_2(q1, q2):
    return np.eye(3)

def jacobian_right(tau):
    K = vec_hat(tau); theta = np.linalg.norm(tau)
    return np.eye(3) - coeff_B(theta) * K + coeff_C(theta) * K @ K

def jacobian_right_inverse(tau):
    K = vec_hat(tau); theta = np.linalg.norm(tau)
    return np.eye(3) + 0.5 * K + coeff_D(theta) * K @ K

def jacobian_left(tau):
    K = vec_hat(tau); theta = np.linalg.norm(tau)
    return np.eye(3) + coeff_B(theta) * K + coeff_C(theta) * K @ K

def jacobian_left_inverse(tau):
    K = vec_hat(tau); theta = np.linalg.norm(tau)
    return np.eye(3) - 0.5 * K + coeff_D(theta) * K @ K

def jacobian_plus_right_1(q, tau):
    return Rq_mat(group_inverse(Exp(tau)))

def jacobian_plus_right_2(q, tau):
    return jacobian_right(tau)

def jacobian_minus_right_1(q1, q2):
    return jacobian_right_inverse(minus_right(q1, q2))

def jacobian_minus_right_2(q1, q2):
    return -jacobian_left_inverse(minus_right(q1, q2))

def jacobian_rotation_action_1(q, x):
    return -Rq_mat(q) @ vec_hat(x)

def jacobian_rotation_action_2(q, x):
    return Rq_mat(q)

def Rq_mat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    Rq = np.array([[w**2 + x**2 - y**2 - z**2, 2*(x*y - w*z), 2*(x*z + w*y)],
                   [2*(x*y + w*z), w**2 - x**2 + y**2 - z**2, 2*(y*z - w*x)],
                   [2*(x*z - w*y), 2*(y*z + w*x), w**2 - x**2 - y**2 + z**2]])
    return Rq

