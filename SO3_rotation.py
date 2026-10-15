import numpy as np
import tolerances
from lie_utils import vec_hat, vec_vee, coeff_A, coeff_B, coeff_C, coeff_D


r'''
tau = theta * u (angle-axis representation) is the cartesian space element, where theta \in R is the angle in radians and u \in R^3 is the unit vector along the axis of rotation
R = I + sin(theta) * u_hat + (1 - cos(theta)) * u_hat^2 is the group element
tau_hat = theta * u_hat = [tau]_x is the algebra element, where u_hat = [0, -u3, u2; u3, 0, -u1; -u2, u1, 0]
'''
tol = tolerances.small_case_tol  # kept for backward compatibility (no longer needed for correctness)

def group_element(tau):
    tau = np.asarray(tau, dtype=float).reshape(3)
    theta = np.linalg.norm(tau)
    K = vec_hat(tau)
    return np.eye(3) + coeff_A(theta) * K + coeff_B(theta) * K @ K

def group_identity():
    return np.eye(3)

def group_composition(R1, R2):
    return R1 @ R2

def group_inverse(R):
    return R.T

def group_action(R, x):
    return R @ x

def algebra_element(tau):
    return vec_hat(tau)

def compose_cartesian_element(theta, u):
    return theta * u

def decompose_cartesian_element(tau):
    tau = np.asarray(tau, dtype=float).reshape(3)
    theta = np.linalg.norm(tau)
    if theta < np.finfo(float).tiny:
        return 0., np.array([0., 0., 1.])
    return theta, tau / theta

def hat(tau):
    return vec_hat(tau)

def vee(tau_hat):
    return vec_vee(tau_hat)

def exp(tau_hat):
    return group_element(vee(tau_hat))

def log(R):
    return vec_hat(Log(R))

def Exp(tau):
    return group_element(tau)

def Log(R):
    """Rotation matrix -> rotation vector theta*u with theta in [0, pi].

    Uses atan2 (accurate near 0, never NaN) and, for theta >= pi/2, the symmetric part of R to
    recover the axis (the antisymmetric part vanishes at theta = pi).
    """
    R = np.asarray(R, dtype=float)
    w = vec_vee((R - R.T) / 2.)                      # sin(theta) * u
    s = np.linalg.norm(w)
    c = np.clip((np.trace(R) - 1.) / 2., -1., 1.)    # cos(theta)
    theta = np.arctan2(s, c)
    if theta < np.pi / 2.:
        return w / coeff_A(theta)                    # theta * u = w * theta / sin(theta)
    uuT = ((R + R.T) / 2. - c * np.eye(3)) / (1. - c)  # = u u^T
    k = int(np.argmax(np.diag(uuT)))
    u = uuT[:, k] / np.linalg.norm(uuT[:, k])
    if w @ u < 0.:
        u = -u
    return theta * u

def plus_right(R, tau):
    return R @ Exp(tau)

def plus_left(R, tau):
    return Exp(tau) @ R

def minus_right(R1, R2):
    return Log(R2.T @ R1)

def minus_left(R1, R2):
    return Log(R1 @ R2.T)

def adjoint(R):
    return R

def jacobian_inverse(R):
    return -R

def jacobian_composition_1(R1, R2):
    return R2.T

def jacobian_composition_2(R1, R2):
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

def jacobian_plus_right_1(R, tau):
    return Exp(tau).T

def jacobian_plus_right_2(R, tau):
    return jacobian_right(tau)

def jacobian_minus_right_1(R1, R2):
    return jacobian_right_inverse(minus_right(R1, R2))

def jacobian_minus_right_2(R1, R2):
    return -jacobian_left_inverse(minus_right(R1, R2))

def jacobian_rotation_action_1(R, x):
    return -R @ vec_hat(x)

def jacobian_rotation_action_2(R, x):
    return R

def qR_quat(R):
    """Rotation matrix -> unit quaternion [w, x, y, z] with w >= 0 (Shepperd's method: robust for every rotation, incl. theta = pi)."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0.:
        s = 2. * np.sqrt(1. + t)
        q = np.array([s / 4., (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = 2. * np.sqrt(1. + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([(R[2, 1] - R[1, 2]) / s, s / 4., (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] >= R[2, 2]:
        s = 2. * np.sqrt(1. + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4., (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2. * np.sqrt(1. + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4.])
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0. else q

