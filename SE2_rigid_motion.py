import numpy as np
import tolerances
from lie_utils import coeff_A, coeff_B, coeff_C, coeff_D


r'''
tau = [rho; theta] is the cartesian space element, where rho \in R^2 is the translation vector and theta \in R is the rotation angle in radians
M = [R, t; 0, 1] is the group element, where R \in R^{2x2} is the rotation matrix and t \in R^2 is the translation vector
tau_hat = [theta_hat, rho; 0, 0] is the algebra element, where theta_hat = [0, -theta; theta, 0]
'''
tol = tolerances.small_case_tol  # kept for backward compatibility (no longer needed for correctness)

_J = np.array([[0., -1.], [1., 0.]])  # 90 degrees rotation generator

def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def group_element(tau):
    return Exp(tau)

def group_identity():
    return np.eye(3)

def group_composition(M1, M2):
    return M1 @ M2

def group_inverse(M):
    R = M[:2, :2]
    t = M[:2, 2]
    return np.block([[R.T, -R.T @ t.reshape(-1, 1)], [np.zeros((1, 2)), 1.]])

def group_action(M, x):
    return M @ x

def algebra_element(tau):
    return hat(tau)

def compose_cartesian_element(rho, theta):
    return np.concatenate([np.asarray(rho, dtype=float).reshape(2), [theta]])

def decompose_cartesian_element(tau):
    tau = np.asarray(tau, dtype=float).reshape(3)
    return tau[:2], tau[2]

def hat(tau):
    rho, theta = decompose_cartesian_element(tau)
    theta_hat = np.array([[0., -theta], [theta, 0.]])
    return np.block([[theta_hat, rho.reshape(-1, 1)], [np.zeros((1, 3))]])

def vee(tau_hat):
    return compose_cartesian_element(tau_hat[:2, 2], tau_hat[1, 0])

def exp(tau_hat):
    return Exp(vee(tau_hat))

def log(M):
    return hat(Log(M))

def Exp(tau):
    rho, theta = decompose_cartesian_element(tau)
    return np.block([[_rot(theta), V(theta) @ rho.reshape(-1, 1)], [np.zeros((1, 2)), 1.]])

def Log(M):
    R = M[:2, :2]
    t = M[:2, 2]
    theta = np.arctan2(R[1, 0] - R[0, 1], R[0, 0] + R[1, 1])
    return compose_cartesian_element(Vinv(theta) @ t, theta)

def plus_right(M, tau):
    return M @ Exp(tau)

def plus_left(M, tau):
    return Exp(tau) @ M

def minus_right(M1, M2):
    return Log(group_inverse(M2) @ M1)

def minus_left(M1, M2):
    return Log(M1 @ group_inverse(M2))

def adjoint(M):
    R = M[:2, :2]
    t = M[:2, 2]
    return np.block([[R, -(_J @ t).reshape(-1, 1)], [np.zeros((1, 2)), 1.]])

def jacobian_inverse(M):
    return -adjoint(M)

def jacobian_composition_1(M1, M2):
    return adjoint(group_inverse(M2))

def jacobian_composition_2(M1, M2):
    return np.eye(3)

def jacobian_right(tau):
    rho, theta = decompose_cartesian_element(tau)
    A, B, C = coeff_A(theta), coeff_B(theta), coeff_C(theta)
    return np.array([[A, theta * B, theta * C * rho[0] - B * rho[1]],
                     [-theta * B, A, B * rho[0] + theta * C * rho[1]],
                     [0., 0., 1.]])

def jacobian_right_inverse(tau):
    return np.linalg.inv(jacobian_right(tau))

def jacobian_left(tau):
    rho, theta = decompose_cartesian_element(tau)
    A, B, C = coeff_A(theta), coeff_B(theta), coeff_C(theta)
    return np.array([[A, -theta * B, theta * C * rho[0] + B * rho[1]],
                     [theta * B, A, -B * rho[0] + theta * C * rho[1]],
                     [0., 0., 1.]])

def jacobian_left_inverse(tau):
    return np.linalg.inv(jacobian_left(tau))

def jacobian_plus_right_1(M, tau):
    return adjoint(group_inverse(Exp(tau)))

def jacobian_plus_right_2(M, tau):
    return jacobian_right(tau)

def jacobian_minus_right_1(M1, M2):
    # d(M1 (-) M2)/dM1 = Jr^-1(M1 (-) M2)
    return jacobian_right_inverse(minus_right(M1, M2))

def jacobian_minus_right_2(M1, M2):
    # d(M1 (-) M2)/dM2 = -Jl^-1(M1 (-) M2)
    return -jacobian_left_inverse(minus_right(M1, M2))

def jacobian_motion_action_1(M, p):
    R = M[:2, :2]
    return np.block([R, (R @ _J @ np.asarray(p, dtype=float)[:2]).reshape(-1, 1)])

def jacobian_motion_action_2(M, p):
    return M[:2, :2]

def V(theta):
    """SE(2) V matrix: V(theta) = sin(t)/t * I + (1 - cos(t))/t * J  (stable for small theta)."""
    return coeff_A(theta) * np.eye(2) + theta * coeff_B(theta) * _J

def Vinv(theta):
    """V(theta)^-1 = (t/2) cot(t/2) * I - (t/2) * J  (stable for small theta)."""
    return (1. - theta ** 2 * coeff_D(theta)) * np.eye(2) - 0.5 * theta * _J
