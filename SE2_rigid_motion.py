import numpy as np


'''
tau = [rho; theta] is the cartesian space element, where rho \in R^2 is the translation vector and theta \in R is the rotation angle in radians
M = [R, t; 0, 1] is the group element, where R \in R^{2x2} is the rotation matrix and t \in R^2 is the translation vector
tau_hat = [theta_hat, rho; 0, 1] is the algebra element, where theta_hat = [0, -theta; theta, 0]
'''
tol = 1e-5  # tolerance for numerical issues

def group_element(tau):
    rho, theta = decompose_cartesian_element(tau)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = V(theta) @ rho
    M = np.block([[R, t.reshape(-1, 1)], [np.zeros((1, 2)),  1.]])
    return M

def group_identity():
    return np.eye(3)

def group_composition(M1, M2):
    return M1 @ M2

def group_inverse(M):
    R = M[:2, :2]
    t = M[:2, 2]
    return np.block([[R.T, -R.T @ t.reshape(-1, 1)], [np.zeros((1, 2)), 1]])

def group_action(M, x):
    return M @ x

def algebra_element(tau):
    rho, theta = decompose_cartesian_element(tau)
    theta_hat = np.array([[0., -theta], [theta, 0.]])
    tau_hat = np.block([[theta_hat, rho.reshape(-1, 1)], [np.zeros((1, 3))]])
    return tau_hat

def compose_cartesian_element(rho, theta):
    return np.block([rho, theta])

def decompose_cartesian_element(tau):
    rho = tau[:2].reshape(-1,)
    theta = tau[2]
    return rho, theta

def hat(tau):
    rho, theta = decompose_cartesian_element(tau)
    theta_hat = np.array([[0., -theta], [theta, 0.]])
    tau_hat = np.block([[theta_hat, rho.reshape(-1, 1)], [np.zeros((1, 3))]])
    return tau_hat

def vee(tau_hat):
    rho = tau_hat[:2, 2]
    theta = tau_hat[1, 0]
    return compose_cartesian_element(rho, theta)

def exp(tau_hat):
    rho = tau_hat[:2, 2]
    theta = tau_hat[1, 0]
    Exp_theta = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    M = np.block([[Exp_theta, V(theta) @ rho.reshape(-1, 1)], [np.zeros((1, 2)),  1.]])
    return M

def log(M):
    R = M[:2, :2]
    t = M[:2, 2]
    theta = np.arctan2(R[1, 0], R[0, 0])
    rho = Vinv(theta) @ t
    theta_hat = np.array([[0., -theta], [theta, 0.]])
    tau_hat = np.block([[theta_hat, rho.reshape(-1, 1)], [np.zeros((1, 3))]])
    return tau_hat

def Exp(tau):
    rho, theta = decompose_cartesian_element(tau)
    Exp_theta = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    M = np.block([[Exp_theta, V(theta) @ rho.reshape(-1, 1)], [np.zeros((1, 2)),  1.]])
    return M

def Log(M):
    R = M[:2, :2]
    t = M[:2, 2]
    theta = np.arctan2(R[1, 0], R[0, 0])
    rho = Vinv(theta) @ t
    return compose_cartesian_element(rho, theta)

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
    return np.block([[R, -np.array([[0., -1.], [1., 0.]]) @ t.reshape(-1, 1)], [np.zeros((1, 2)), 1.]])

def jacobian_inverse(M):
    return -adjoint(M)

def jacobian_composition_1(M1, M2):
    M2_inv = group_inverse(M2)
    return adjoint(M2_inv)

def jacobian_composition_2(R1, R2):
    return np.eye(3)

def jacobian_right(tau):
    rho, theta = decompose_cartesian_element(tau)
    rho1, rho2 = rho[0], rho[1]
    if abs(theta) >= tol:
        Jr = np.array([[np.sin(theta) / theta, (1. - np.cos(theta)) / theta, (theta * rho1 - rho2 + rho2 * np.cos(theta) - rho1 * np.sin(theta)) / theta**2], \
                        [(np.cos(theta) - 1.) / theta, np.sin(theta) / theta, (rho1 + theta * rho2 - rho1 * np.cos(theta) - rho2 * np.sin(theta)) / theta**2], \
                        [0., 0., 1.]])
    else:
        Jr = np.block([[np.eye(2), np.array([[-rho2 / 2.], [rho1 / 2.]])], [np.zeros((1, 2)), 1.]])
    return Jr

def jacobian_right_inverse(tau):
    Jrinv = np.linalg.inv(jacobian_right(tau))
    return Jrinv

def jacobian_left(tau):
    rho, theta = decompose_cartesian_element(tau)
    rho1, rho2 = rho[0], rho[1]
    if abs(theta) >= tol:
        Jl = np.array([[np.sin(theta) / theta, (np.cos(theta) - 1.) / theta, (theta * rho1 + rho2 - rho2 * np.cos(theta) - rho1 * np.sin(theta)) / theta**2], \
                        [(1. - np.cos(theta)) / theta, np.sin(theta) / theta, (-rho1 + theta * rho2 + rho1 * np.cos(theta) - rho2 * np.sin(theta)) / theta**2], \
                        [0., 0., 1.]])
    else:
        Jl = np.block([[np.eye(2), np.array([[rho2 / 2.], [-rho1 / 2.]])], [np.zeros((1, 2)), 1.]])
    return Jl

def jacobian_left_inverse(tau):
    Jlinv = np.linalg.inv(jacobian_left(tau))
    return Jlinv

def jacobian_plus_right_1(M, tau):
    return adjoint(group_inverse(Exp(tau)))

def jacobian_plus_right_2(M, tau):
    return jacobian_right(tau)

def jacobian_minus_right_1(M1, M2):
    return -jacobian_left_inverse(minus_right(M1, M2))

def jacobian_minus_right_2(M1, M2):
    return jacobian_right_inverse(minus_right(M1, M2))

def jacobian_motion_action_1(M, p):
    R = M[:2, :2]
    return np.block([R, R @ np.array([[0., -1.], [1., 0.]]) @ p[:2].reshape(-1, 1)])

def jacobian_motion_action_2(M, p):
    R = M[:2, :2]
    return R

def V(theta):
    return np.sin(theta) / theta * np.eye(2) + (1 - np.cos(theta)) / theta * np.array([[0, -1], [1, 0]]) if abs(theta) >= tol else np.eye(2)

def Vinv(theta):
    return theta / (2 * (1 - np.cos(theta))) * (np.sin(theta) * np.eye(2) - (1 - np.cos(theta)) * np.array([[0, -1], [1, 0]])) if abs(theta) >= tol else np.eye(2)
