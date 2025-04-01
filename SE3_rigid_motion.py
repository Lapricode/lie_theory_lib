import numpy as np


'''
tau = [rho; theta * u] is the cartesian space element, where rho \in R^3 is the translation vector and theta * u \in R^3 is the angle-axis representation
M = [R, t; 0, 1] is the group element, where R \in R^{3x3} is the rotation matrix and t \in R^3 is the translation vector
tau_hat = [theta * u_hat, rho; 0, 1] is the algebra element, where u_hat = [0, -u3, u2; u3, 0, -u1; -u2, u1, 0]
'''
tol = 1e-5  # tolerance for numerical issues

def group_element(rho, theta, u):
    u_hat = vec_hat(u)
    R = np.eye(3) + np.sin(theta) * u_hat + (1 - np.cos(theta)) * u_hat @ u_hat
    t = V(theta) @ rho
    M = np.block([[R, t.reshape(-1, 1)], [np.zeros((1, 3)), 1.]])
    return M

def group_composition(M1, M2):
    return M1 @ M2

def group_inverse(M):
    R = M[:3, :3]
    t = M[:3, 3]
    return np.block([[R.T, -R.T @ t.reshape(-1, 1)], [np.zeros((1, 3)), 1.]])

def group_action(M, x):
    return M @ x

def algebra_element(rho, theta, u):
    v_hat = theta * vec_hat(u)
    tau_hat = np.block([[v_hat, rho.reshape(-1, 1)], [np.zeros((1, 4))]])
    return tau_hat

def hat(rho, theta, u):
    v_hat = theta * vec_hat(u)
    tau_hat = np.block([[v_hat, rho.reshape(-1, 1)], [np.zeros((1, 4))]])
    return tau_hat

def vee(tau_hat):
    rho = tau_hat[:3, 3]
    v = np.array([tau_hat[2, 1], tau_hat[0, 2], tau_hat[1, 0]])
    theta = np.linalg.norm(v)
    u = v / theta if abs(theta) >= tol else np.array([[0.], [0.], [1.]])
    return rho, theta, u

def exp(tau_hat):
    rho = tau_hat[:3, 3]
    v = np.array([tau_hat[2, 1], tau_hat[0, 2], tau_hat[1, 0]])
    theta = np.linalg.norm(v)
    u = v / theta if abs(theta) >= tol else np.array([[0.], [0.], [1.]])
    u_hat = vec_hat(u)
    Exp_theta = np.eye(3) + np.sin(theta) * u_hat + (1 - np.cos(theta)) * u_hat @ u_hat
    M = np.block([[Exp_theta, V(theta, u) @ rho.reshape(-1, 1)], [np.zeros((1, 3)), 1.]])
    return M

def log(M):
    R = M[:3, :3]
    t = M[:3, 3]
    theta = np.arccos((np.trace(R) - 1.) / 2.)
    rho = Vinv(theta, t) @ t
    v_hat = theta * (R - R.T) / (2. * np.sin(theta)) if abs(theta) >= tol else (R - R.T) / 2.
    tau_hat = np.block([[v_hat, rho.reshape(-1, 1)], [np.zeros((1, 4))]])
    return tau_hat

def Exp(rho, theta, u):
    u_hat = vec_hat(u)
    Exp_theta = np.eye(3) + np.sin(theta) * u_hat + (1 - np.cos(theta)) * u_hat @ u_hat
    M = np.block([[Exp_theta, V(theta, u) @ rho.reshape(-1, 1)], [np.zeros((1, 3)), 1.]])
    return M

def Log(M):
    R = M[:3, :3]
    t = M[:3, 3]
    theta = np.arccos((np.trace(R) - 1.) / 2.)
    rho = Vinv(theta, t) @ t
    v_hat = theta * (R - R.T) / (2. * np.sin(theta)) if abs(theta) >= tol else (R - R.T) / 2.
    v = np.array([v_hat[2, 1], v_hat[0, 2], v_hat[1, 0]])
    u = v / theta if abs(theta) >= tol else np.array([[0.], [0.], [1.]])
    return rho, theta, u

def plus_right(M, rho, theta, u):
    return M @ Exp(rho, theta, u)

def plus_left(M, rho, theta, u):
    return Exp(rho, theta, u) @ M

def minus_right(M1, M2):
    return Log(group_inverse(M2) @ M1)

def minus_left(M1, M2):
    return Log(M1 @ group_inverse(M2))

def adjoint(M):
    R = M[:3, :3]
    t = M[:3, 3]
    return np.block([[R, vec_hat(t) @ R], [np.zeros((3, 3)), R]])

def jacobian_inverse(M):
    return -adjoint(M)

def jacobian_composition_1(M1, M2):
    R2 = M2[:3, :3]
    t2 = M2[:3, 3]
    return np.block([[R2.T, -R2.T @ vec_hat(t2)], [np.zeros((3, 3)), R2.T]])

def jacobian_composition_2(R1, R2):
    return np.eye(6)

def jacobian_right(rho, theta, u):
    return np.block([[Jl_theta(-theta, u), Q(-rho, -theta, u)], [np.zeros((3, 3)), Jl_theta(-theta, u)]])

def jacobian_right_inverse(rho, theta, u):
    return np.block([[Jl_inv_theta(-theta, u), -Jl_inv_theta(-theta, u) @ Q(-rho, -theta, u) @ Jl_inv_theta(-theta, u)], [np.zeros((3, 3)), Jl_inv_theta(-theta, u)]])

def jacobian_left(rho, theta, u):
    return np.block([[Jl_theta(theta, u), Q(rho, theta, u)], [np.zeros((3, 3)), Jl_theta(theta, u)]])

def jacobian_left_inverse(rho, theta, u):
    return np.block([[Jl_inv_theta(theta, u), -Jl_inv_theta(theta, u) @ Q(rho, theta, u) @ Jl_inv_theta(theta, u)], [np.zeros((3, 3)), Jl_inv_theta(theta, u)]])

def jacobian_plus_right_1(M, rho, theta, u):
    return adjoint(group_inverse(Exp(rho, theta, u)))

def jacobian_plus_right_2(M, rho, theta, u):
    return jacobian_right(rho, theta, u)

def jacobian_minus_right_1(M1, M2):
    return -jacobian_left_inverse(minus_right(M1, M2))

def jacobian_minus_right_2(M1, M2):
    return jacobian_right_inverse(minus_right(M1, M2))

def jacobian_rotation_action_1(M, p):
    R = M[:3, :3]
    return np.block([R, -R @ vec_hat(p)])

def jacobian_rotation_action_2(M, p):
    R = M[:3, :3]
    return R

def vec_hat(v):
    return np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])

def V(theta, u):
    u_hat = vec_hat(u)
    return np.eye(3) + (1. - np.cos(theta)) / theta * u_hat + (theta - np.sin(theta)) / theta * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)

def Vinv(theta, u):
    return np.linalg.inv(V(theta, u)) if abs(theta) >= tol else np.eye(3)

def Jl_theta(theta, u):
    u_hat = vec_hat(u)
    return np.eye(3) + (1 - np.cos(theta)) / theta * u_hat + (theta - np.sin(theta)) / theta * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)

def Jl_inv_theta(theta, u):
    u_hat = vec_hat(u)
    return np.eye(3) - 0.5 * theta * u_hat + (1. - 0.5 * theta * (1. + np.cos(theta)) / np.sin(theta)) * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)

def Q(rho, theta, u):
    rho_hat = vec_hat(rho)
    u_hat = vec_hat(u)
    if abs(theta) >= tol:
        Q = 0.5 * rho_hat + (theta - np.sin(theta)) / theta**2 * (u_hat @ rho_hat + rho_hat @ u_hat + u_hat @ rho_hat @ u_hat) - \
            (1. - 0.5 * theta**2 - np.cos(theta)) / theta**2 * (u_hat @ u_hat @ rho_hat + rho_hat @ u_hat @ u_hat - 3. * u_hat @ rho_hat @ u_hat) - \
            0.5 * ((1. - theta**2 / 2. - np.cos(theta)) / theta - 3. * (theta - np.sin(theta) - theta**3 / 6.) / theta**2) * (u_hat @ rho_hat @ u_hat @ u_hat + u_hat @ u_hat @ rho_hat @ u_hat)
    else:
        Q = 0.5 * rho_hat
    return Q


def testing():
    pass
