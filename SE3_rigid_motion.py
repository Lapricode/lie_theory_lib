import numpy as np
import tolerances


'''
tau = [rho; theta * u] is the cartesian space element, where rho \in R^3 is the translation vector and theta * u \in R^3 is the angle-axis representation
M = [R, t; 0, 1] is the group element, where R \in R^{3x3} is the rotation matrix and t \in R^3 is the translation vector
tau_hat = [theta * u_hat, rho; 0, 0] is the algebra element, where u_hat = [0, -u3, u2; u3, 0, -u1; -u2, u1, 0]
'''
tol = tolerances.small_case_tol  # tolerance for numerical issues

def group_element(tau):
    rho, theta, u = decompose_cartesian_element(tau)
    u_hat = vec_hat(u)
    R = np.eye(3) + np.sin(theta) * u_hat + (1 - np.cos(theta)) * u_hat @ u_hat
    t = V(theta, u) @ rho
    M = np.block([[R, t.reshape(-1, 1)], [np.zeros((1, 3)), 1.]])
    return M

def group_identity():
    return np.eye(4)

def group_composition(M1, M2):
    return M1 @ M2

def group_inverse(M):
    R = M[:3, :3]
    t = M[:3, 3]
    return np.block([[R.T, -R.T @ t.reshape(-1, 1)], [np.zeros((1, 3)), 1.]])

def group_action(M, x):
    return M @ x

def algebra_element(tau):
    rho, theta, u = decompose_cartesian_element(tau)
    v_hat = theta * vec_hat(u)
    tau_hat = np.block([[v_hat, rho.reshape(-1, 1)], [np.zeros((1, 4))]])
    return tau_hat

def compose_cartesian_element(rho, theta, u):
    return np.block([rho, theta * u])

def decompose_cartesian_element(tau):
    rho = tau[:3]
    v = tau[3:]
    theta = np.linalg.norm(v)
    if abs(theta) < tol:
        return rho, 0., np.array([0., 0., 1.])
    u = v / theta
    return rho, theta, u

def hat(tau):
    rho, theta, u = decompose_cartesian_element(tau)
    v_hat = theta * vec_hat(u)
    tau_hat = np.block([[v_hat, rho.reshape(-1, 1)], [np.zeros((1, 4))]])
    return tau_hat

def vee(tau_hat):
    rho = tau_hat[:3, 3]
    v = np.array([tau_hat[2, 1], tau_hat[0, 2], tau_hat[1, 0]])
    theta = np.linalg.norm(v)
    if abs(theta) < tol:
        return compose_cartesian_element(rho, 0., np.array([0., 0., 1.]))
    u = v / theta
    return compose_cartesian_element(rho, theta, u)

def exp(tau_hat):
    rho = tau_hat[:3, 3]
    v = np.array([tau_hat[2, 1], tau_hat[0, 2], tau_hat[1, 0]])
    theta = np.linalg.norm(v)
    if abs(theta) < tol:
        return np.block([[np.eye(3), rho.reshape(-1, 1)], [np.zeros((1, 3)), 1.]])
    u = v / theta
    u_hat = vec_hat(u)
    Exp_theta = np.eye(3) + np.sin(theta) * u_hat + (1 - np.cos(theta)) * u_hat @ u_hat
    M = np.block([[Exp_theta, V(theta, u) @ rho.reshape(-1, 1)], [np.zeros((1, 3)), 1.]])
    return M

def log(M):
    R = M[:3, :3]
    t = M[:3, 3]
    theta = np.arccos(np.clip((np.trace(R) - 1.) / 2., -1.0, 1.0))
    if abs(theta) < tol:
        return np.block([[(R - R.T) / 2., t.reshape(-1, 1)], [np.zeros((1, 4))]])
    v_hat = theta * (R - R.T) / (2. * np.sin(theta))
    u = np.array([v_hat[2, 1], v_hat[0, 2], v_hat[1, 0]]) / theta
    rho = Vinv(theta, u) @ t
    tau_hat = np.block([[v_hat, rho.reshape(-1, 1)], [np.zeros((1, 4))]])
    return tau_hat
    # R = M[:3, :3]
    # t = M[:3, 3]
    # theta = np.arccos(np.clip((np.trace(R) - 1.) / 2., -1.0, 1.0))
    # if np.isclose(theta, 0., rtol = tol, atol = tol):
    #     return np.block([[(R - R.T) / 2., t.reshape(-1, 1)], [np.zeros((1, 4))]])
    # elif np.isclose(theta, np.pi, rtol = tol, atol = tol):
    #     r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    #     r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    #     r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]
    #     if not np.isclose(r22, -1., rtol = tol, atol = tol):
    #         multiplier = theta / np.sqrt(2. * (1. + r22))
    #         return multiplier * np.array([r02, r12, 1. + r22])
    #     elif not np.isclose(r11, -1., rtol = tol, atol = tol):
    #         multiplier = theta / np.sqrt(2. * (1. + r11))
    #         return multiplier * np.array([r01, 1. + r11, r21])
    #     elif not np.isclose(r00, -1., rtol = tol, atol = tol):
    #         multiplier = theta / np.sqrt(2. * (1. + r00))
    #         return multiplier * np.array([1. + r00, r10, r20])
    # v_hat = theta * (R - R.T) / (2. * np.sin(theta))
    # u = np.array([v_hat[2, 1], v_hat[0, 2], v_hat[1, 0]]) / theta
    # rho = Vinv(theta, u) @ t
    # tau_hat = np.block([[v_hat, rho.reshape(-1, 1)], [np.zeros((1, 4))]])
    # return tau_hat
    
def Exp(tau):
    rho, theta, u = decompose_cartesian_element(tau)
    u_hat = vec_hat(u)
    Exp_theta = np.eye(3) + np.sin(theta) * u_hat + (1 - np.cos(theta)) * u_hat @ u_hat
    M = np.block([[Exp_theta, V(theta, u) @ rho.reshape(-1, 1)], [np.zeros((1, 3)), 1.]])
    return M

def Log(M):
    R = M[:3, :3]
    t = M[:3, 3]
    theta = np.arccos(np.clip((np.trace(R) - 1.) / 2., -1.0, 1.0))
    if abs(theta) < tol:
        return compose_cartesian_element(t, 0., np.array([0., 0., 1.]))
    v_hat = theta * (R - R.T) / (2. * np.sin(theta))
    u = np.array([v_hat[2, 1], v_hat[0, 2], v_hat[1, 0]]) / theta
    rho = Vinv(theta, u) @ t
    return compose_cartesian_element(rho, theta, u)
    # R = M[:3, :3]
    # t = M[:3, 3]
    # theta = np.arccos(np.clip((np.trace(R) - 1.) / 2., -1.0, 1.0))
    # if np.isclose(theta, 0., rtol = tol, atol = tol):
    #     return compose_cartesian_element(t, theta, np.array([0., 0., 1.]))
    # elif np.isclose(theta, np.pi, rtol = tol, atol = tol):
    #     r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    #     r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    #     r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]
    #     if not np.isclose(r22, -1., rtol = tol, atol = tol):
    #         multiplier = theta / np.sqrt(2. * (1. + r22))
    #         return multiplier * np.array([r02, r12, 1. + r22])
    #     elif not np.isclose(r11, -1., rtol = tol, atol = tol):
    #         multiplier = theta / np.sqrt(2. * (1. + r11))
    #         return multiplier * np.array([r01, 1. + r11, r21])
    #     elif not np.isclose(r00, -1., rtol = tol, atol = tol):
    #         multiplier = theta / np.sqrt(2. * (1. + r00))
    #         return multiplier * np.array([1. + r00, r10, r20])
    # tau_hat = theta * (R - R.T) / (2. * np.sin(theta))
    # u = np.block([tau_hat[2, 1], tau_hat[0, 2], tau_hat[1, 0]]) / theta
    # rho = Vinv(theta, u) @ t
    # return compose_cartesian_element(rho, theta, u)

def plus_right(M, tau):
    return M @ Exp(tau)

def plus_left(M, tau):
    return Exp(tau) @ M

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

def jacobian_right(tau):
    rho, theta, u = decompose_cartesian_element(tau)
    Jl = Jl_mat(-theta, u)
    Q = Q_mat(-rho, -theta, u)
    return np.block([[Jl, Q], [np.zeros((3, 3)), Jl]])

def jacobian_right_inverse(tau):
    rho, theta, u = decompose_cartesian_element(tau)
    Jl_inv = Jl_inv_mat(-theta, u)
    Q = Q_mat(-rho, -theta, u)
    return np.block([[Jl_inv, -Jl_inv @ Q @ Jl_inv], [np.zeros((3, 3)), Jl_inv]])

def jacobian_left(tau):
    rho, theta, u = decompose_cartesian_element(tau)
    Jl = Jl_mat(theta, u)
    Q = Q_mat(rho, theta, u)
    return np.block([[Jl, Q], [np.zeros((3, 3)), Jl]])

def jacobian_left_inverse(tau):
    rho, theta, u = decompose_cartesian_element(tau)
    Jl_inv = Jl_inv_mat(theta, u)
    Q = Q_mat(rho, theta, u)
    return np.block([[Jl_inv, -Jl_inv @ Q @ Jl_inv], [np.zeros((3, 3)), Jl_inv]])

def jacobian_plus_right_1(M, tau):
    return adjoint(group_inverse(Exp(tau)))

def jacobian_plus_right_2(M, tau):
    return jacobian_right(tau)

def jacobian_minus_right_1(M1, M2):
    return -jacobian_left_inverse(minus_right(M1, M2))

def jacobian_minus_right_2(M1, M2):
    return jacobian_right_inverse(minus_right(M1, M2))

def jacobian_motion_action_1(M, p):
    R = M[:3, :3]
    return np.block([R, -R @ vec_hat(p)])

def jacobian_motion_action_2(M, p):
    R = M[:3, :3]
    return R

def vec_hat(v):
    return np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])

def V(theta, u):
    u_hat = vec_hat(u)
    return np.eye(3) + (1. - np.cos(theta)) / theta * u_hat + (theta - np.sin(theta)) / theta * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)

def Vinv(theta, u):
    return np.linalg.inv(V(theta, u)) if abs(theta) >= tol else np.eye(3)

def Jl_mat(theta, u):
    u_hat = vec_hat(u)
    return np.eye(3) + (1 - np.cos(theta)) / theta * u_hat + (theta - np.sin(theta)) / theta * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)

def Jl_inv_mat(theta, u):
    u_hat = vec_hat(u)
    return np.eye(3) - 0.5 * theta * u_hat + (1. - 0.5 * theta * (1. + np.cos(theta)) / np.sin(theta)) * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)

def Q_mat(rho, theta, u):
    rho_hat = vec_hat(rho)
    u_hat = vec_hat(u)
    if abs(theta) >= tol:
        Q = 0.5 * rho_hat + (theta - np.sin(theta)) / theta**2 * (u_hat @ rho_hat + rho_hat @ u_hat + theta * u_hat @ rho_hat @ u_hat) - \
            (1. - 0.5 * theta**2 - np.cos(theta)) / theta**2 * (u_hat @ u_hat @ rho_hat + rho_hat @ u_hat @ u_hat - 3. * u_hat @ rho_hat @ u_hat) - \
            0.5 * ((1. - theta**2 / 2. - np.cos(theta)) / theta - 3. * (theta - np.sin(theta) - theta**3 / 6.) / theta**2) * (u_hat @ rho_hat @ u_hat @ u_hat + u_hat @ u_hat @ rho_hat @ u_hat)
    else:
        Q = 0.5 * rho_hat
    return Q
