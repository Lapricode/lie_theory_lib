import numpy as np


'''
tau = [rho; theta * u] is the cartesian space element, where rho \in R^{3x1} is the translation vector and theta * u \in R^{3x1} is the angle-axis representation
M = [R, t; 0, 1] is the group element, where R \in R^{3x3} is the rotation matrix and t \in R^{3x1} is the translation vector
tau_hat = [theta * u_hat, rho; 0, 1] is the algebra element, where u_hat = [0, -u3, u2; u3, 0, -u1; -u2, u1, 0]
'''
tol = 1e-5  # tolerance for numerical issues

def group_element(rho, theta, u):
    u_hat = vec_hat(u)
    R = np.eye(3) + np.sin(theta) * u_hat + (1 - np.cos(theta)) * u_hat @ u_hat
    t = V(theta) @ rho
    M = np.block([[R, t], [np.zeros((1, 3)), 1.]])
    return M

def group_composition(M1, M2):
    return M1 @ M2

def group_inverse(M):
    R = M[:3, :3]
    t = M[:3, 3]
    return np.block([[R.T, -R.T @ t], [np.zeros((1, 3)), 1.]])

def group_action(M, x):
    return M @ x

def algebra_element(rho, theta, u):
    v_hat = theta * vec_hat(u)
    tau_hat = np.block([[v_hat, rho], [np.zeros((1, 4))]])
    return tau_hat

def hat(rho, theta, u):
    v_hat = theta * vec_hat(u)
    tau_hat = np.block([[v_hat, rho], [np.zeros((1, 4))]])
    return tau_hat

def vee(tau_hat):
    rho = tau_hat[:3, 3]
    v = np.array([tau_hat[2, 1], tau_hat[0, 2], tau_hat[1, 0]])
    theta = np.linalg.norm(v)
    u = v / theta if abs(theta) >= tol else np.array([[0.], [0.], [1.]])
    return rho, theta, u

def exp(tau_hat):
    return

def log(M):
    return

def Exp(rho, theta, u):
    return

def Log(M):
    return

def plus_right(M, rho, theta, u):
    return M @ Exp(rho, theta, u)

def plus_left(M, rho, theta, u):
    return Exp(rho, theta, u) @ M

def minus_right(M1, M2):
    return Log(M2.T @ M1)

def minus_left(M1, M2):
    return Log(M1 @ M2.T)

def adjoint(M):
    return

# def jacobian_inverse(M):
#     return -adjoint(M)

# def jacobian_composition_1(M1, M2):
#     M2_inv = group_inverse(M2)
#     return adjoint(M2_inv)

# def jacobian_composition_2(R1, R2):
#     return np.eye(3)

# def jacobian_plus_right_1(M, rho, theta, u):
#     return

# def jacobian_plus_right_2(M, rho, theta, u):
#     return

# def jacobian_minus_right_1(M1, M2):
#     return

# def jacobian_minus_right_2(M1, M2):
#     return


def vec_hat(v):
    return np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])

def V(theta, u):
    u_hat = vec_hat(u)
    return np.eye(3) + (1. - np.cos(theta)) / theta * u_hat + (theta - np.sin(theta)) / theta * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)

def Vinv(theta, u):
    return np.linalg.inv(V(theta, u)) if abs(theta) >= tol else np.eye(3)


def testing():
    pass
