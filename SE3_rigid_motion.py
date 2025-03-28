import numpy as np


'''
theta * u is the angle-axis representation, where theta is the angle in radians and u is the 3d unit vector along the axis of rotation
[rho, theta * u]^T is the cartesian space element, where rho is the translation vector and theta * u is the angle-axis representation
M = [R, t; 0, 1] is the group element, where R \in R^{3x3} is the rotation matrix and t \in R^{3x1} is the translation vector
tau_hat = [theta * u_hat, rho; 0, 1] is the algebra element, where u_hat = [0, -u3, u2; u3, 0, -u1; -u2, u1, 0]
'''

def group_element(R, t):
    M = np.block([[R, t], [np.zeros((1, 3)), 1]])
    return M

def group_composition(M1, M2):
    return M1 @ M2

def group_inverse(M):
    R = M[:3, :3]
    t = M[:3, 3]
    return np.block([[R.T, -R.T @ t], [np.zeros((1, 3)), 1]])

def group_action(M, x):
    return M @ x

def algebra_element(rho, theta, u):
    v_hat = theta * np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    tau_hat = np.block([[v_hat, rho], [np.zeros((1, 4))]])
    return tau_hat


def testing():
    pass
