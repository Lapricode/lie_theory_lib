import numpy as np


'''
[rho, theta]^T is the cartesian space element, where rho is the translation vector and theta is the rotation angle in radians'
'''

def group_element(R, t):
    return np.block([[R, t], [0, 0, 0, 1]])

def group_element_inverse():
    return

# def group_composition(R1, R2):
#     return R1 @ R2

# def group_inverse(R):
#     return R.T

# def group_action(R, x):
#     return R @ x

# def algebra_element(theta, u):
#     tavec_hat = theta * np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
#     return tavec_hat

# def hat(theta, u):
#     tavec_hat = theta * np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
#     return tavec_hat

# def vee(v_hat):
#     v = np.array([v_hat[2, 1], v_hat[0, 2], v_hat[1, 0]])
#     theta = np.linalg.norm(v)
#     u = v / theta
#     return theta, u

# def Exp(theta, u):
#     R = np.eye(3) + np.sin(theta) * vec_hat(u) + (1. - np.cos(theta)) * vec_hat(u) @ vec_hat(u)
#     return R

# def Log(R):
#     theta = np.arccos((np.trace(R) - 1.) / 2.)
#     vec_hat = (R - R.T) / (2. * np.sin(theta))
#     u = vee(vec_hat)
#     return theta, u

# def plus_right(R, theta, u):
#     return R @ Exp(theta, u)

# def plus_left(R, theta, u):
#     return Exp(theta, u) @ R

# def minus_right(R1, R2):
#     return Log(R2.T @ R1)

# def minus_left(R1, R2):
#     return Log(R1 @ R2.T)


def testing():
    pass