import numpy as np


''' theta * u is the cartesian space element '''

def group_element(theta, u):
    R = np.eye(3) + np.sin(theta) * u_hat(u) + (1 - np.cos(theta)) * u_hat(u) @ u_hat(u)
    return R

def group_composition(R1, R2):
    return R1 @ R2

def group_inverse(R):
    return R.T

def group_action(R, x):
    return R @ x

def algebra_element(theta, u):
    tau_hat = theta * np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return tau_hat

def hat(theta, u):
    tau_hat = theta * np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return tau_hat

def vee(v_hat):
    v = np.array([v_hat[2, 1], v_hat[0, 2], v_hat[1, 0]])
    theta = np.linalg.norm(v)
    u = v / theta
    return theta, u

def Exp(theta, u):
    R = np.eye(3) + np.sin(theta) * u_hat(u) + (1. - np.cos(theta)) * u_hat(u) @ u_hat(u)
    return R

def Log(R):
    theta = np.arccos((np.trace(R) - 1.) / 2.)
    u_hat = (R - R.T) / (2. * np.sin(theta))
    u = vee(u_hat)
    return theta, u

def plus_right(R, theta, u):
    return R @ Exp(theta, u)

def plus_left(R, theta, u):
    return Exp(theta, u) @ R

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

def jacobian_right(theta, u):
    return np.eye(3) - (1 - np.cos(theta)) / theta * u_hat(u) + (theta - np.sin(theta)) / theta * u_hat(u) @ u_hat(u)

def jacobian_left(theta, u):
    return np.eye(3) + (1 - np.cos(theta)) / theta * u_hat(u) + (theta - np.sin(theta)) / theta * u_hat(u) @ u_hat(u)

def jacobian_plus_right_1(R, theta, u):
    return

def jacobian_plus_right_2(R, theta, u):
    return

def jacobian_minus_right_1(R1, R2):
    return

def jacobian_minus_right_2(R1, R2):
    return

def u_hat(u):
    return np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])


def testing():
    pass
