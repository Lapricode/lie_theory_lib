import numpy as np


'''
v = theta * u (angle-axis representation) is the cartesian space element, where theta is the angle in radians and u is the 3d unit vector along the axis of rotation
R = I + sin(theta) * u_hat + (1 - cos(theta)) * u_hat^2 is the group element
v_hat = theta * u_hat is the algebra element, where u_hat = [0, -u3, u2; u3, 0, -u1; -u2, u1, 0]
'''
tol = 1e-5  # tolerance for numerical issues

def group_element(theta, u):
    u_hat = vec_hat(u)
    R = np.eye(3) + np.sin(theta) * u_hat + (1 - np.cos(theta)) * u_hat @ u_hat
    return R

def group_composition(R1, R2):
    return R1 @ R2

def group_inverse(R):
    return R.T

def group_action(R, x):
    return R @ x

def algebra_element(theta, u):
    v_hat = theta * vec_hat(u)
    return v_hat

def hat(theta, u):
    v_hat = theta * vec_hat(u)
    return v_hat

def vee(v_hat):
    v = np.array([v_hat[2, 1], v_hat[0, 2], v_hat[1, 0]])
    theta = np.linalg.norm(v)
    u = v / theta if abs(theta) >= tol else np.array([[0.], [0.], [1.]])
    return theta, u

def exp(v_hat):
    v = np.array([v_hat[2, 1], v_hat[0, 2], v_hat[1, 0]])
    theta = np.linalg.norm(v)
    u = v / theta if abs(theta) >= tol else np.array([[0.], [0.], [1.]])
    u_hat = vec_hat(u)
    R = np.eye(3) + np.sin(theta) * u_hat + (1 - np.cos(theta)) * u_hat @ u_hat
    return R

def log(R):
    theta = np.arccos((np.trace(R) - 1.) / 2.)
    v_hat = theta * (R - R.T) / (2. * np.sin(theta)) if abs(theta) >= tol else (R - R.T) / 2.
    return v_hat

def Exp(theta, u):
    u_hat = vec_hat(u)
    R = np.eye(3) + np.sin(theta) * u_hat + (1. - np.cos(theta)) * u_hat @ u_hat
    return R

def Log(R):
    theta = np.arccos((np.trace(R) - 1.) / 2.)
    v_hat = theta * (R - R.T) / (2. * np.sin(theta)) if abs(theta) >= tol else (R - R.T) / 2.
    v = np.array([v_hat[2, 1], v_hat[0, 2], v_hat[1, 0]])
    u = v / theta if abs(theta) >= tol else np.array([[0.], [0.], [1.]])
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
    u_hat = vec_hat(u)
    return np.eye(3) - (1 - np.cos(theta)) / theta * u_hat + (theta - np.sin(theta)) / theta * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)

def jacobian_right_inverse(theta, u):
    u_hat = vec_hat(u)
    return np.eye(3) + 0.5 * theta * u_hat + (1. - 0.5 * theta * (1. + np.cos(theta)) / np.sin(theta)) * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)

def jacobian_left(theta, u):
    u_hat = vec_hat(u)
    return np.eye(3) + (1 - np.cos(theta)) / theta * u_hat + (theta - np.sin(theta)) / theta * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)

def jacobian_left_inverse(theta, u):
    u_hat = vec_hat(u)
    return np.eye(3) - 0.5 * theta * u_hat + (1. - 0.5 * theta * (1. + np.cos(theta)) / np.sin(theta)) * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)

def jacobian_plus_right_1(R, theta, u):
    u_hat = vec_hat(u)
    R = np.eye(3) + np.sin(theta) * u_hat + (1. - np.cos(theta)) * u_hat @ u_hat
    return R.T

def jacobian_plus_right_2(R, theta, u):
    return jacobian_right(theta, u)

def jacobian_minus_right_1(R1, R2):
    theta, u = Log(R2.T @ R1)
    return jacobian_right_inverse(theta, u)

def jacobian_minus_right_2(R1, R2):
    theta, u = Log(R2.T @ R1)
    return -jacobian_left_inverse(theta, u)

def jacobian_rotation_action_1(R, v):
    return -R @ vec_hat(v)

def jacobian_rotation_action_2(R, v):
    return R

def vec_hat(v):
    return np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])


def testing():
    pass
