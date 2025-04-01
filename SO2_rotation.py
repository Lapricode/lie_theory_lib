import numpy as np


'''
theta is the cartesian space element, where theta is the angle in radians
R = [cos(theta), -sin(theta); sin(theta), cos(theta)] is the group element
theta_hat = [0, -theta; theta, 0] is the algebra element
'''

def group_element(theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R

def group_composition(R1, R2):
    return R1 @ R2

def group_inverse(R):
    return R.T

def group_action(R, x):
    return R @ x

def algebra_element(theta):
    theta_hat = np.array([[0., -theta], [theta, 0.]])
    return theta_hat

def compose_cartesian_element(theta):
    return theta

def decompose_cartesian_element(theta):
    return theta

def hat(theta):
    theta_hat = np.array([[0., -theta], [theta, 0.]])
    return theta_hat

def vee(theta_hat):
    theta = theta_hat[1, 0]
    return theta

def exp(theta_hat):
    theta = theta_hat[1, 0]
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R

def log(R):
    theta = np.arctan2(R[1, 0], R[0, 0])
    theta_hat = np.array([[0., -theta], [theta, 0.]])
    return theta_hat

def Exp(theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R

def Log(R):
    theta = np.arctan2(R[1, 0], R[0, 0])
    return theta

def plus_right(R, theta):
    return R @ Exp(theta)

def plus_left(R, theta):
    return Exp(theta) @ R

def minus_right(R1, R2):
    return Log(R2.T @ R1)

def minus_left(R1, R2):
    return Log(R1 @ R2.T)

def adjoint(R):
    return 1.

def jacobian_inverse(R):
    return -1.

def jacobian_composition_1(R1, R2):
    return 1.

def jacobian_composition_2(R1, R2):
    return 1.

def jacobian_right(theta):
    return 1.

def jacobian_left(theta):
    return 1.

def jacobian_plus_right_1(R, theta):
    return 1.

def jacobian_plus_right_2(R, theta):
    return 1.

def jacobian_minus_right_1(R1, R2):
    return 1.

def jacobian_minus_right_2(R1, R2):
    return -1.

def jacobian_rotation_action_1(R, v):
    return R @ np.array([[0, -1], [1, 0]]) @ v

def jacobian_rotation_action_2(R, v):
    return R


def testing(R, theta):
    R = group_element(np.pi / 10)
    theta = np.pi / 5
    adjoint_R_theta = adjoint(R, theta)
    print(theta, vee(adjoint_R_theta))
