import numpy as np


''' theta is the cartesian space element '''

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
    tau_hat = np.array([[0, -theta], [theta, 0]])
    return tau_hat

def hat(theta):
    tau_hat = np.array([[0, -theta], [theta, 0]])
    return tau_hat

def vee(tau_hat):
    theta = tau_hat[1, 0]
    return theta

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


def testing(R, theta):
    # R = group_element(np.pi / 10)
    # theta = np.pi / 5
    # adjoint_R_theta = adjoint(R, theta)
    # print(theta, vee(adjoint_R_theta))
    pass
