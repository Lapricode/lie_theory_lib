import numpy as np


'''
theta is the cartesian space element, where theta is the angle in radians
R = [cos(theta), -sin(theta); sin(theta), cos(theta)] is the group element
theta_hat = [0, -theta; theta, 0] is the algebra element
'''

def group_element(theta):
    R = np.block([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R

def group_composition(R1, R2):
    return R1 @ R2

def group_inverse(R):
    return R.T

def group_action(R, x):
    return R @ x

def algebra_element(theta):
    theta_hat = np.block([[0., -theta], [theta, 0.]])
    return theta_hat

def compose_cartesian_element(theta):
    return theta

def decompose_cartesian_element(theta):
    return theta

def hat(theta):
    theta_hat = np.block([[0., -theta], [theta, 0.]])
    return theta_hat

def vee(theta_hat):
    theta = theta_hat[1, 0]
    return theta

def exp(theta_hat):
    theta = theta_hat[1, 0]
    R = np.block([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R

def log(R):
    theta = np.arctan2(R[1, 0], R[0, 0])
    theta_hat = np.array([[0., -theta], [theta, 0.]])
    return theta_hat

def Exp(theta):
    R = np.block([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
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

def jacobian_right_inverse(theta):
    return 1.

def jacobian_left(theta):
    return 1.

def jacobian_left_inverse(theta):
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


def printing(theta1, theta2, action_vec):
    R1 = group_element(theta1)
    R2 = group_element(theta2)
    theta1_hat = algebra_element(theta1)
    cartesian1 = decompose_cartesian_element(theta1)
    print(f"theta1:\n {theta1}")
    print(f"theta2:\n {theta2}")
    print(f"action_vec:\n {action_vec}")
    print(f"R1:\n {R1}")
    print(f"R2:\n {R2}")
    print(f"theta1_hat:\n {theta1_hat}")
    print(f"cartesian1:\n {cartesian1}")
    print(f"inverse_R1:\n {group_inverse(R1)}")
    print(f"action_R1:\n {group_action(R1, action_vec)}")
    print(f"composition_R1_R2:\n {group_composition(R1, R2)}")
    print(f"right_plus_R1_theta1:\n {plus_right(R1, theta1)}")
    print(f"left_plus_R1_theta1:\n {plus_left(R1, theta1)}")
    print(f"right_minus_R1_R2:\n {minus_right(R1, R2)}")
    print(f"left_minus_R1_R2:\n {minus_left(R1, R2)}")
    print(f"adjoint_R1:\n {adjoint(R1)}")
    print(f"jacobian_inverse_R1:\n {jacobian_inverse(R1)}")
    print(f"jacobian_composition_1_R1_R2:\n {jacobian_composition_1(R1, R2)}")
    print(f"jacobian_composition_2_R1_R2:\n {jacobian_composition_2(R1, R2)}")
    print(f"jacobian_right_theta1:\n {jacobian_right(theta1)}")
    print(f"jacobian_right_inverse_theta1:\n {jacobian_right_inverse(theta1)}")
    print(f"jacobian_left_theta1:\n {jacobian_left(theta1)}")
    print(f"jacobian_left_inverse_theta1:\n {jacobian_left_inverse(theta1)}")
    print(f"jacobian_plus_right_1_R1_theta1:\n {jacobian_plus_right_1(R1, theta1)}")
    print(f"jacobian_plus_right_2_R1_theta1:\n {jacobian_plus_right_2(R1, theta1)}")
    print(f"jacobian_minus_right_1_R1_R2:\n {jacobian_minus_right_1(R1, R2)}")
    print(f"jacobian_minus_right_2_R1_R2:\n {jacobian_minus_right_2(R1, R2)}")
    print(f"jacobian_rotation_action_1_R1:\n {jacobian_rotation_action_1(R1, action_vec)}")
    print(f"jacobian_rotation_action_2_R1:\n {jacobian_rotation_action_2(R1, action_vec)}")

def testing(theta1, theta2, action_vec):
    R1 = group_element(theta1)
    R2 = group_element(theta2)
    theta1_hat = algebra_element(theta1)
    cartesian1 = decompose_cartesian_element(theta1)
    inverse_R1 = group_inverse(R1)
    action_R1 = group_action(R1, action_vec)
    composition_R1_R2 = group_composition(R1, R2)
    right_plus_R1_theta1 = plus_right(R1, theta1)
    left_plus_R1_theta1 = plus_left(R1, theta1)
    right_minus_R1_R2 = minus_right(R1, R2)
    left_minus_R1_R2 = minus_left(R1, R2)
    adjoint_R1 = adjoint(R1)
    assert np.allclose(R1, exp(theta1_hat), atol = 1e-10)
    assert np.allclose(R1, Exp(theta1), atol = 1e-10)
    assert np.allclose(theta1, compose_cartesian_element(cartesian1), atol = 1e-10)
    assert np.allclose(theta1_hat, hat(theta1), atol = 1e-10)
    assert np.allclose(theta1, vee(theta1_hat), atol = 1e-10)
    assert np.allclose(theta1, Log(R1), atol = 1e-10)
    assert np.allclose(theta1_hat, log(R1), atol = 1e-10)
    assert np.allclose(right_plus_R1_theta1, left_plus_R1_theta1, atol = 1e-10)
    print("\nAll tests passed!")

def run_test_example():
    np.random.seed(0)
    theta1 = np.random.randn(1)
    theta2 = np.random.randn(1)
    action_vec = np.random.randn(2, 1)
    printing(theta1, theta2, action_vec)
    # testing(theta1, theta2, action_vec)

run_test_example()
