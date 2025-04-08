import numpy as np


'''
tau = theta * u (angle-axis representation) is the cartesian space element, where theta \in R is the angle in radians and u \in R^3 unit vector along the axis of rotation
R = I + sin(theta) * u_hat + (1 - cos(theta)) * u_hat^2 is the group element
tau_hat = theta * u_hat is the algebra element, where u_hat = [0, -u3, u2; u3, 0, -u1; -u2, u1, 0]
'''
tol = 1e-5  # tolerance for numerical issues

def group_element(tau):
    theta, u = decompose_cartesian_element(tau)
    u_hat = vec_hat(u)
    R = np.eye(3) + np.sin(theta) * u_hat + (1 - np.cos(theta)) * u_hat @ u_hat
    return R

def group_composition(R1, R2):
    return R1 @ R2

def group_inverse(R):
    return R.T

def group_action(R, x):
    return R @ x

def algebra_element(tau):
    theta, u = decompose_cartesian_element(tau)
    tau_hat = theta * vec_hat(u)
    return tau_hat

def compose_cartesian_element(theta, u):
    return theta * u

def decompose_cartesian_element(tau):
    theta = np.linalg.norm(tau)
    u = tau / theta if abs(theta) >= tol else np.array([0., 0., 1.])
    return theta, u

def hat(tau):
    theta, u = decompose_cartesian_element(tau)
    tau_hat = theta * vec_hat(u)
    return tau_hat

def vee(tau_hat):
    tau = np.block([tau_hat[2, 1], tau_hat[0, 2], tau_hat[1, 0]])
    theta = np.linalg.norm(tau)
    u = tau / theta if abs(theta) >= tol else np.array([0., 0., 1.])
    return compose_cartesian_element(theta, u)

def exp(tau_hat):
    tau = np.block([tau_hat[2, 1], tau_hat[0, 2], tau_hat[1, 0]])
    theta = np.linalg.norm(tau)
    u = tau / theta if abs(theta) >= tol else np.array([0., 0., 1.])
    u_hat = vec_hat(u)
    R = np.eye(3) + np.sin(theta) * u_hat + (1 - np.cos(theta)) * u_hat @ u_hat
    return R

def log(R):
    theta = np.arccos((np.trace(R) - 1.) / 2.)
    tau_hat = theta * (R - R.T) / (2. * np.sin(theta)) if abs(theta) >= tol else (R - R.T) / 2.
    return tau_hat

def Exp(tau):
    theta, u = decompose_cartesian_element(tau)
    u_hat = vec_hat(u)
    R = np.eye(3) + np.sin(theta) * u_hat + (1. - np.cos(theta)) * u_hat @ u_hat
    return R

def Log(R):
    theta = np.arccos((np.trace(R) - 1.) / 2.)
    tau_hat = theta * (R - R.T) / (2. * np.sin(theta)) if abs(theta) >= tol else (R - R.T) / 2.
    tau = np.block([tau_hat[2, 1], tau_hat[0, 2], tau_hat[1, 0]])
    u = tau / theta if abs(theta) >= tol else np.array([0., 0., 1.])
    return compose_cartesian_element(theta, u)

def plus_right(R, tau):
    return R @ Exp(tau)

def plus_left(R, tau):
    return Exp(tau) @ R

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

def jacobian_right(tau):
    theta, u = decompose_cartesian_element(tau)
    u_hat = vec_hat(u)
    return np.eye(3) - (1 - np.cos(theta)) / theta * u_hat + (theta - np.sin(theta)) / theta * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)

def jacobian_right_inverse(tau):
    theta, u = decompose_cartesian_element(tau)
    u_hat = vec_hat(u)
    return np.eye(3) + 0.5 * theta * u_hat + (1. - 0.5 * theta * (1. + np.cos(theta)) / np.sin(theta)) * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)

def jacobian_left(tau):
    theta, u = decompose_cartesian_element(tau)
    u_hat = vec_hat(u)
    return np.eye(3) + (1 - np.cos(theta)) / theta * u_hat + (theta - np.sin(theta)) / theta * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)

def jacobian_left_inverse(tau):
    theta, u = decompose_cartesian_element(tau)
    u_hat = vec_hat(u)
    return np.eye(3) - 0.5 * theta * u_hat + (1. - 0.5 * theta * (1. + np.cos(theta)) / np.sin(theta)) * u_hat @ u_hat if abs(theta) >= tol else np.eye(3)

def jacobian_plus_right_1(R, tau):
    theta, u = decompose_cartesian_element(tau)
    u_hat = vec_hat(u)
    R = np.eye(3) + np.sin(theta) * u_hat + (1. - np.cos(theta)) * u_hat @ u_hat
    return R.T

def jacobian_plus_right_2(R, tau):
    return jacobian_right(tau)

def jacobian_minus_right_1(R1, R2):
    return jacobian_right_inverse(Log(R2.T @ R1))

def jacobian_minus_right_2(R1, R2):
    return -jacobian_left_inverse(Log(R2.T @ R1))

def jacobian_rotation_action_1(R, x):
    return -R @ vec_hat(x)

def jacobian_rotation_action_2(R, x):
    return R

def vec_hat(v):
    return np.block([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])


def printing(tau1, tau2, action_vec):
    R1 = group_element(tau1)
    R2 = group_element(tau2)
    theta1_hat = algebra_element(tau1)
    cartesian1 = decompose_cartesian_element(tau1)
    print(f"tau1:\n {tau1}")
    print(f"tau2:\n {tau2}")
    print(f"action_vec:\n {action_vec}")
    print(f"R1:\n {R1}")
    print(f"R2:\n {R2}")
    print(f"theta1_hat:\n {theta1_hat}")
    print(f"cartesian1:\n {cartesian1}")
    print(f"inverse_R1:\n {group_inverse(R1)}")
    print(f"action_R1:\n {group_action(R1, action_vec)}")
    print(f"composition_R1_R2:\n {group_composition(R1, R2)}")
    print(f"right_plus_R1_tau1:\n {plus_right(R1, tau1)}")
    print(f"left_plus_R1_tau1:\n {plus_left(R1, tau1)}")
    print(f"right_minus_R1_R2:\n {minus_right(R1, R2)}")
    print(f"left_minus_R1_R2:\n {minus_left(R1, R2)}")
    print(f"adjoint_R1:\n {adjoint(R1)}")
    print(f"jacobian_inverse_R1:\n {jacobian_inverse(R1)}")
    print(f"jacobian_composition_1_R1_R2:\n {jacobian_composition_1(R1, R2)}")
    print(f"jacobian_composition_2_R1_R2:\n {jacobian_composition_2(R1, R2)}")
    print(f"jacobian_right_tau1:\n {jacobian_right(tau1)}")
    print(f"jacobian_right_inverse_tau1:\n {jacobian_right_inverse(tau1)}")
    print(f"jacobian_left_tau1:\n {jacobian_left(tau1)}")
    print(f"jacobian_left_inverse_tau1:\n {jacobian_left_inverse(tau1)}")
    print(f"jacobian_plus_right_1_R1_tau1:\n {jacobian_plus_right_1(R1, tau1)}")
    print(f"jacobian_plus_right_2_R1_tau1:\n {jacobian_plus_right_2(R1, tau1)}")
    print(f"jacobian_minus_right_1_R1_R2:\n {jacobian_minus_right_1(R1, R2)}")
    print(f"jacobian_minus_right_2_R1_R2:\n {jacobian_minus_right_2(R1, R2)}")
    print(f"jacobian_rotation_action_1_R1:\n {jacobian_rotation_action_1(R1, action_vec)}")
    print(f"jacobian_rotation_action_2_R1:\n {jacobian_rotation_action_2(R1, action_vec)}")

def testing(tau1, tau2, action_vec):
    R1 = group_element(tau1)
    R2 = group_element(tau2)
    theta1_hat = algebra_element(tau1)
    cartesian1_1, cartesian1_2 = decompose_cartesian_element(tau1)
    inverse_R1 = group_inverse(R1)
    action_R1 = group_action(R1, action_vec)
    composition_R1_R2 = group_composition(R1, R2)
    right_plus_R1_tau1 = plus_right(R1, tau1)
    left_plus_R1_tau1 = plus_left(R1, tau1)
    right_minus_R1_R2 = minus_right(R1, R2)
    left_minus_R1_R2 = minus_left(R1, R2)
    adjoint_R1 = adjoint(R1)
    jacobian_inverse_R1 = jacobian_inverse(R1)
    jacobian_composition_1_R1_R2 = jacobian_composition_1(R1, R2)
    jacobian_composition_2_R1_R2 = jacobian_composition_2(R1, R2)
    jacobian_right_tau1 = jacobian_right(tau1)
    jacobian_right_inverse_tau1 = jacobian_right_inverse(tau1)
    jacobian_left_tau1 = jacobian_left(tau1)
    jacobian_left_inverse_tau1 = jacobian_left_inverse(tau1)
    jacobian_plus_right_1_R1_tau1 = jacobian_plus_right_1(R1, tau1)
    jacobian_plus_right_2_R1_tau1 = jacobian_plus_right_2(R1, tau1)
    jacobian_minus_right_1_R1_R2 = jacobian_minus_right_1(R1, R2)
    jacobian_minus_right_2_R1_R2 = jacobian_minus_right_2(R1, R2)
    jacobian_rotation_action_1_R1 = jacobian_rotation_action_1(R1, action_vec)
    jacobian_rotation_action_2_R1 = jacobian_rotation_action_2(R1, action_vec)
    assert np.allclose(R1, exp(theta1_hat), atol = 1e-10)
    assert np.allclose(R1, Exp(tau1), atol = 1e-10)
    assert np.allclose(tau1, compose_cartesian_element(cartesian1_1, cartesian1_2), atol = 1e-10)
    assert np.allclose(theta1_hat, hat(tau1), atol = 1e-10)
    assert np.allclose(tau1, vee(theta1_hat), atol = 1e-5)
    assert np.allclose(tau1, Log(R1), atol = 1e-10)
    assert np.allclose(theta1_hat, log(R1), atol = 1e-10)
    assert np.allclose(right_plus_R1_tau1, left_plus_R1_tau1, atol = 1e-10)
    assert np.allclose(adjoint(Exp(tau1)), jacobian_left(tau1) @ jacobian_right_inverse(tau1), atol = 1e-10)
    print("\nAll tests passed!")

def run_test_example():
    np.random.seed(0)
    tau1 = np.random.randn(3,)
    tau2 = np.random.randn(3,)
    action_vec = np.random.randn(3, 1)
    printing(tau1, tau2, action_vec)
    testing(tau1, tau2, action_vec)

run_test_example()
