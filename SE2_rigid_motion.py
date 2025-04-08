import numpy as np


'''
tau = [rho; theta] is the cartesian space element, where rho \in R^2 is the translation vector and theta \in R is the rotation angle in radians
M = [R, t; 0, 1] is the group element, where R \in R^{2x2} is the rotation matrix and t \in R^2 is the translation vector
tau_hat = [theta_hat, rho; 0, 1] is the algebra element, where theta_hat = [0, -theta; theta, 0]
'''
tol = 1e-5  # tolerance for numerical issues

def group_element(tau):
    rho, theta = decompose_cartesian_element(tau)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = V(theta) @ rho
    M = np.block([[R, t.reshape(-1, 1)], [np.zeros((1, 2)),  1.]])
    return M

def group_composition(M1, M2):
    return M1 @ M2

def group_inverse(M):
    R = M[:2, :2]
    t = M[:2, 2]
    return np.block([[R.T, -R.T @ t.reshape(-1, 1)], [np.zeros((1, 2)), 1]])

def group_action(M, x):
    return M @ x

def algebra_element(tau):
    rho, theta = decompose_cartesian_element(tau)
    theta_hat = np.array([[0., -theta], [theta, 0.]])
    tau_hat = np.block([[theta_hat, rho.reshape(-1, 1)], [np.zeros((1, 3))]])
    return tau_hat

def compose_cartesian_element(rho, theta):
    return np.block([rho, theta])

def decompose_cartesian_element(tau):
    rho = tau[:2].reshape(-1,)
    theta = tau[2]
    return rho, theta

def hat(tau):
    rho, theta = decompose_cartesian_element(tau)
    theta_hat = np.array([[0., -theta], [theta, 0.]])
    tau_hat = np.block([[theta_hat, rho.reshape(-1, 1)], [np.zeros((1, 3))]])
    return tau_hat

def vee(tau_hat):
    rho = tau_hat[:2, 2]
    theta = tau_hat[1, 0]
    return compose_cartesian_element(rho, theta)

def exp(tau_hat):
    rho = tau_hat[:2, 2]
    theta = tau_hat[1, 0]
    Exp_theta = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    M = np.block([[Exp_theta, V(theta) @ rho.reshape(-1, 1)], [np.zeros((1, 2)),  1.]])
    return M

def log(M):
    R = M[:2, :2]
    t = M[:2, 2]
    theta = np.arctan2(R[1, 0], R[0, 0])
    rho = Vinv(theta) @ t
    theta_hat = np.array([[0., -theta], [theta, 0.]])
    tau_hat = np.block([[theta_hat, rho.reshape(-1, 1)], [np.zeros((1, 3))]])
    return tau_hat

def Exp(tau):
    rho, theta = decompose_cartesian_element(tau)
    Exp_theta = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    M = np.block([[Exp_theta, V(theta) @ rho.reshape(-1, 1)], [np.zeros((1, 2)),  1.]])
    return M

def Log(M):
    R = M[:2, :2]
    t = M[:2, 2]
    theta = np.arctan2(R[1, 0], R[0, 0])
    rho = Vinv(theta) @ t
    return compose_cartesian_element(rho, theta)

def plus_right(M, tau):
    return M @ Exp(tau)

def plus_left(M, tau):
    return Exp(tau) @ M

def minus_right(M1, M2):
    return Log(group_inverse(M2) @ M1)

def minus_left(M1, M2):
    return Log(M1 @ group_inverse(M2))

def adjoint(M):
    R = M[:2, :2]
    t = M[:2, 2]
    return np.block([[R, -np.array([[0., -1.], [1., 0.]]) @ t.reshape(-1, 1)], [np.zeros((1, 2)), 1.]])

def jacobian_inverse(M):
    return -adjoint(M)

def jacobian_composition_1(M1, M2):
    M2_inv = group_inverse(M2)
    return adjoint(M2_inv)

def jacobian_composition_2(R1, R2):
    return np.eye(3)

def jacobian_right(tau):
    rho, theta = decompose_cartesian_element(tau)
    rho1, rho2 = rho[0], rho[1]
    if abs(theta) >= tol:
        Jr = np.array([[np.sin(theta) / theta, (1. - np.cos(theta)) / theta, (theta * rho1 - rho2 + rho2 * np.cos(theta) - rho1 * np.sin(theta)) / theta**2], \
                        [(np.cos(theta) - 1.) / theta, np.sin(theta) / theta, (rho1 + theta * rho2 - rho1 * np.cos(theta) - rho2 * np.sin(theta)) / theta**2], \
                        [0., 0., 1.]])
    else:
        Jr = np.block([[np.eye(2), np.array([[-rho2 / 2.], [rho1 / 2.]])], [np.zeros((1, 2)), 1.]])
    return Jr

def jacobian_right_inverse(tau):
    Jrinv = np.linalg.inv(jacobian_right(tau))
    return Jrinv

def jacobian_left(tau):
    rho, theta = decompose_cartesian_element(tau)
    rho1, rho2 = rho[0], rho[1]
    if abs(theta) >= tol:
        Jl = np.array([[np.sin(theta) / theta, (np.cos(theta) - 1.) / theta, (theta * rho1 + rho2 - rho2 * np.cos(theta) - rho1 * np.sin(theta)) / theta**2], \
                        [(1. - np.cos(theta)) / theta, np.sin(theta) / theta, (-rho1 + theta * rho2 + rho1 * np.cos(theta) - rho2 * np.sin(theta)) / theta**2], \
                        [0., 0., 1.]])
    else:
        Jl = np.block([[np.eye(2), np.array([[rho2 / 2.], [-rho1 / 2.]])], [np.zeros((1, 2)), 1.]])
    return Jl

def jacobian_left_inverse(tau):
    Jlinv = np.linalg.inv(jacobian_left(tau))
    return Jlinv

def jacobian_plus_right_1(M, tau):
    return adjoint(group_inverse(Exp(tau)))

def jacobian_plus_right_2(M, tau):
    return jacobian_right(tau)

def jacobian_minus_right_1(M1, M2):
    return -jacobian_left_inverse(minus_right(M1, M2))

def jacobian_minus_right_2(M1, M2):
    return jacobian_right_inverse(minus_right(M1, M2))

def jacobian_motion_action_1(M, p):
    R = M[:2, :2]
    return np.block([R, R @ np.array([[0., -1.], [1., 0.]]) @ p[:2].reshape(-1, 1)])

def jacobian_motion_action_2(M, p):
    R = M[:2, :2]
    return R

def V(theta):
    return np.sin(theta) / theta * np.eye(2) + (1 - np.cos(theta)) / theta * np.array([[0, -1], [1, 0]]) if abs(theta) >= tol else np.eye(2)

def Vinv(theta):
    return theta / (2 * (1 - np.cos(theta))) * (np.sin(theta) * np.eye(2) - (1 - np.cos(theta)) * np.array([[0, -1], [1, 0]])) if abs(theta) >= tol else np.eye(2)


def printing(tau1, tau2, action_vec):
    M1 = group_element(tau1)
    M2 = group_element(tau2)
    tau1_hat = algebra_element(tau1)
    cartesian1 = decompose_cartesian_element(tau1)
    print(f"tau1:\n {tau1}")
    print(f"tau2:\n {tau2}")
    print(f"action_vec:\n {action_vec}")
    print(f"M1:\n {M1}")
    print(f"M2:\n {M2}")
    print(f"tau1_hat:\n {tau1_hat}")
    print(f"cartesian1:\n {cartesian1}")
    print(f"inverse_M1:\n {group_inverse(M1)}")
    print(f"action_M1:\n {group_action(M1, action_vec)}")
    print(f"composition_M1_M2:\n {group_composition(M1, M2)}")
    print(f"right_plus_M1_tau1:\n {plus_right(M1, tau1)}")
    print(f"left_plus_M1_tau1:\n {plus_left(M1, tau1)}")
    print(f"right_minus_M1_M2:\n {minus_right(M1, M2)}")
    print(f"left_minus_M1_M2:\n {minus_left(M1, M2)}")
    print(f"adjoint_M1:\n {adjoint(M1)}")
    print(f"jacobian_inverse_M1:\n {jacobian_inverse(M1)}")
    print(f"jacobian_composition_1_M1_M2:\n {jacobian_composition_1(M1, M2)}")
    print(f"jacobian_composition_2_M1_M2:\n {jacobian_composition_2(M1, M2)}")
    print(f"jacobian_right_tau1:\n {jacobian_right(tau1)}")
    print(f"jacobian_right_inverse_tau1:\n {jacobian_right_inverse(tau1)}")
    print(f"jacobian_left_tau1:\n {jacobian_left(tau1)}")
    print(f"jacobian_left_inverse_tau1:\n {jacobian_left_inverse(tau1)}")
    print(f"jacobian_plus_right_1_M1_tau1:\n {jacobian_plus_right_1(M1, tau1)}")
    print(f"jacobian_plus_right_2_M1_tau1:\n {jacobian_plus_right_2(M1, tau1)}")
    print(f"jacobian_minus_right_1_M1_M2:\n {jacobian_minus_right_1(M1, M2)}")
    print(f"jacobian_minus_right_2_M1_M2:\n {jacobian_minus_right_2(M1, M2)}")
    print(f"jacobian_motion_action_1_M1:\n {jacobian_motion_action_1(M1, action_vec)}")
    print(f"jacobian_motion_action_2_M1:\n {jacobian_motion_action_2(M1, action_vec)}")

def testing(tau1, tau2, action_vec):
    M1 = group_element(tau1)
    M2 = group_element(tau2)
    tau1_hat = algebra_element(tau1)
    cartesian1_1, cartesian1_2 = decompose_cartesian_element(tau1)
    inverse_M1 = group_inverse(M1)
    action_M1 = group_action(M1, action_vec)
    composition_M1_M2 = group_composition(M1, M2)
    right_plus_M1_tau1 = plus_right(M1, tau1)
    left_plus_M1_tau1 = plus_left(M1, tau1)
    right_minus_M1_M2 = minus_right(M1, M2)
    left_minus_M1_M2 = minus_left(M1, M2)
    adjoint_M1 = adjoint(M1)
    jacobian_inverse_M1 = jacobian_inverse(M1)
    jacobian_composition_1_M1_M2 = jacobian_composition_1(M1, M2)
    jacobian_composition_2_M1_M2 = jacobian_composition_2(M1, M2)
    jacobian_right_tau1 = jacobian_right(tau1)
    jacobian_right_inverse_tau1 = jacobian_right_inverse(tau1)
    jacobian_left_tau1 = jacobian_left(tau1)
    jacobian_left_inverse_tau1 = jacobian_left_inverse(tau1)
    jacobian_plus_right_1_M1_tau1 = jacobian_plus_right_1(M1, tau1)
    jacobian_plus_right_2_M1_tau1 = jacobian_plus_right_2(M1, tau1)
    jacobian_minus_right_1_M1_M2 = jacobian_minus_right_1(M1, M2)
    jacobian_minus_right_2_M1_M2 = jacobian_minus_right_2(M1, M2)
    jacobian_motion_action_1_M1 = jacobian_motion_action_1(M1, action_vec)
    jacobian_motion_action_2_M1 = jacobian_motion_action_2(M1, action_vec)
    assert np.allclose(M1, exp(tau1_hat), atol = 1e-10)
    assert np.allclose(M1, Exp(tau1), atol = 1e-10)
    assert np.allclose(tau1, compose_cartesian_element(cartesian1_1, cartesian1_2), atol = 1e-10)
    assert np.allclose(tau1_hat, hat(tau1), atol = 1e-10)
    assert np.allclose(tau1, vee(tau1_hat), atol = 1e-10)
    assert np.allclose(tau1, Log(M1), atol = 1e-10)
    assert np.allclose(tau1_hat, log(M1), atol = 1e-10)
    assert np.allclose(right_plus_M1_tau1, left_plus_M1_tau1, atol = 1e-10)
    assert np.allclose(adjoint(Exp(tau1)), jacobian_left(tau1) @ jacobian_right_inverse(tau1), atol = 1e-10)
    print("\nAll tests passed!")

def run_test_example():
    np.random.seed(0)
    tau1 = np.random.rand(3,)
    tau2 = np.random.rand(3,)
    action_vec = np.random.rand(3, 1)
    printing(tau1, tau2, action_vec)
    testing(tau1, tau2, action_vec)

run_test_example()
