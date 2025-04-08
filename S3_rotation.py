import numpy as np


'''
tau = theta * u (angle-axis representation) is the cartesian space element, where theta \in R is the angle in radians and u \in R^3 is the unit vector along the axis of rotation (pure qaternion)
q = [cos(theta/2); sin(theta/2) * u] is the group element, where u \in R^3 is the unit vector along the axis of rotation (pure quaternion)
tau_hat = theta * u_hat is the algebra element, where u_hat = [0, -u3, u2; u3, 0, -u1; -u2, u1, 0]
'''
tol = 1e-5  # tolerance for numerical issues

def group_element(tau):
    theta, u = decompose_cartesian_element(tau)
    q = np.array([np.cos(theta / 2), np.sin(theta / 2) * u[0], np.sin(theta / 2) * u[1], np.sin(theta / 2) * u[2]])
    return q

def group_composition(q1, q2):
    w1, v1, v2, v3 = q1[0], q1[1], q1[2], q1[3]
    w2, u1, u2, u3 = q2[0], q2[1], q2[2], q2[3]
    q = np.array([w1 * w2 - v1 * u1 - v2 * u2 - v3 * u3,
                    w1 * u1 + w2 * v1 + v2 * u3 - v3 * u2,
                    w1 * u2 + w2 * v2 + v3 * u1 - v1 * u3,
                    w1 * u3 + w2 * v3 + v1 * u2 - v2 * u1])
    return q

def group_inverse(q):
    w, v1, v2, v3 = q[0], q[1], q[2], q[3]
    return np.array([w, -v1, -v2, -v3])

def group_action(q, x):
    q_inv = group_inverse(q)
    return group_composition(group_composition(q, np.block([0., x])), q_inv)

def algebra_element(tau):
    tau_hat = tau / 2.
    return tau_hat

def compose_cartesian_element(theta, u):
    return theta * u

def decompose_cartesian_element(tau):
    theta = np.linalg.norm(tau)
    u = tau / theta if abs(theta) >= tol else np.array([0., 0., 1.])
    return theta, u

def hat(tau):
    tau_hat = tau / 2.
    return tau_hat

def vee(tau_hat):
    tau = 2. * tau_hat
    return tau

def exp(tau_hat):
    tau = 2. * tau_hat
    theta = np.linalg.norm(tau)
    u = tau / theta if abs(theta) >= tol else np.array([0., 0., 1.])
    q = np.array([np.cos(theta / 2), np.sin(theta / 2) * u[0], np.sin(theta / 2) * u[1], np.sin(theta / 2) * u[2]])
    return q

def log(q):
    w, v1, v2, v3 = q[0], q[1], q[2], q[3]
    v = np.array([v1, v2, v3])
    v_norm = np.linalg.norm(v)
    if abs(v_norm) < tol:
        return np.zeros(3,)
    else:
        tau_hat = np.arctan2(v_norm, w) * v / v_norm
        return tau_hat

def Exp(tau):
    theta = np.linalg.norm(tau)
    u = tau / theta if abs(theta) >= tol else np.array([0., 0., 1.])
    q = np.array([np.cos(theta / 2), np.sin(theta / 2) * u[0], np.sin(theta / 2) * u[1], np.sin(theta / 2) * u[2]])
    return q

def Log(q):
    w, v1, v2, v3 = q[0], q[1], q[2], q[3]
    v = np.array([v1, v2, v3])
    v_norm = np.linalg.norm(v)
    if abs(v_norm) < tol:
        return np.zeros(3,)
    else:
        tau = 2. * np.arctan2(v_norm, w) * v / v_norm
        return tau

def plus_right(q, tau):
    return group_composition(q, Exp(tau))

def plus_left(q, tau):
    return group_composition(Exp(tau), q)

def minus_right(q1, q2):
    return Log(group_composition(group_inverse(q2), q1))

def minus_left(q1, q2):
    return Log(group_composition(q1, group_inverse(q2)))

def adjoint(q):
    return Rq_mat(q)

def jacobian_inverse(q):
    return -Rq_mat(q)

def jacobian_composition_1(q1, q2):
    return Rq_mat(group_inverse(q2))

def jacobian_composition_2(q1, q2):
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

def jacobian_plus_right_1(q, tau):
    Exp_tau = Exp(tau)
    return Rq_mat(group_inverse(Exp_tau))

def jacobian_plus_right_2(q, tau):
    return jacobian_right(tau)

def jacobian_minus_right_1(q1, q2):
    return jacobian_right_inverse(minus_right(q1, q2))

def jacobian_minus_right_2(q1, q2):
    return -jacobian_left_inverse(minus_right(q1, q2))

def jacobian_rotation_action_1(q, x):
    return -Rq_mat(q) @ vec_hat(x)

def jacobian_rotation_action_2(q, x):
    return Rq_mat(q)

def Rq_mat(q):
    w, v1, v2, v3 = q[0], q[1], q[2], q[3]
    Rq = np.array([[w**2 + v1**2 - v2**2 - v3**2, 2*(v1*v2 - w*v3), 2*(v1*v3 + w*v2)],
                  [2*(v1*v2 + w*v3), w**2 - v1**2 + v2**2 - v3**2, 2*(v2*v3 - w*v1)],
                  [2*(v1*v3 - w*v2), 2*(v2*v3 + w*v1), w**2 - v1**2 - v2**2 + v3**2]])
    return Rq

def vec_hat(v):
    return np.block([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])


def printing(tau1, tau2, action_vec):
    q1 = group_element(tau1)
    q2 = group_element(tau2)
    theta1_hat = algebra_element(tau1)
    cartesian1 = decompose_cartesian_element(tau1)
    print(f"tau1:\n {tau1}")
    print(f"tau2:\n {tau2}")
    print(f"action_vec:\n {action_vec}")
    print(f"q1:\n {q1}")
    print(f"q2:\n {q2}")
    print(f"theta1_hat:\n {theta1_hat}")
    print(f"cartesian1:\n {cartesian1}")
    print(f"inverse_q1:\n {group_inverse(q1)}")
    print(f"action_q1:\n {group_action(q1, action_vec)}")
    print(f"composition_q1_q2:\n {group_composition(q1, q2)}")
    print(f"right_plus_q1_tau1:\n {plus_right(q1, tau1)}")
    print(f"left_plus_q1_tau1:\n {plus_left(q1, tau1)}")
    print(f"right_minus_q1_q2:\n {minus_right(q1, q2)}")
    print(f"left_minus_q1_q2:\n {minus_left(q1, q2)}")
    print(f"adjoint_q1:\n {adjoint(q1)}")
    print(f"jacobian_inverse_q1:\n {jacobian_inverse(q1)}")
    print(f"jacobian_composition_1_q1_q2:\n {jacobian_composition_1(q1, q2)}")
    print(f"jacobian_composition_2_q1_q2:\n {jacobian_composition_2(q1, q2)}")
    print(f"jacobian_right_tau1:\n {jacobian_right(tau1)}")
    print(f"jacobian_right_inverse_tau1:\n {jacobian_right_inverse(tau1)}")
    print(f"jacobian_left_tau1:\n {jacobian_left(tau1)}")
    print(f"jacobian_left_inverse_tau1:\n {jacobian_left_inverse(tau1)}")
    print(f"jacobian_plus_right_1_q1_tau1:\n {jacobian_plus_right_1(q1, tau1)}")
    print(f"jacobian_plus_right_2_q1_tau1:\n {jacobian_plus_right_2(q1, tau1)}")
    print(f"jacobian_minus_right_1_q1_q2:\n {jacobian_minus_right_1(q1, q2)}")
    print(f"jacobian_minus_right_2_q1_q2:\n {jacobian_minus_right_2(q1, q2)}")
    print(f"jacobian_rotation_action_1_q1:\n {jacobian_rotation_action_1(q1, action_vec)}")
    print(f"jacobian_rotation_action_2_q1:\n {jacobian_rotation_action_2(q1, action_vec)}")

def testing(tau1, tau2, action_vec):
    q1 = group_element(tau1)
    q2 = group_element(tau2)
    theta1_hat = algebra_element(tau1)
    cartesian1_1, cartesian1_2 = decompose_cartesian_element(tau1)
    inverse_q1 = group_inverse(q1)
    action_q1 = group_action(q1, action_vec)
    composition_q1_q2 = group_composition(q1, q2)
    right_plus_q1_tau1 = plus_right(q1, tau1)
    left_plus_q1_tau1 = plus_left(q1, tau1)
    right_minus_q1_q2 = minus_right(q1, q2)
    left_minus_q1_q2 = minus_left(q1, q2)
    adjoint_q1 = adjoint(q1)
    jacobian_inverse_q1 = jacobian_inverse(q1)
    jacobian_composition_1_q1_q2 = jacobian_composition_1(q1, q2)
    jacobian_composition_2_q1_q2 = jacobian_composition_2(q1, q2)
    jacobian_right_tau1 = jacobian_right(tau1)
    jacobian_right_inverse_tau1 = jacobian_right_inverse(tau1)
    jacobian_left_tau1 = jacobian_left(tau1)
    jacobian_left_inverse_tau1 = jacobian_left_inverse(tau1)
    jacobian_plus_right_1_q1_tau1 = jacobian_plus_right_1(q1, tau1)
    jacobian_plus_right_2_q1_tau1 = jacobian_plus_right_2(q1, tau1)
    jacobian_minus_right_1_q1_q2 = jacobian_minus_right_1(q1, q2)
    jacobian_minus_right_2_q1_q2 = jacobian_minus_right_2(q1, q2)
    jacobian_rotation_action_1_q1 = jacobian_rotation_action_1(q1, action_vec)
    jacobian_rotation_action_2_q1 = jacobian_rotation_action_2(q1, action_vec)
    assert np.allclose(q1, exp(theta1_hat), atol = 1e-10)
    assert np.allclose(q1, Exp(tau1), atol = 1e-10)
    assert np.allclose(tau1, compose_cartesian_element(cartesian1_1, cartesian1_2), atol = 1e-10)
    assert np.allclose(theta1_hat, hat(tau1), atol = 1e-10)
    assert np.allclose(tau1, vee(theta1_hat), atol = 1e-5)
    assert np.allclose(tau1, Log(q1), atol = 1e-10)
    assert np.allclose(theta1_hat, log(q1), atol = 1e-10)
    assert np.allclose(right_plus_q1_tau1, left_plus_q1_tau1, atol = 1e-10)
    assert np.allclose(adjoint(Exp(tau1)), jacobian_left(tau1) @ jacobian_right_inverse(tau1), atol = 1e-10)
    print("\nAll tests passed!")

def run_test_example():
    np.random.seed(0)
    tau1 = np.random.randn(3,)
    tau2 = np.random.randn(3,)
    action_vec = np.random.randn(3,)
    printing(tau1, tau2, action_vec)
    testing(tau1, tau2, action_vec)

run_test_example()
