import numpy as np


'''
t is the cartesian space element, where t \in R^n is the translation vector
T = [I, t; 0, 1] is the group element, where t \in R^n is the translation vector
t_hat = [0, t; 0, 0] is the algebra element, where t \in R^n is the translation vector
'''

def group_element(t):
    T = np.block([[np.eye(np.size(t)), t.reshape(-1, 1)], [np.zeros((1, np.size(t))), 1.]])
    return T

def group_composition(T1, T2):
    return T1 @ T2

def group_inverse(T):
    t = T[:-1, -1]
    return np.block([[np.eye(np.size(t)), -t.reshape(-1, 1)], [np.zeros((1, np.size(t))), 1.]])

def group_action(T, x):
    return T @ x

def algebra_element(t):
    t_hat = np.block([[np.zeros((np.size(t), np.size(t))), t.reshape(-1, 1)], [np.zeros((1, np.size(t))), 0.]])
    return t_hat

def compose_cartesian_element(t):
    return t

def decompose_cartesian_element(t):
    return t

def hat(t):
    t_hat = np.block([[np.zeros((np.size(t), np.size(t))), t.reshape(-1, 1)], [np.zeros((1, np.size(t))), 0.]])
    return t_hat

def vee(t_hat):
    t = t_hat[:-1, -1]
    return t

def exp(t_hat):
    t = t_hat[:-1, -1]
    T = np.block([[np.eye(np.size(t)), t.reshape(-1, 1)], [np.zeros((1, np.size(t))), 1.]])
    return T

def log(T):
    t = T[:-1, -1]
    t_hat = np.block([[np.zeros((np.size(t), np.size(t))), t.reshape(-1, 1)], [np.zeros((1, np.size(t))), 0.]])
    return t_hat

def Exp(t):
    T = np.block([[np.eye(np.size(t)), t.reshape(-1, 1)], [np.zeros((1, np.size(t))), 1.]])
    return T

def Log(T):
    t = T[:-1, -1]
    return t

def plus_right(T, t):
    return T[:-1, -1].reshape(-1, 1) + t.reshape(-1, 1)

def plus_left(T, t):
    return T[:-1, -1].reshape(-1, 1) + t.reshape(-1, 1)

def minus_right(T1, T2):
    return T2[:-1, -1].reshape(-1, 1) - T1[:-1, -1].reshape(-1, 1)

def minus_left(T1, T2):
    return T2[:-1, -1].reshape(-1, 1) - T1[:-1, -1].reshape(-1, 1)

def adjoint(T):
    return np.eye(T.shape[0] - 1)

def jacobian_inverse(T):
    return -np.eye(T.shape[0] - 1)

def jacobian_composition_1(T1, T2):
    return np.eye(T1.shape[0] - 1)

def jacobian_composition_2(T1, T2):
    return np.eye(T1.shape[0] - 1)

def jacobian_right(t):
    return np.eye(np.size(t))

def jacobian_right_inverse(t):
    return np.eye(np.size(t))

def jacobian_left(t):
    return np.eye(np.size(t))

def jacobian_left_inverse(t):
    return np.eye(np.size(t))

def jacobian_plus_right_1(T, t):
    return np.eye(T.shape[0] - 1)

def jacobian_plus_right_2(T, t):
    return np.eye(T.shape[0] - 1)

def jacobian_minus_right_1(T1, T2):
    return np.eye(T1.shape[0] - 1)

def jacobian_minus_right_2(T1, T2):
    return -np.eye(T1.shape[0] - 1)


def printing(t1, t2, action_vec):
    T1 = group_element(t1)
    T2 = group_element(t2)
    t1_hat = algebra_element(t1)
    cartesian1 = decompose_cartesian_element(t1)
    print(f"t1:\n {t1}")
    print(f"t2:\n {t2}")
    print(f"action_vec:\n {action_vec}")
    print(f"T1:\n {T1}")
    print(f"T2:\n {T2}")
    print(f"t1_hat:\n {t1_hat}")
    print(f"cartesian1:\n {cartesian1}")
    print(f"inverse_T1:\n {group_inverse(T1)}")
    print(f"action_T1:\n {group_action(T1, action_vec)}")
    print(f"composition_T1_T2:\n {group_composition(T1, T2)}")
    print(f"right_plus_T1_t1:\n {plus_right(T1, t1)}")
    print(f"left_plus_T1_t1:\n {plus_left(T1, t1)}")
    print(f"right_minus_T1_T2:\n {minus_right(T1, T2)}")
    print(f"left_minus_T1_T2:\n {minus_left(T1, T2)}")
    print(f"adjoint_T1:\n {adjoint(T1)}")
    print(f"jacobian_inverse_T1:\n {jacobian_inverse(T1)}")
    print(f"jacobian_composition_1_T1_T2:\n {jacobian_composition_1(T1, T2)}")
    print(f"jacobian_composition_2_T1_T2:\n {jacobian_composition_2(T1, T2)}")
    print(f"jacobian_right_t1:\n {jacobian_right(t1)}")
    print(f"jacobian_right_inverse_t1:\n {jacobian_right_inverse(t1)}")
    print(f"jacobian_left_t1:\n {jacobian_left(t1)}")
    print(f"jacobian_left_inverse_t1:\n {jacobian_left_inverse(t1)}")
    print(f"jacobian_plus_right_1_T1_t1:\n {jacobian_plus_right_1(T1, t1)}")
    print(f"jacobian_plus_right_2_T1_t1:\n {jacobian_plus_right_2(T1, t1)}")
    print(f"jacobian_minus_right_1_T1_T2:\n {jacobian_minus_right_1(T1, T2)}")
    print(f"jacobian_minus_right_2_T1_T2:\n {jacobian_minus_right_2(T1, T2)}")

def testing(t1, t2, action_vec):
    T1 = group_element(t1)
    T2 = group_element(t2)
    t1_hat = algebra_element(t1)
    cartesian1 = decompose_cartesian_element(t1)
    inverse_T1 = group_inverse(T1)
    action_T1 = group_action(T1, action_vec)
    composition_T1_T2 = group_composition(T1, T2)
    right_plus_T1_t1 = plus_right(T1, t1)
    left_plus_T1_t1 = plus_left(T1, t1)
    right_minus_T1_T2 = minus_right(T1, T2)
    left_minus_T1_T2 = minus_left(T1, T2)
    adjoint_T1 = adjoint(T1)
    jacobian_inverse_T1 = jacobian_inverse(T1)
    jacobian_composition_1_T1_T2 = jacobian_composition_1(T1, T2)
    jacobian_composition_2_T1_T2 = jacobian_composition_2(T1, T2)
    jacobian_right_t1 = jacobian_right(t1)
    jacobian_right_inverse_t1 = jacobian_right_inverse(t1)
    jacobian_left_t1 = jacobian_left(t1)
    jacobian_left_inverse_t1 = jacobian_left_inverse(t1)
    jacobian_plus_right_1_T1_t1 = jacobian_plus_right_1(T1, t1)
    jacobian_plus_right_2_T1_t1 = jacobian_plus_right_2(T1, t1)
    jacobian_minus_right_1_T1_T2 = jacobian_minus_right_1(T1, T2)
    jacobian_minus_right_2_T1_T2 = jacobian_minus_right_2(T1, T2)
    assert np.allclose(T1, exp(t1_hat), atol = 1e-10)
    assert np.allclose(T1, Exp(t1), atol = 1e-10)
    assert np.allclose(t1, compose_cartesian_element(cartesian1), atol = 1e-10)
    assert np.allclose(t1_hat, hat(t1), atol = 1e-10)
    assert np.allclose(t1, vee(t1_hat), atol = 1e-10)
    assert np.allclose(t1, Log(T1), atol = 1e-10)
    assert np.allclose(t1_hat, log(T1), atol = 1e-10)
    assert np.allclose(right_plus_T1_t1, left_plus_T1_t1, atol = 1e-10)
    print("\nAll tests passed!")

def run_test_example():
    np.random.seed(0)
    n = 3
    t1 = np.random.rand(n,)
    t2 = np.random.rand(n,)
    action_vec = np.random.rand(n+1, 1)
    printing(t1, t2, action_vec)
    testing(t1, t2, action_vec)

run_test_example()
