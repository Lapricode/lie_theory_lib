import numpy as np
import tolerances


r'''
t is the cartesian space element, where t \in R^n is the translation vector
T = [I, t; 0, 1] is the group element, where t \in R^n is the translation vector
t_hat = [0, t; 0, 0] is the algebra element, where t \in R^n is the translation vector
'''
tol = tolerances.small_case_tol  # kept for backward compatibility

def _vec(t):
    return np.asarray(t, dtype=float).reshape(-1)

def group_element(t):
    t = _vec(t)
    n = t.size
    return np.block([[np.eye(n), t.reshape(-1, 1)], [np.zeros((1, n)), 1.]])

def group_identity(n = 3):
    return np.eye(n + 1)

def group_composition(T1, T2):
    return T1 @ T2

def group_inverse(T):
    t = T[:-1, -1]
    n = t.size
    return np.block([[np.eye(n), -t.reshape(-1, 1)], [np.zeros((1, n)), 1.]])

def group_action(T, x):
    return T @ x

def algebra_element(t):
    return hat(t)

def compose_cartesian_element(t):
    return t

def decompose_cartesian_element(t):
    return t

def hat(t):
    t = _vec(t)
    n = t.size
    return np.block([[np.zeros((n, n)), t.reshape(-1, 1)], [np.zeros((1, n)), 0.]])

def vee(t_hat):
    return t_hat[:-1, -1]

def exp(t_hat):
    return group_element(vee(t_hat))

def log(T):
    return hat(Log(T))

def Exp(t):
    return group_element(t)

def Log(T):
    return T[:-1, -1]

def plus_right(T, t):
    return T @ Exp(t)

def plus_left(T, t):
    return Exp(t) @ T

def minus_right(T1, T2):
    # Log(T2^-1 T1) = t1 - t2
    return Log(group_inverse(T2) @ T1)

def minus_left(T1, T2):
    # Log(T1 T2^-1) = t1 - t2
    return Log(T1 @ group_inverse(T2))

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

def jacobian_translation_action_1(T, p):
    return np.eye(T.shape[0] - 1)

def jacobian_translation_action_2(T, p):
    return np.eye(T.shape[0] - 1)
