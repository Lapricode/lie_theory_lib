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


def testing():
    pass
