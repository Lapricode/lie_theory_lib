import numpy as np


'''
theta is the cartesian space element
z = np.cos(theta) + i * np.sin(theta) is the group element
tau_hat = i * theta is the algebra element
'''
tol = 1e-5  # tolerance for numerical issues

def group_element(theta):
    z = np.cos(theta) + 1j * np.sin(theta)
    return z

def group_identity():
    return np.array([1.])

def group_composition(z1, z2):
    return z1 * z2

def group_inverse(z):
    return z.conjugate()

def group_action(z, x):
    return z * x

def algebra_element(theta):
    tau_hat = 1j * theta
    return tau_hat

def compose_cartesian_element(theta):
    return theta

def decompose_cartesian_element(theta):
    return theta

def hat(theta):
    tau_hat = 1j * theta
    return tau_hat

def vee(tau_hat):
    theta = tau_hat.imag
    return theta

def exp(tau_hat):
    theta = -1j * tau_hat
    z = np.cos(theta) + 1j * np.sin(theta)
    return z

def log(z):
    theta = np.angle(z)
    tau_hat = 1j * theta
    return tau_hat

def Exp(theta):
    return np.exp(1j * theta)  # np.exp(1j * theta) = np.cos(theta) + 1j * np.sin(theta)

def Log(z):
    return np.angle(z)  # np.angle(z) = np.arctan2(z.imag, z.real)

def plus_right(z, theta):
    return z * Exp(theta)

def plus_left(z, theta):
    return Exp(theta) * z

def minus_right(z1, z2):
    return Log(z2.conjugate() * z1)

def minus_left(z1, z2):
    return Log(z1 * z2.conjugate())

def adjoint(z):
    return np.array([1.])

def jacobian_inverse(z):
    return -np.array([1.])

def jacobian_composition_1(z1, z2):
    return np.array([1.])

def jacobian_composition_2(z1, z2):
    return np.array([1.])

def jacobian_right(theta):
    return np.array([1.])

def jacobian_right_inverse(theta):
    return np.array([1.])

def jacobian_left(theta):
    return np.array([1.])

def jacobian_left_inverse(theta):
    return np.array([1.])

def jacobian_plus_right_1(z, theta):
    return np.array([1.])

def jacobian_plus_right_2(z, theta):
    return np.array([1.])

def jacobian_minus_right_1(z1, z2):
    return np.array([1.])

def jacobian_minus_right_2(z1, z2):
    return -np.array([1.])

def jacobian_rotation_action_1(z, v):
    return z * 1j * v

def jacobian_rotation_action_2(z, v):
    return z
