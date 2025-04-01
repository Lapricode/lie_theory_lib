import numpy as np


'''
theta is the cartesian space element
z = np.cos(theta) + i * np.sin(theta) is the lie group element
tau_hat = i * theta is the lie algebra element
'''

def group_element(theta):
    z = np.cos(theta) + 1j * np.sin(theta)
    return z

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


def testing(z, theta):
    pass
