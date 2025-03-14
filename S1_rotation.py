import numpy as np


def group_element(theta):
    z = np.cos(theta) + 1j * np.sin(theta)
    return z

def algebra_element(theta):
    tau_hat = 1j * theta
    return tau_hat

def hat(theta):
    return 1j * theta

def vee(z):
    return z.imag

def group_action(z, w):
    return z * w

def Exp(theta):
    return np.exp(1j * theta)  # np.exp(1j * theta) = np.cos(theta) + 1j * np.sin(theta)

def Log(z):
    return np.angle(z)  # np.angle(z) = np.arctan2(z.imag, z.real)

def plus_right(z, tau):
    return z * Exp(tau)

def plus_left(z, tau):
    return Exp(tau) * z

def minus_right(z, w):
    return Log(w.conjugate() * z)

def minus_left(z, w):
    return Log(z * w.conjugate())
