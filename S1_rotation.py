import numpy as np


def group_element(theta):
    z = np.cos(theta) + 1j * np.sin(theta)
    return z

def algebra_element(theta):
    z = 1j * theta
    return z

