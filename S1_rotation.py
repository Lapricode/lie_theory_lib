import numpy as np


'''
theta is the cartesian space element
z = np.cos(theta) + i * np.sin(theta) is the group element
tau_hat = i * theta is the algebra element
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


def printing(theta1, theta2, action_vec):
    z1 = group_element(theta1)
    z2 = group_element(theta2)
    theta1_hat = algebra_element(theta1)
    cartesian1 = decompose_cartesian_element(theta1)
    print(f"theta1:\n {theta1}")
    print(f"theta2:\n {theta2}")
    print(f"action_vec:\n {action_vec}")
    print(f"z1:\n {z1}")
    print(f"z2:\n {z2}")
    print(f"theta1_hat:\n {theta1_hat}")
    print(f"cartesian1:\n {cartesian1}")
    print(f"inverse_z1:\n {group_inverse(z1)}")
    print(f"action_z1:\n {group_action(z1, action_vec)}")
    print(f"composition_z1_z2:\n {group_composition(z1, z2)}")
    print(f"theta1_right_plus_z1_theta1:\n {plus_right(z1, theta1)}")
    print(f"theta1_left_plus_z1_theta1:\n {plus_left(z1, theta1)}")
    print(f"right_minus_z1_z2:\n {minus_right(z1, z2)}")
    print(f"left_minus_z1_z2:\n {minus_left(z1, z2)}")

def testing(theta1, theta2, action_vec):
    z1 = group_element(theta1)
    z2 = group_element(theta2)
    theta1_hat = algebra_element(theta1)
    cartesian1 = decompose_cartesian_element(theta1)
    inverse_z1 = group_inverse(z1)
    action_z1 = group_action(z1, action_vec)
    composition_z1_z2 = group_composition(z1, z2)
    right_plus_z1_theta1 = plus_right(z1, theta1)
    left_plus_z1_theta1 = plus_left(z1, theta1)
    right_minus_z1_z2 = minus_right(z1, z2)
    left_minus_z1_z2 = minus_left(z1, z2)
    assert np.allclose(z1, exp(theta1_hat), atol = 1e-10)
    assert np.allclose(z1, Exp(theta1), atol = 1e-10)
    assert np.allclose(theta1, compose_cartesian_element(cartesian1), atol = 1e-10)
    assert np.allclose(theta1_hat, hat(theta1), atol = 1e-10)
    assert np.allclose(theta1, vee(theta1_hat), atol = 1e-10)
    assert np.allclose(theta1, Log(z1), atol = 1e-10)
    assert np.allclose(theta1_hat, log(z1), atol = 1e-10)
    assert np.allclose(right_plus_z1_theta1, left_plus_z1_theta1, atol = 1e-10)
    print("\nAll tests passed!")

def run_test_example():
    np.random.seed(0)
    theta1 = np.random.randn(1)
    theta2 = np.random.randn(1)
    action_vec = np.random.randn(1)
    printing(theta1, theta2, action_vec)
    # testing(theta1, theta2, action_vec)
