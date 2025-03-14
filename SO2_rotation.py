import numpy as np


def group_element(theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R

def algebra_element(theta):
    tau_hat = np.array([[0, -theta], [theta, 0]])
    return tau_hat

def hat(theta):
    return np.array([[0, -theta], [theta, 0]])

def vee(R):
    return R[1, 0]

def group_action(R, w):
    return R @ w

def Exp(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def Log(R):
    return np.arctan2(R[1, 0], R[0, 0])

