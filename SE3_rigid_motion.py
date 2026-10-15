import numpy as np
import tolerances
import SO3_rotation as SO3
from lie_utils import vec_hat, vec_vee, coeff_A, coeff_B, coeff_C, coeff_D, coeff_c2, coeff_c3


r'''
tau = [rho; theta * u] is the cartesian space element, where rho \in R^3 is the translation vector and theta * u \in R^3 is the angle-axis representation
M = [R, t; 0, 1] is the group element, where R \in R^{3x3} is the rotation matrix and t \in R^3 is the translation vector
tau_hat = [theta * u_hat, rho; 0, 0] is the algebra element, where u_hat = [0, -u3, u2; u3, 0, -u1; -u2, u1, 0]
'''
tol = tolerances.small_case_tol  # kept for backward compatibility (no longer needed for correctness)

def group_element(tau):
    return Exp(tau)

def group_identity():
    return np.eye(4)

def group_composition(M1, M2):
    return M1 @ M2

def group_inverse(M):
    R = M[:3, :3]
    t = M[:3, 3]
    return np.block([[R.T, -R.T @ t.reshape(-1, 1)], [np.zeros((1, 3)), 1.]])

def group_action(M, x):
    return M @ x

def algebra_element(tau):
    return hat(tau)

def compose_cartesian_element(rho, theta, u):
    return np.concatenate([np.asarray(rho, dtype=float).reshape(3), theta * np.asarray(u, dtype=float).reshape(3)])

def decompose_cartesian_element(tau):
    tau = np.asarray(tau, dtype=float).reshape(6)
    rho = tau[:3]
    v = tau[3:]
    theta = np.linalg.norm(v)
    if theta < np.finfo(float).tiny:
        return rho, 0., np.array([0., 0., 1.])
    return rho, theta, v / theta

def hat(tau):
    tau = np.asarray(tau, dtype=float).reshape(6)
    return np.block([[vec_hat(tau[3:]), tau[:3].reshape(-1, 1)], [np.zeros((1, 4))]])

def vee(tau_hat):
    return np.concatenate([tau_hat[:3, 3], vec_vee(tau_hat[:3, :3])])

def exp(tau_hat):
    return Exp(vee(tau_hat))

def log(M):
    return hat(Log(M))

def Exp(tau):
    tau = np.asarray(tau, dtype=float).reshape(6)
    rho, v = tau[:3], tau[3:]
    return np.block([[SO3.Exp(v), (_V(v) @ rho).reshape(-1, 1)], [np.zeros((1, 3)), 1.]])

def Log(M):
    R = M[:3, :3]
    t = M[:3, 3]
    v = SO3.Log(R)
    return np.concatenate([_Vinv(v) @ t, v])

def plus_right(M, tau):
    return M @ Exp(tau)

def plus_left(M, tau):
    return Exp(tau) @ M

def minus_right(M1, M2):
    return Log(group_inverse(M2) @ M1)

def minus_left(M1, M2):
    return Log(M1 @ group_inverse(M2))

def adjoint(M):
    R = M[:3, :3]
    t = M[:3, 3]
    return np.block([[R, vec_hat(t) @ R], [np.zeros((3, 3)), R]])

def jacobian_inverse(M):
    return -adjoint(M)

def jacobian_composition_1(M1, M2):
    return adjoint(group_inverse(M2))

def jacobian_composition_2(M1, M2):
    return np.eye(6)

def jacobian_right(tau):
    rho, v = np.asarray(tau, dtype=float)[:3], np.asarray(tau, dtype=float)[3:]
    Jl = _Jl(-v)
    Q = _Q(-rho, -v)
    return np.block([[Jl, Q], [np.zeros((3, 3)), Jl]])

def jacobian_right_inverse(tau):
    rho, v = np.asarray(tau, dtype=float)[:3], np.asarray(tau, dtype=float)[3:]
    Jl_inv = _Jl_inv(-v)
    Q = _Q(-rho, -v)
    return np.block([[Jl_inv, -Jl_inv @ Q @ Jl_inv], [np.zeros((3, 3)), Jl_inv]])

def jacobian_left(tau):
    rho, v = np.asarray(tau, dtype=float)[:3], np.asarray(tau, dtype=float)[3:]
    Jl = _Jl(v)
    Q = _Q(rho, v)
    return np.block([[Jl, Q], [np.zeros((3, 3)), Jl]])

def jacobian_left_inverse(tau):
    rho, v = np.asarray(tau, dtype=float)[:3], np.asarray(tau, dtype=float)[3:]
    Jl_inv = _Jl_inv(v)
    Q = _Q(rho, v)
    return np.block([[Jl_inv, -Jl_inv @ Q @ Jl_inv], [np.zeros((3, 3)), Jl_inv]])

def jacobian_plus_right_1(M, tau):
    return adjoint(group_inverse(Exp(tau)))

def jacobian_plus_right_2(M, tau):
    return jacobian_right(tau)

def jacobian_minus_right_1(M1, M2):
    # d(M1 (-) M2)/dM1 = Jr^-1(M1 (-) M2)
    return jacobian_right_inverse(minus_right(M1, M2))

def jacobian_minus_right_2(M1, M2):
    # d(M1 (-) M2)/dM2 = -Jl^-1(M1 (-) M2)
    return -jacobian_left_inverse(minus_right(M1, M2))

def jacobian_motion_action_1(M, p):
    R = M[:3, :3]
    return np.block([R, -R @ vec_hat(np.asarray(p, dtype=float)[:3])])

def jacobian_motion_action_2(M, p):
    return M[:3, :3]

# ---- helpers (vector form: v = theta * u, stable for every angle) ------------------------------

def _Jl(v):
    K = vec_hat(v); theta = np.linalg.norm(v)
    return np.eye(3) + coeff_B(theta) * K + coeff_C(theta) * K @ K

def _Jl_inv(v):
    K = vec_hat(v); theta = np.linalg.norm(v)
    return np.eye(3) - 0.5 * K + coeff_D(theta) * K @ K

def _V(v):
    return _Jl(v)

def _Vinv(v):
    return _Jl_inv(v)

def _Q(rho, v):
    P = vec_hat(rho); K = vec_hat(v); theta = np.linalg.norm(v)
    c1, c2, c3 = coeff_C(theta), coeff_c2(theta), coeff_c3(theta)
    KP, PK, KPK = K @ P, P @ K, K @ P @ K
    return (0.5 * P
            + c1 * (KP + PK + KPK)
            + c2 * (K @ KP + PK @ K - 3. * KPK)
            + c3 * (KPK @ K + K @ KPK))

# ---- (theta, u) API kept for backward compatibility ---------------------------------------------

def V(theta, u):
    return _V(theta * np.asarray(u, dtype=float))

def Vinv(theta, u):
    return _Vinv(theta * np.asarray(u, dtype=float))

def Jl_mat(theta, u):
    return _Jl(theta * np.asarray(u, dtype=float))

def Jl_inv_mat(theta, u):
    return _Jl_inv(theta * np.asarray(u, dtype=float))

def Q_mat(rho, theta, u):
    return _Q(np.asarray(rho, dtype=float), theta * np.asarray(u, dtype=float))
