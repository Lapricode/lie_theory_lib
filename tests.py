"""Strict test-suite for the Lie theory library.

Everything is checked against *independent* ground truth (scipy's expm / matrix inverse and
central finite differences of the group operations), not against the library itself.
Run with:  python tests.py        (or: pytest tests.py)
Requires: numpy, scipy
"""
import sys
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

import S1_rotation as S1
import SO2_rotation as SO2
import S3_rotation as S3
import SO3_rotation as SO3
import SE2_rigid_motion as SE2
import SE3_rigid_motion as SE3
import T_translation as T

ATOL = 1e-7          # comparison tolerance
H = 1e-5             # finite-difference step
REGIMES = ("normal", "small", "tiny", "near_pi")


def vh(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0.]])

def vv(W):
    return np.array([W[2, 1], W[0, 2], W[1, 0]])

def blk(A, b):
    n = A.shape[0]; M = np.zeros((n + 1, n + 1)); M[:n, :n] = A; M[:n, n] = b; return M

def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

def R2q(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat(); return np.array([w, x, y, z])


class Spec:
    """Describes one group through independent (matrix) ground truth."""
    def __init__(self, mod, dim, pt, act_name, hatm, vee_i, tomat, hat_lib, rand_tau, rand_x, act):
        self.mod, self.dim, self.pt, self.act_name = mod, dim, pt, act_name
        self.hatm, self.vee_i, self.tomat, self.hat_lib = hatm, vee_i, tomat, hat_lib
        self.rand_tau, self.rand_x, self.act = rand_tau, rand_x, act


def make_specs(regime, rng):
    def angle_vec(lo, hi):
        v = rng.standard_normal(3); v /= np.linalg.norm(v); return v * rng.uniform(lo, hi)
    rng_ang = {"normal": (0.1, 3.0), "small": (1e-9, 1e-3), "tiny": (1e-12, 1e-7), "near_pi": (np.pi - 1e-4, np.pi - 1e-9)}[regime]
    tau_ang = lambda: angle_vec(*rng_ang)
    X_ang = lambda: angle_vec(0.1, 3.0)
    specs = {}
    specs["SO3"] = Spec(SO3, 3, 3, "rotation", vh, vv, lambda X: X, vh, tau_ang,
                        lambda: expm(vh(X_ang())), lambda X, p: X @ p)
    specs["S3"] = Spec(S3, 3, 3, "rotation", vh, vv, q2R, lambda t: t / 2, tau_ang,
                       lambda: R2q(expm(vh(X_ang()))), lambda X, p: q2R(X) @ p)
    se3h = lambda t: blk(vh(t[3:]), t[:3])
    specs["SE3"] = Spec(SE3, 6, 3, "motion", se3h, lambda M: np.concatenate([M[:3, 3], vv(M[:3, :3])]),
                        lambda X: X, se3h,
                        lambda: np.concatenate([rng.standard_normal(3), tau_ang()]),
                        lambda: expm(se3h(np.concatenate([rng.standard_normal(3) * 2, X_ang()]))),
                        lambda X, p: X[:3, :3] @ p + X[:3, 3])
    se2h = lambda t: blk(np.array([[0, -t[2]], [t[2], 0.]]), t[:2])
    ang2 = {"normal": (0.1, 3.0), "small": (1e-9, 1e-3), "tiny": (1e-12, 1e-7), "near_pi": (np.pi - 1e-4, np.pi - 1e-9)}[regime]
    specs["SE2"] = Spec(SE2, 3, 2, "motion", se2h, lambda M: np.array([M[0, 2], M[1, 2], M[1, 0]]),
                        lambda X: X, se2h,
                        lambda: np.array([*rng.standard_normal(2), rng.uniform(*ang2) * rng.choice([-1, 1])]),
                        lambda: expm(se2h(np.array([*rng.standard_normal(2) * 2, rng.uniform(-3, 3)]))),
                        lambda X, p: X[:2, :2] @ p + X[:2, 2])
    Th = lambda t: blk(np.zeros((3, 3)), t)
    specs["T"] = Spec(T, 3, 3, "translation", Th, lambda M: M[:3, 3], lambda X: X, Th,
                      lambda: rng.standard_normal(3), lambda: expm(Th(rng.standard_normal(3))),
                      lambda X, p: X[:3, 3] + p)
    return specs


def check_group(name, regime, n_samples=25, seed=0):
    """Returns a list of failure descriptions (empty list == all good)."""
    rng = np.random.default_rng(seed)
    g = make_specs(regime, rng)[name]; m, d = g.mod, g.dim
    fails = []

    def chk(label, a, b, atol=ATOL):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        if a.size != b.size or not np.all(np.isfinite(a)) or not np.allclose(a.reshape(-1), b.reshape(-1), atol=atol):
            fails.append(f"{name}.{label} [{regime}]")

    def fd(f):
        return np.array([(np.asarray(f(np.eye(d)[i] * H)) - np.asarray(f(-np.eye(d)[i] * H))) / (2 * H) for i in range(d)]).T

    pr, ms = m.plus_right, m.minus_right
    asm = lambda fn, *a: np.asarray(getattr(m, fn)(*a)).reshape(d, d)
    for _ in range(n_samples):
        tau, X, Y, p = g.rand_tau(), g.rand_x(), g.rand_x(), rng.standard_normal(g.pt)
        E = m.Exp(tau); Xm = g.tomat(X)
        chk("Exp == expm(hat)", g.tomat(E), expm(g.hatm(tau)))
        chk("group_element == expm(hat)", g.tomat(m.group_element(tau)), expm(g.hatm(tau)))
        chk("hat / algebra_element", m.hat(tau), g.hat_lib(tau)); chk("algebra_element", m.algebra_element(tau), g.hat_lib(tau))
        chk("vee(hat(tau)) == tau", m.vee(m.hat(tau)), tau)
        chk("exp(hat(tau)) == expm", g.tomat(m.exp(m.hat(tau))), expm(g.hatm(tau)))
        chk("Log(Exp(tau)) == tau", m.Log(E), tau)
        chk("vee(log(Exp(tau))) == tau", m.vee(m.log(E)), tau)
        chk("composition", g.tomat(m.group_composition(X, Y)), Xm @ g.tomat(Y))
        chk("inverse", g.tomat(m.group_inverse(X)), np.linalg.inv(Xm))
        chk("plus_right", g.tomat(pr(X, tau)), Xm @ expm(g.hatm(tau)))
        chk("plus_left", g.tomat(m.plus_left(X, tau)), expm(g.hatm(tau)) @ Xm)
        small = tau * 0.5 if regime == "normal" else tau
        chk("minus_right(X+t, X) == t", ms(pr(X, small), X), small)
        chk("minus_left(t+X, X) == t", m.minus_left(m.plus_left(X, small), X), small)
        chk("adjoint", asm("adjoint", X) @ tau, g.vee_i(Xm @ g.hatm(tau) @ np.linalg.inv(Xm)))
        Xi, XY = m.group_inverse(X), m.group_composition(X, Y)
        chk("jacobian_inverse", asm("jacobian_inverse", X), fd(lambda e: ms(m.group_inverse(pr(X, e)), Xi)))
        chk("jacobian_composition_1", asm("jacobian_composition_1", X, Y), fd(lambda e: ms(m.group_composition(pr(X, e), Y), XY)))
        chk("jacobian_composition_2", asm("jacobian_composition_2", X, Y), fd(lambda e: ms(m.group_composition(X, pr(Y, e)), XY)))
        Jr, Jl = asm("jacobian_right", tau), asm("jacobian_left", tau)
        chk("jacobian_right", Jr, fd(lambda e: ms(m.Exp(tau + e), E)))
        chk("jacobian_left", Jl, fd(lambda e: m.minus_left(m.Exp(tau + e), E)))
        chk("jacobian_right_inverse", asm("jacobian_right_inverse", tau), np.linalg.inv(Jr))
        chk("jacobian_left_inverse", asm("jacobian_left_inverse", tau), np.linalg.inv(Jl))
        chk("Ad(Exp(tau)) == Jl Jr^-1", asm("adjoint", E), Jl @ np.linalg.inv(Jr))
        Xt = pr(X, tau)
        chk("jacobian_plus_right_1", asm("jacobian_plus_right_1", X, tau), fd(lambda e: ms(pr(pr(X, e), tau), Xt)))
        chk("jacobian_plus_right_2", asm("jacobian_plus_right_2", X, tau), fd(lambda e: ms(pr(X, tau + e), Xt)))
        Z = np.asarray(ms(X, Y))
        chk("jacobian_minus_right_1", asm("jacobian_minus_right_1", X, Y), fd(lambda e: np.asarray(ms(pr(X, e), Y)) - Z))
        chk("jacobian_minus_right_2", asm("jacobian_minus_right_2", X, Y), fd(lambda e: np.asarray(ms(X, pr(Y, e))) - Z))
        an = f"jacobian_{g.act_name}_action_"
        chk(an + "1", np.asarray(getattr(m, an + "1")(X, p)).reshape(g.pt, d), fd(lambda e: g.act(pr(X, e), p)))
        E_pt = np.eye(g.pt)
        chk(an + "2", np.asarray(getattr(m, an + "2")(X, p)).reshape(g.pt, g.pt),
            np.array([(g.act(X, p + E_pt[i] * H) - g.act(X, p - E_pt[i] * H)) / (2 * H) for i in range(g.pt)]).T)
    return sorted(set(fails))


def test_S1_SO2_scalar_groups():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b = rng.uniform(-1.5, 1.5, 2); v = rng.standard_normal() + 1j * rng.standard_normal()
        z1, z2 = S1.Exp(a), S1.Exp(b); R1, R2 = SO2.Exp(a), SO2.Exp(b)
        assert np.allclose(S1.group_composition(z1, S1.group_inverse(z1)), S1.group_identity())
        assert np.isclose(S1.Log(z1), a) and np.isclose(SO2.Log(R1), a)
        assert np.allclose(SO2.group_composition(R1, R2), expm(np.array([[0, -(a + b)], [a + b, 0.]])))
        assert np.isclose(S1.minus_right(z1, z2), a - b) and np.isclose(SO2.minus_right(R1, R2), a - b)
        assert np.isclose(S1.jacobian_minus_right_1(z1, z2), 1) and np.isclose(S1.jacobian_minus_right_2(z1, z2), -1)
        h = 1e-6  # action Jacobian wrt the angle
        assert np.allclose(S1.jacobian_rotation_action_1(z1, v), (S1.group_action(S1.Exp(a + h), v) - S1.group_action(S1.Exp(a - h), v)) / (2 * h), atol=1e-6)
        w = rng.standard_normal(2)
        assert np.allclose(SO2.jacobian_rotation_action_1(R1, w).reshape(-1), (SO2.Exp(a + h) @ w - SO2.Exp(a - h) @ w) / (2 * h), atol=1e-6)


def test_SO3_log_robustness():
    assert np.allclose(SO3.Log(np.eye(3) * (1 + 2e-16)), 0)            # used to return NaN
    for ax in ([1, -1, 0], [0, -1, 0], [1, -1, -1], [0, 0, 1]):         # rotations by exactly pi
        ax = np.array(ax, float); ax /= np.linalg.norm(ax); R = expm(vh(ax * np.pi))
        assert np.allclose(expm(vh(SO3.Log(R))), R, atol=1e-9)
        assert np.allclose(S3.Rq_mat(SO3.qR_quat(R)), R, atol=1e-9)
    q = np.array([np.cos(.4), *(np.sin(.4) * np.array([1, 2, 3.]) / np.sqrt(14))])
    assert np.allclose(S3.Log(-q), S3.Log(q))                          # q and -q are the same rotation
    assert np.linalg.norm(S3.Log(-q)) <= np.pi + 1e-12


def test_T_plus_minus_semantics():
    T1, T2 = T.Exp(np.array([3., 3, 3])), T.Exp(np.array([1., 1, 1]))
    assert T.plus_right(T1, np.ones(3)).shape == (4, 4)
    assert np.allclose(T.minus_right(T1, T2), [2, 2, 2]) and np.allclose(T.minus_left(T1, T2), [2, 2, 2])


def _make_test(name, regime):
    def t():
        f = check_group(name, regime)
        assert not f, "\n".join(f)
    t.__name__ = f"test_{name}_{regime}"
    return t

for _n in ("SO3", "S3", "SE3", "SE2", "T"):
    for _r in (REGIMES if _n != "T" else ("normal",)):
        globals()[f"test_{_n}_{_r}"] = _make_test(_n, _r)


if __name__ == "__main__":
    total_fail = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            try:
                f(); print(f"PASS  {k}")
            except AssertionError as e:
                total_fail += 1; print(f"FAIL  {k}\n      " + str(e).replace("\n", "\n      "))
    print("\nALL TESTS PASSED" if total_fail == 0 else f"\n{total_fail} TEST(S) FAILED")
    sys.exit(1 if total_fail else 0)
