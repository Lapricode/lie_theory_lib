"""Compare the SE3 implementation against Pinocchio (pip install pin)."""
import time
import numpy as np
import pinocchio as pin
import SE3_rigid_motion as mySE3

np.random.seed(42)

xi1 = 0.1 * np.random.randn(6)
xi2 = 0.1 * np.random.randn(6)
delta = 0.01 * np.random.randn(6)

g1 = pin.exp6(xi1)
g2 = pin.exp6(xi2)

# --- Pinocchio ---------------------------------------------------------------------------------
pin_start_time = time.perf_counter()
pin_comp = g1 * g2
pin_inv = g1.inverse()
pin_adjoint = g1.action
pin_right_plus = g1 * pin.exp6(delta)
pin_right_minus = pin.log6(g2.inverse() * g1)
pin_left_plus = pin.exp6(delta) * g1
pin_left_minus = pin.log6(g1 * g2.inverse())
pin_right_jacobian = pin.Jexp6(delta)
pin_end_time = time.perf_counter()

# Pinocchio objects -> plain numpy arrays (so that np.allclose compares numbers, not objects)
as_mat = lambda g: np.array(g.homogeneous)
as_vec = lambda m: np.array(m.vector) if hasattr(m, "vector") else np.array(m)

G1, G2 = as_mat(g1), as_mat(g2)

# --- This library --------------------------------------------------------------------------------
my_start_time = time.perf_counter()
my_comp = mySE3.group_composition(G1, G2)
my_inv = mySE3.group_inverse(G1)
my_adjoint = mySE3.adjoint(G1)
my_right_plus = mySE3.plus_right(G1, delta)
my_right_minus = mySE3.minus_right(G1, G2)
my_left_plus = mySE3.plus_left(G1, delta)
my_left_minus = mySE3.minus_left(G1, G2)
my_right_jacobian = mySE3.jacobian_right(delta)
my_end_time = time.perf_counter()

rows = [
    ("composition", my_comp, as_mat(pin_comp)),
    ("inverse", my_inv, as_mat(pin_inv)),
    ("adjoint", my_adjoint, np.array(pin_adjoint)),
    ("right plus", my_right_plus, as_mat(pin_right_plus)),
    ("right minus", my_right_minus, as_vec(pin_right_minus)),
    ("left plus", my_left_plus, as_mat(pin_left_plus)),
    ("left minus", my_left_minus, as_vec(pin_left_minus)),
    ("right jacobian", my_right_jacobian, np.array(pin_right_jacobian)),
]
print("Quantity".ljust(18) + "Mine == Pinocchio")
print("-" * 36)
for name, mine, ref in rows:
    print(name.ljust(18) + str(np.allclose(mine, ref)))

my_time, pin_time = my_end_time - my_start_time, pin_end_time - pin_start_time
print(f"\nMy implementation (sec): {my_time:.3e}")
print(f"Pinocchio (sec):         {pin_time:.3e}")
print(f"Pinocchio is {my_time / pin_time:.1f}x faster than this (pure-numpy) implementation" if my_time > pin_time
      else f"This implementation is {pin_time / my_time:.1f}x faster than Pinocchio")
