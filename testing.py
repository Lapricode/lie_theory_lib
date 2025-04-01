import numpy as np
import time
from manifpy import SE3, SE3Tangent
import pinocchio as pin

np.random.seed(42)

# X = SE3.Random()
# w = SE3Tangent.Random()

# J_o_x = np.zeros((SE3.DoF, SE3.DoF))
# J_o_w = np.zeros((SE3.DoF, SE3.DoF))

# X_plus_w = X.plus(w, J_o_x, J_o_w)

# print(X)
# print(w)
# print(X_plus_w)
# print(X + w)
# # print(J_o_x)
# # print(J_o_w)


xi1 = 0.1 * np.random.randn(6)
xi2 = 0.1 * np.random.randn(6)
delta = 0.01 * np.random.randn(6)

g1 = pin.exp6(xi1)
g2 = pin.exp6(xi2)

pin_start_time = time.time()
pin_comp = g1 * g2
pin_inv = g1.inverse()
pin_adjoint = g1.action
pin_right_plus = g1 * pin.exp6(delta)
pin_right_minus = pin.log6(g2.inverse() * g1)
pin_left_plus = pin.exp6(delta) * g1
pin_left_minus = pin.log6(g1 * g2.inverse())
pin_end_time = time.time()

# # Compute the right Jacobian at the tangent vector xi1, and its inverse
# Jr = pin.Jr(xi1)
# Jr_inv = pin.Jrinv(xi1)

# # Compute the left Jacobian at the tangent vector xi1, and its inverse.
# Jl = pin.Jl(xi1)
# Jl_inv = pin.Jlinv(xi1)

# Display the Results

print(f"g1: {np.array(g1)}")
print(f"g2: {np.array(g2)}")
print(f"Composition (g1 * g2): {np.array(pin_comp)}")
print(f"Inverse of g1: {np.array(pin_inv)}")
print(f"Adjoint of g1: {np.array(pin_adjoint)}")
print(f"Right plus: {np.array(pin_right_plus)}")
print(f"Right minus: {np.array(pin_right_minus)}")
print(f"Left plus: {np.array(pin_left_plus)}")
print(f"Left minus: {np.array(pin_left_minus)}")
# print("\nRight Jacobian at xi1:")
# print(Jr)


import SE3_rigid_motion as mySE3

g1 = np.array(g1)
g2 = np.array(g2)
my_start_time = time.time()
my_comp = mySE3.group_composition(g1, g2)
my_inv = mySE3.group_inverse(g1)
my_adjoint = mySE3.adjoint(g1)
my_right_plus = mySE3.plus_right(g1, delta)
my_right_minus = mySE3.minus_right(g1, g2)
my_left_plus = mySE3.plus_left(g1, delta)
my_left_minus = mySE3.minus_left(g1, g2)
my_end_time = time.time()

print()
left_width = 20
equal_width = 5
right_width = 25
print("My implementation".ljust(left_width) + "==".ljust(equal_width) + "Pinocchio".ljust(right_width) + "Result")
print("-" * (left_width + equal_width + right_width + 8))
print("my_comp".ljust(left_width) + "==".ljust(equal_width) + "pin_comp".ljust(right_width) + str(np.allclose(my_comp, pin_comp)))
print("my_inv".ljust(left_width) + "==".ljust(equal_width) + "pin_inv".ljust(right_width) + str(np.allclose(my_inv, pin_inv)))
print("my_adjoint".ljust(left_width) + "==".ljust(equal_width) + "pin_adjoint".ljust(right_width) + str(np.allclose(my_adjoint, pin_adjoint)))
print("my_right_plus".ljust(left_width) + "==".ljust(equal_width) + "pin_right_plus".ljust(right_width) + str(np.allclose(my_right_plus, pin_right_plus)))
print("my_right_minus".ljust(left_width) + "==".ljust(equal_width) + "pin_right_minus".ljust(right_width) + str(np.allclose(my_right_minus, pin_right_minus)))
print("my_left_plus".ljust(left_width) + "==".ljust(equal_width) + "pin_left_plus".ljust(right_width) + str(np.allclose(my_left_plus, pin_left_plus)))
print("my_left_minus".ljust(left_width) + "==".ljust(equal_width) + "pin_left_minus".ljust(right_width) + str(np.allclose(my_left_minus, pin_left_minus)))

print()
left_padding = 50
my_time = my_end_time - my_start_time
pin_time = pin_end_time - pin_start_time
print("Time elapsed for my implementation (sec):".ljust(left_padding), my_time)
print("Time elapsed for pinocchio (sec):".ljust(left_padding), pin_time, "".ljust(10), f"({np.round(my_time / pin_time, 3)} times faster)")
