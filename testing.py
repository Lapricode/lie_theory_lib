import numpy as np
import manifpy as mp
from manifpy import SE3, SE3Tangent

X = SE3.Random()
w = SE3Tangent.Random()

J_o_x = np.zeros((SE3.DoF, SE3.DoF))
J_o_w = np.zeros((SE3.DoF, SE3.DoF))

X_plus_w = X.plus(w, J_o_x, J_o_w)

print(X_plus_w)
