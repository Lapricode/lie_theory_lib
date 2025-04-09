import numpy as np
import S1_rotation as S1
import S3_rotation as S3
import SO2_rotation as SO2
import SO3_rotation as SO3
import SE2_rigid_motion as SE2
import SE3_rigid_motion as SE3
import T_translation as T


# np.random.seed(0)

def S1_rotation_tests(printing = False):
    theta1 = np.random.randn(1)
    theta2 = np.random.randn(1)
    action_vec = np.random.randn(1) + 1j * np.random.randn(1)

    # printing
    if printing:
        print("\n- S1 rotation group printings:\n")
        z1 = S1.group_element(theta1)
        z2 = S1.group_element(theta2)
        theta1_hat = S1.algebra_element(theta1)
        cartesian1 = S1.decompose_cartesian_element(theta1)
        print(f"theta1:\n {theta1}")
        print(f"theta2:\n {theta2}")
        print(f"action_vec:\n {action_vec}")
        print(f"z1:\n {z1}")
        print(f"z2:\n {z2}")
        print(f"theta1_hat:\n {theta1_hat}")
        print(f"cartesian1:\n {cartesian1}")
        print(f"inverse_z1:\n {S1.group_inverse(z1)}")
        print(f"action_z1:\n {S1.group_action(z1, action_vec)}")
        print(f"composition_z1_z2:\n {S1.group_composition(z1, z2)}")
        print(f"theta1_right_plus_z1_theta1:\n {S1.plus_right(z1, theta1)}")
        print(f"theta1_left_plus_z1_theta1:\n {S1.plus_left(z1, theta1)}")
        print(f"right_minus_z1_z2:\n {S1.minus_right(z1, z2)}")
        print(f"left_minus_z1_z2:\n {S1.minus_left(z1, z2)}")
        print(f"adjoint_z1:\n {S1.adjoint(z1)}")
        print(f"jacobian_inverse_z1:\n {S1.jacobian_inverse(z1)}")
        print(f"jacobian_composition_1_z1_z2:\n {S1.jacobian_composition_1(z1, z2)}")
        print(f"jacobian_composition_2_z1_z2:\n {S1.jacobian_composition_2(z1, z2)}")
        print(f"jacobian_right_theta1:\n {S1.jacobian_right(theta1)}")
        print(f"jacobian_right_inverse_theta1:\n {S1.jacobian_right_inverse(theta1)}")
        print(f"jacobian_left_theta1:\n {S1.jacobian_left(theta1)}")
        print(f"jacobian_left_inverse_theta1:\n {S1.jacobian_left_inverse(theta1)}")
        print(f"jacobian_plus_right_1_z1_theta1:\n {S1.jacobian_plus_right_1(z1, theta1)}")
        print(f"jacobian_plus_right_2_z1_theta1:\n {S1.jacobian_plus_right_2(z1, theta1)}")
        print(f"jacobian_minus_right_1_z1_z2:\n {S1.jacobian_minus_right_1(z1, z2)}")
        print(f"jacobian_minus_right_2_z1_z2:\n {S1.jacobian_minus_right_2(z1, z2)}")
        print(f"jacobian_rotation_action_1_z1:\n {S1.jacobian_rotation_action_1(z1, action_vec)}")
        print(f"jacobian_rotation_action_2_z1:\n {S1.jacobian_rotation_action_2(z1, action_vec)}")

    # testing
    print("\n- S1 rotation group testings ...\n")
    z1 = S1.group_element(theta1)
    z2 = S1.group_element(theta2)
    theta1_hat = S1.algebra_element(theta1)
    cartesian1 = S1.decompose_cartesian_element(theta1)
    inverse_z1 = S1.group_inverse(z1)
    action_z1 = S1.group_action(z1, action_vec)
    composition_z1_z2 = S1.group_composition(z1, z2)
    right_plus_z1_theta1 = S1.plus_right(z1, theta1)
    left_plus_z1_theta1 = S1.plus_left(z1, theta1)
    right_minus_z1_z2 = S1.minus_right(z1, z2)
    left_minus_z1_z2 = S1.minus_left(z1, z2)
    adjoint_z1 = S1.adjoint(z1)
    jacobian_inverse_z1 = S1.jacobian_inverse(z1)
    jacobian_composition_1_z1_z2 = S1.jacobian_composition_1(z1, z2)
    jacobian_composition_2_z1_z2 = S1.jacobian_composition_2(z1, z2)
    jacobian_right_theta1 = S1.jacobian_right(theta1)
    jacobian_right_inverse_theta1 = S1.jacobian_right_inverse(theta1)
    jacobian_left_theta1 = S1.jacobian_left(theta1)
    jacobian_left_inverse_theta1 = S1.jacobian_left_inverse(theta1)
    jacobian_plus_right_1_z1_theta1 = S1.jacobian_plus_right_1(z1, theta1)
    jacobian_plus_right_2_z1_theta1 = S1.jacobian_plus_right_2(z1, theta1)
    jacobian_minus_right_1_z1_z2 = S1.jacobian_minus_right_1(z1, z2)
    jacobian_minus_right_2_z1_z2 = S1.jacobian_minus_right_2(z1, z2)
    jacobian_rotation_action_1_z1 = S1.jacobian_rotation_action_1(z1, action_vec)
    jacobian_rotation_action_2_z1 = S1.jacobian_rotation_action_2(z1, action_vec)
    assert np.allclose(S1.group_composition(z1, S1.group_inverse(z1)), S1.group_composition(S1.group_inverse(z1), z1), atol = 1e-10)
    assert np.allclose(S1.group_composition(z1, S1.group_inverse(z1)), S1.group_identity(), atol = 1e-10)
    assert np.allclose(S1.group_action(S1.group_composition(z1, z2), action_vec), S1.group_action(z1, S1.group_action(z2, action_vec)), atol = 1e-10)
    assert np.allclose(z1, S1.exp(theta1_hat), atol = 1e-10)
    assert np.allclose(z1, S1.Exp(theta1), atol = 1e-10)
    assert np.allclose(theta1, S1.compose_cartesian_element(cartesian1), atol = 1e-10)
    assert np.allclose(theta1_hat, S1.hat(theta1), atol = 1e-10)
    assert np.allclose(theta1, S1.vee(theta1_hat), atol = 1e-10)
    assert np.allclose(theta1, S1.Log(z1), atol = 1e-10)
    assert np.allclose(theta1_hat, S1.log(z1), atol = 1e-10)
    assert np.allclose(right_plus_z1_theta1, left_plus_z1_theta1, atol = 1e-10)
    assert np.allclose(S1.adjoint(S1.Exp(theta1)), S1.jacobian_left(theta1) @ S1.jacobian_right_inverse(theta1), atol = 1e-10)
    assert np.allclose(S1.adjoint(S1.group_composition(z1, z2)), S1.adjoint(z1) @ S1.adjoint(z2), atol = 1e-10)
    assert np.allclose(right_plus_z1_theta1, S1.plus_left(z1, adjoint_z1 @ theta1), atol = 1e-10)
    print("All tests passed for S1 rotation group!")

def S3_rotation_tests(printing = False):
    tau1 = np.random.randn(3,)
    tau2 = np.random.randn(3,)
    action_vec = np.random.randn(3,)

    # printing
    if printing:
        print("\n- S3 rotation group printings:\n")
        q1 = S3.group_element(tau1)
        q2 = S3.group_element(tau2)
        tau1_hat = S3.algebra_element(tau1)
        cartesian1 = S3.decompose_cartesian_element(tau1)
        print(f"tau1:\n {tau1}")
        print(f"tau2:\n {tau2}")
        print(f"action_vec:\n {action_vec}")
        print(f"q1:\n {q1}")
        print(f"q2:\n {q2}")
        print(f"tau1_hat:\n {tau1_hat}")
        print(f"cartesian1:\n {cartesian1}")
        print(f"inverse_q1:\n {S3.group_inverse(q1)}")
        print(f"action_q1:\n {S3.group_action(q1, action_vec)}")
        print(f"composition_q1_q2:\n {S3.group_composition(q1, q2)}")
        print(f"right_plus_q1_tau1:\n {S3.plus_right(q1, tau1)}")
        print(f"left_plus_q1_tau1:\n {S3.plus_left(q1, tau1)}")
        print(f"right_minus_q1_q2:\n {S3.minus_right(q1, q2)}")
        print(f"left_minus_q1_q2:\n {S3.minus_left(q1, q2)}")
        print(f"adjoint_q1:\n {S3.adjoint(q1)}")
        print(f"jacobian_inverse_q1:\n {S3.jacobian_inverse(q1)}")
        print(f"jacobian_composition_1_q1_q2:\n {S3.jacobian_composition_1(q1, q2)}")
        print(f"jacobian_composition_2_q1_q2:\n {S3.jacobian_composition_2(q1, q2)}")
        print(f"jacobian_right_tau1:\n {S3.jacobian_right(tau1)}")
        print(f"jacobian_right_inverse_tau1:\n {S3.jacobian_right_inverse(tau1)}")
        print(f"jacobian_left_tau1:\n {S3.jacobian_left(tau1)}")
        print(f"jacobian_left_inverse_tau1:\n {S3.jacobian_left_inverse(tau1)}")
        print(f"jacobian_plus_right_1_q1_tau1:\n {S3.jacobian_plus_right_1(q1, tau1)}")
        print(f"jacobian_plus_right_2_q1_tau1:\n {S3.jacobian_plus_right_2(q1, tau1)}")
        print(f"jacobian_minus_right_1_q1_q2:\n {S3.jacobian_minus_right_1(q1, q2)}")
        print(f"jacobian_minus_right_2_q1_q2:\n {S3.jacobian_minus_right_2(q1, q2)}")
        print(f"jacobian_rotation_action_1_q1:\n {S3.jacobian_rotation_action_1(q1, action_vec)}")
        print(f"jacobian_rotation_action_2_q1:\n {S3.jacobian_rotation_action_2(q1, action_vec)}")

    # testing
    print("\n- S3 rotation group testings ...\n")
    q1 = S3.group_element(tau1)
    q2 = S3.group_element(tau2)
    tau1_hat = S3.algebra_element(tau1)
    cartesian1_theta, cartesian1_u = S3.decompose_cartesian_element(tau1)
    inverse_q1 = S3.group_inverse(q1)
    action_q1 = S3.group_action(q1, action_vec)
    composition_q1_q2 = S3.group_composition(q1, q2)
    right_plus_q1_tau1 = S3.plus_right(q1, tau1)
    left_plus_q1_tau1 = S3.plus_left(q1, tau1)
    right_minus_q1_q2 = S3.minus_right(q1, q2)
    left_minus_q1_q2 = S3.minus_left(q1, q2)
    adjoint_q1 = S3.adjoint(q1)
    jacobian_inverse_q1 = S3.jacobian_inverse(q1)
    jacobian_composition_1_q1_q2 = S3.jacobian_composition_1(q1, q2)
    jacobian_composition_2_q1_q2 = S3.jacobian_composition_2(q1, q2)
    jacobian_right_tau1 = S3.jacobian_right(tau1)
    jacobian_right_inverse_tau1 = S3.jacobian_right_inverse(tau1)
    jacobian_left_tau1 = S3.jacobian_left(tau1)
    jacobian_left_inverse_tau1 = S3.jacobian_left_inverse(tau1)
    jacobian_plus_right_1_q1_tau1 = S3.jacobian_plus_right_1(q1, tau1)
    jacobian_plus_right_2_q1_tau1 = S3.jacobian_plus_right_2(q1, tau1)
    jacobian_minus_right_1_q1_q2 = S3.jacobian_minus_right_1(q1, q2)
    jacobian_minus_right_2_q1_q2 = S3.jacobian_minus_right_2(q1, q2)
    jacobian_rotation_action_1_q1 = S3.jacobian_rotation_action_1(q1, action_vec)
    jacobian_rotation_action_2_q1 = S3.jacobian_rotation_action_2(q1, action_vec)
    assert np.allclose(S3.group_composition(q1, S3.group_inverse(q1)), S3.group_composition(S3.group_inverse(q1), q1), atol = 1e-10)
    assert np.allclose(S3.group_composition(q1, S3.group_inverse(q1)), S3.group_identity(), atol = 1e-10)
    assert np.allclose(S3.group_action(S3.group_composition(q1, q2), action_vec), S3.group_action(q1, S3.group_action(q2, action_vec)), atol = 1e-10)
    assert np.allclose(q1, S3.exp(tau1_hat), atol = 1e-10)
    assert np.allclose(q1, S3.Exp(tau1), atol = 1e-10)
    assert np.allclose(tau1, S3.compose_cartesian_element(cartesian1_theta, cartesian1_u), atol = 1e-10)
    assert np.allclose(tau1_hat, S3.hat(tau1), atol = 1e-10)
    assert np.allclose(tau1, S3.vee(tau1_hat), atol=1e-5)
    assert np.allclose(tau1, S3.Log(q1), atol = 1e-10)
    assert np.allclose(tau1_hat, S3.log(q1), atol = 1e-10)
    assert np.allclose(right_plus_q1_tau1, left_plus_q1_tau1, atol = 1e-10)
    assert np.allclose(S3.adjoint(S3.Exp(tau1)), S3.jacobian_left(tau1) @ S3.jacobian_right_inverse(tau1), atol = 1e-10)
    assert np.allclose(S3.adjoint(S3.group_composition(q1, q2)), S3.adjoint(q1) @ S3.adjoint(q2), atol = 1e-10)
    assert np.allclose(right_plus_q1_tau1, S3.plus_left(q1, adjoint_q1 @ tau1), atol = 1e-10)
    print("All tests passed for S3 rotation group!")

def SO2_rotation_tests(printing = False):
    theta1 = np.random.randn(1)
    theta2 = np.random.randn(1)
    action_vec = np.random.randn(2,)

    # printing
    if printing:
        print("\n- SO2 rotation group printings:")
        R1 = SO2.group_element(theta1)
        R2 = SO2.group_element(theta2)
        theta1_hat = SO2.algebra_element(theta1)
        cartesian1 = SO2.decompose_cartesian_element(theta1)
        print(f"theta1:\n {theta1}")
        print(f"theta2:\n {theta2}")
        print(f"action_vec:\n {action_vec}")
        print(f"R1:\n {R1}")
        print(f"R2:\n {R2}")
        print(f"theta1_hat:\n {theta1_hat}")
        print(f"cartesian1:\n {cartesian1}")
        print(f"inverse_R1:\n {SO2.group_inverse(R1)}")
        print(f"action_R1:\n {SO2.group_action(R1, action_vec)}")
        print(f"composition_R1_R2:\n {SO2.group_composition(R1, R2)}")
        print(f"right_plus_R1_theta1:\n {SO2.plus_right(R1, theta1)}")
        print(f"left_plus_R1_theta1:\n {SO2.plus_left(R1, theta1)}")
        print(f"right_minus_R1_R2:\n {SO2.minus_right(R1, R2)}")
        print(f"left_minus_R1_R2:\n {SO2.minus_left(R1, R2)}")
        print(f"adjoint_R1:\n {SO2.adjoint(R1)}")
        print(f"jacobian_inverse_R1:\n {SO2.jacobian_inverse(R1)}")
        print(f"jacobian_composition_1_R1_R2:\n {SO2.jacobian_composition_1(R1, R2)}")
        print(f"jacobian_composition_2_R1_R2:\n {SO2.jacobian_composition_2(R1, R2)}")
        print(f"jacobian_right_theta1:\n {SO2.jacobian_right(theta1)}")
        print(f"jacobian_right_inverse_theta1:\n {SO2.jacobian_right_inverse(theta1)}")
        print(f"jacobian_left_theta1:\n {SO2.jacobian_left(theta1)}")
        print(f"jacobian_left_inverse_theta1:\n {SO2.jacobian_left_inverse(theta1)}")
        print(f"jacobian_plus_right_1_R1_theta1:\n {SO2.jacobian_plus_right_1(R1, theta1)}")
        print(f"jacobian_plus_right_2_R1_theta1:\n {SO2.jacobian_plus_right_2(R1, theta1)}")
        print(f"jacobian_minus_right_1_R1_R2:\n {SO2.jacobian_minus_right_1(R1, R2)}")
        print(f"jacobian_minus_right_2_R1_R2:\n {SO2.jacobian_minus_right_2(R1, R2)}")
        print(f"jacobian_rotation_action_1_R1:\n {SO2.jacobian_rotation_action_1(R1, action_vec)}")
        print(f"jacobian_rotation_action_2_R1:\n {SO2.jacobian_rotation_action_2(R1, action_vec)}")

    # testing
    print("\n- SO2 rotation group testings:")
    R1 = SO2.group_element(theta1)
    R2 = SO2.group_element(theta2)
    theta1_hat = SO2.algebra_element(theta1)
    cartesian1 = SO2.decompose_cartesian_element(theta1)
    inverse_R1 = SO2.group_inverse(R1)
    action_R1 = SO2.group_action(R1, action_vec)
    composition_R1_R2 = SO2.group_composition(R1, R2)
    right_plus_R1_theta1 = SO2.plus_right(R1, theta1)
    left_plus_R1_theta1 = SO2.plus_left(R1, theta1)
    right_minus_R1_R2 = SO2.minus_right(R1, R2)
    left_minus_R1_R2 = SO2.minus_left(R1, R2)
    adjoint_R1 = SO2.adjoint(R1)
    jacobian_inverse_R1 = SO2.jacobian_inverse(R1)
    jacobian_composition_1_R1_R2 = SO2.jacobian_composition_1(R1, R2)
    jacobian_composition_2_R1_R2 = SO2.jacobian_composition_2(R1, R2)
    jacobian_right_theta1 = SO2.jacobian_right(theta1)
    jacobian_right_inverse_theta1 = SO2.jacobian_right_inverse(theta1)
    jacobian_left_theta1 = SO2.jacobian_left(theta1)
    jacobian_left_inverse_theta1 = SO2.jacobian_left_inverse(theta1)
    jacobian_plus_right_1_R1_theta1 = SO2.jacobian_plus_right_1(R1, theta1)
    jacobian_plus_right_2_R1_theta1 = SO2.jacobian_plus_right_2(R1, theta1)
    jacobian_minus_right_1_R1_R2 = SO2.jacobian_minus_right_1(R1, R2)
    jacobian_minus_right_2_R1_R2 = SO2.jacobian_minus_right_2(R1, R2)
    jacobian_rotation_action_1_R1 = SO2.jacobian_rotation_action_1(R1, action_vec)
    jacobian_rotation_action_2_R1 = SO2.jacobian_rotation_action_2(R1, action_vec)
    assert np.allclose(SO2.group_composition(R1, SO2.group_inverse(R1)), SO2.group_composition(SO2.group_inverse(R1), R1), atol = 1e-10)
    assert np.allclose(SO2.group_composition(R1, SO2.group_inverse(R1)), SO2.group_identity(), atol = 1e-10)
    assert np.allclose(SO2.group_action(SO2.group_composition(R1, R2), action_vec), SO2.group_action(R1, SO2.group_action(R2, action_vec)), atol = 1e-10)
    assert np.allclose(R1, SO2.exp(theta1_hat), atol = 1e-10)
    assert np.allclose(R1, SO2.Exp(theta1), atol = 1e-10)
    assert np.allclose(theta1, SO2.compose_cartesian_element(cartesian1), atol = 1e-10)
    assert np.allclose(theta1_hat, SO2.hat(theta1), atol = 1e-10)
    assert np.allclose(theta1, SO2.vee(theta1_hat), atol = 1e-10)
    assert np.allclose(theta1, SO2.Log(R1), atol = 1e-10)
    assert np.allclose(theta1_hat, SO2.log(R1), atol = 1e-10)
    assert np.allclose(right_plus_R1_theta1, left_plus_R1_theta1, atol = 1e-10)
    assert np.allclose(SO2.adjoint(SO2.Exp(theta1)), SO2.jacobian_left(theta1) @ SO2.jacobian_right_inverse(theta1), atol = 1e-10)
    assert np.allclose(SO2.adjoint(SO2.group_composition(R1, R2)), SO2.adjoint(R1) @ SO2.adjoint(R2), atol = 1e-10)
    assert np.allclose(right_plus_R1_theta1, SO2.plus_left(R1, adjoint_R1 @ theta1), atol = 1e-10)
    print("\nAll tests passed for SO2 rotation group!")

def SO3_rotation_tests(printing = False):
    tau1 = np.random.randn(3,)
    tau2 = np.random.randn(3,)
    action_vec = np.random.randn(3,)

    # printing
    if printing:
        print("\n- SO3 rotation group printings:")
        R1 = SO3.group_element(tau1)
        R2 = SO3.group_element(tau2)
        theta1_hat = SO3.algebra_element(tau1)
        cartesian1 = SO3.decompose_cartesian_element(tau1)
        print(f"tau1:\n {tau1}")
        print(f"tau2:\n {tau2}")
        print(f"action_vec:\n {action_vec}")
        print(f"R1:\n {R1}")
        print(f"R2:\n {R2}")
        print(f"theta1_hat:\n {theta1_hat}")
        print(f"cartesian1:\n {cartesian1}")
        print(f"inverse_R1:\n {SO3.group_inverse(R1)}")
        print(f"action_R1:\n {SO3.group_action(R1, action_vec)}")
        print(f"composition_R1_R2:\n {SO3.group_composition(R1, R2)}")
        print(f"right_plus_R1_tau1:\n {SO3.plus_right(R1, tau1)}")
        print(f"left_plus_R1_tau1:\n {SO3.plus_left(R1, tau1)}")
        print(f"right_minus_R1_R2:\n {SO3.minus_right(R1, R2)}")
        print(f"left_minus_R1_R2:\n {SO3.minus_left(R1, R2)}")
        print(f"adjoint_R1:\n {SO3.adjoint(R1)}")
        print(f"jacobian_inverse_R1:\n {SO3.jacobian_inverse(R1)}")
        print(f"jacobian_composition_1_R1_R2:\n {SO3.jacobian_composition_1(R1, R2)}")
        print(f"jacobian_composition_2_R1_R2:\n {SO3.jacobian_composition_2(R1, R2)}")
        print(f"jacobian_right_tau1:\n {SO3.jacobian_right(tau1)}")
        print(f"jacobian_right_inverse_tau1:\n {SO3.jacobian_right_inverse(tau1)}")
        print(f"jacobian_left_tau1:\n {SO3.jacobian_left(tau1)}")
        print(f"jacobian_left_inverse_tau1:\n {SO3.jacobian_left_inverse(tau1)}")
        print(f"jacobian_plus_right_1_R1_tau1:\n {SO3.jacobian_plus_right_1(R1, tau1)}")
        print(f"jacobian_plus_right_2_R1_tau1:\n {SO3.jacobian_plus_right_2(R1, tau1)}")
        print(f"jacobian_minus_right_1_R1_R2:\n {SO3.jacobian_minus_right_1(R1, R2)}")
        print(f"jacobian_minus_right_2_R1_R2:\n {SO3.jacobian_minus_right_2(R1, R2)}")
        print(f"jacobian_rotation_action_1_R1:\n {SO3.jacobian_rotation_action_1(R1, action_vec)}")
        print(f"jacobian_rotation_action_2_R1:\n {SO3.jacobian_rotation_action_2(R1, action_vec)}")

    # testing
    print("\n- SO3 rotation group testings:")
    R1 = SO3.group_element(tau1)
    R2 = SO3.group_element(tau2)
    theta1_hat = SO3.algebra_element(tau1)
    cartesian1_1, cartesian1_2 = SO3.decompose_cartesian_element(tau1)
    inverse_R1 = SO3.group_inverse(R1)
    action_R1 = SO3.group_action(R1, action_vec)
    composition_R1_R2 = SO3.group_composition(R1, R2)
    right_plus_R1_tau1 = SO3.plus_right(R1, tau1)
    left_plus_R1_tau1 = SO3.plus_left(R1, tau1)
    right_minus_R1_R2 = SO3.minus_right(R1, R2)
    left_minus_R1_R2 = SO3.minus_left(R1, R2)
    adjoint_R1 = SO3.adjoint(R1)
    jacobian_inverse_R1 = SO3.jacobian_inverse(R1)
    jacobian_composition_1_R1_R2 = SO3.jacobian_composition_1(R1, R2)
    jacobian_composition_2_R1_R2 = SO3.jacobian_composition_2(R1, R2)
    jacobian_right_tau1 = SO3.jacobian_right(tau1)
    jacobian_right_inverse_tau1 = SO3.jacobian_right_inverse(tau1)
    jacobian_left_tau1 = SO3.jacobian_left(tau1)
    jacobian_left_inverse_tau1 = SO3.jacobian_left_inverse(tau1)
    jacobian_plus_right_1_R1_tau1 = SO3.jacobian_plus_right_1(R1, tau1)
    jacobian_plus_right_2_R1_tau1 = SO3.jacobian_plus_right_2(R1, tau1)
    jacobian_minus_right_1_R1_R2 = SO3.jacobian_minus_right_1(R1, R2)
    jacobian_minus_right_2_R1_R2 = SO3.jacobian_minus_right_2(R1, R2)
    jacobian_rotation_action_1_R1 = SO3.jacobian_rotation_action_1(R1, action_vec)
    jacobian_rotation_action_2_R1 = SO3.jacobian_rotation_action_2(R1, action_vec)
    assert np.allclose(SO3.group_composition(R1, SO3.group_inverse(R1)), SO3.group_composition(SO3.group_inverse(R1), R1), atol = 1e-10)
    assert np.allclose(SO3.group_composition(R1, SO3.group_inverse(R1)), SO3.group_identity(), atol = 1e-10)
    assert np.allclose(SO3.group_action(SO3.group_composition(R1, R2), action_vec), SO3.group_action(R1, SO3.group_action(R2, action_vec)), atol = 1e-10)
    assert np.allclose(R1, SO3.exp(theta1_hat), atol = 1e-10)
    assert np.allclose(R1, SO3.Exp(tau1), atol = 1e-10)
    assert np.allclose(tau1, SO3.compose_cartesian_element(cartesian1_1, cartesian1_2), atol = 1e-10)
    assert np.allclose(theta1_hat, SO3.hat(tau1), atol = 1e-10)
    assert np.allclose(tau1, SO3.vee(theta1_hat), atol = 1e-5)
    assert np.allclose(tau1, SO3.Log(R1), atol = 1e-10)
    assert np.allclose(theta1_hat, SO3.log(R1), atol = 1e-10)
    assert np.allclose(right_plus_R1_tau1, left_plus_R1_tau1, atol = 1e-10)
    assert np.allclose(SO3.adjoint(SO3.Exp(tau1)), SO3.jacobian_left(tau1) @ SO3.jacobian_right_inverse(tau1), atol = 1e-10)
    assert np.allclose(SO3.adjoint(SO3.group_composition(R1, R2)), SO3.adjoint(R1) @ SO3.adjoint(R2), atol = 1e-10)
    assert np.allclose(right_plus_R1_tau1, SO3.plus_left(R1, adjoint_R1 @ tau1), atol = 1e-10)
    print("\nAll tests passed for SO3 rotation group!")

def SE2_rigid_motion_tests(printing = False):
    tau1 = np.random.rand(3,)
    tau2 = np.random.rand(3,)
    action_vec = np.random.rand(3,)

    # printing
    if printing:
        print("\n- SE2 rigid motion group printings:")
        M1 = SE2.group_element(tau1)
        M2 = SE2.group_element(tau2)
        tau1_hat = SE2.algebra_element(tau1)
        cartesian1 = SE2.decompose_cartesian_element(tau1)
        print(f"tau1:\n {tau1}")
        print(f"tau2:\n {tau2}")
        print(f"action_vec:\n {action_vec}")
        print(f"M1:\n {M1}")
        print(f"M2:\n {M2}")
        print(f"tau1_hat:\n {tau1_hat}")
        print(f"cartesian1:\n {cartesian1}")
        print(f"inverse_M1:\n {SE2.group_inverse(M1)}")
        print(f"action_M1:\n {SE2.group_action(M1, action_vec)}")
        print(f"composition_M1_M2:\n {SE2.group_composition(M1, M2)}")
        print(f"right_plus_M1_tau1:\n {SE2.plus_right(M1, tau1)}")
        print(f"left_plus_M1_tau1:\n {SE2.plus_left(M1, tau1)}")
        print(f"right_minus_M1_M2:\n {SE2.minus_right(M1, M2)}")
        print(f"left_minus_M1_M2:\n {SE2.minus_left(M1, M2)}")
        print(f"adjoint_M1:\n {SE2.adjoint(M1)}")
        print(f"jacobian_inverse_M1:\n {SE2.jacobian_inverse(M1)}")
        print(f"jacobian_composition_1_M1_M2:\n {SE2.jacobian_composition_1(M1, M2)}")
        print(f"jacobian_composition_2_M1_M2:\n {SE2.jacobian_composition_2(M1, M2)}")
        print(f"jacobian_right_tau1:\n {SE2.jacobian_right(tau1)}")
        print(f"jacobian_right_inverse_tau1:\n {SE2.jacobian_right_inverse(tau1)}")
        print(f"jacobian_left_tau1:\n {SE2.jacobian_left(tau1)}")
        print(f"jacobian_left_inverse_tau1:\n {SE2.jacobian_left_inverse(tau1)}")
        print(f"jacobian_plus_right_1_M1_tau1:\n {SE2.jacobian_plus_right_1(M1, tau1)}")
        print(f"jacobian_plus_right_2_M1_tau1:\n {SE2.jacobian_plus_right_2(M1, tau1)}")
        print(f"jacobian_minus_right_1_M1_M2:\n {SE2.jacobian_minus_right_1(M1, M2)}")
        print(f"jacobian_minus_right_2_M1_M2:\n {SE2.jacobian_minus_right_2(M1, M2)}")
        print(f"jacobian_motion_action_1_M1:\n {SE2.jacobian_motion_action_1(M1, action_vec)}")
        print(f"jacobian_motion_action_2_M1:\n {SE2.jacobian_motion_action_2(M1, action_vec)}")

    # testing
    print("\n- SE2 rigid motion group testings:")
    M1 = SE2.group_element(tau1)
    M2 = SE2.group_element(tau2)
    tau1_hat = SE2.algebra_element(tau1)
    cartesian1_1, cartesian1_2 = SE2.decompose_cartesian_element(tau1)
    inverse_M1 = SE2.group_inverse(M1)
    action_M1 = SE2.group_action(M1, action_vec)
    composition_M1_M2 = SE2.group_composition(M1, M2)
    right_plus_M1_tau1 = SE2.plus_right(M1, tau1)
    left_plus_M1_tau1 = SE2.plus_left(M1, tau1)
    right_minus_M1_M2 = SE2.minus_right(M1, M2)
    left_minus_M1_M2 = SE2.minus_left(M1, M2)
    adjoint_M1 = SE2.adjoint(M1)
    jacobian_inverse_M1 = SE2.jacobian_inverse(M1)
    jacobian_composition_1_M1_M2 = SE2.jacobian_composition_1(M1, M2)
    jacobian_composition_2_M1_M2 = SE2.jacobian_composition_2(M1, M2)
    jacobian_right_tau1 = SE2.jacobian_right(tau1)
    jacobian_right_inverse_tau1 = SE2.jacobian_right_inverse(tau1)
    jacobian_left_tau1 = SE2.jacobian_left(tau1)
    jacobian_left_inverse_tau1 = SE2.jacobian_left_inverse(tau1)
    jacobian_plus_right_1_M1_tau1 = SE2.jacobian_plus_right_1(M1, tau1)
    jacobian_plus_right_2_M1_tau1 = SE2.jacobian_plus_right_2(M1, tau1)
    jacobian_minus_right_1_M1_M2 = SE2.jacobian_minus_right_1(M1, M2)
    jacobian_minus_right_2_M1_M2 = SE2.jacobian_minus_right_2(M1, M2)
    jacobian_motion_action_1_M1 = SE2.jacobian_motion_action_1(M1, action_vec)
    jacobian_motion_action_2_M1 = SE2.jacobian_motion_action_2(M1, action_vec)
    assert np.allclose(SE2.group_composition(M1, SE2.group_inverse(M1)), SE2.group_composition(SE2.group_inverse(M1), M1), atol = 1e-10)
    assert np.allclose(SE2.group_composition(M1, SE2.group_inverse(M1)), SE2.group_identity(), atol = 1e-10)
    assert np.allclose(SE2.group_action(SE2.group_composition(M1, M2), action_vec), SE2.group_action(M1, SE2.group_action(M2, action_vec)), atol = 1e-10)
    assert np.allclose(M1, SE2.exp(tau1_hat), atol = 1e-10)
    assert np.allclose(M1, SE2.Exp(tau1), atol = 1e-10)
    assert np.allclose(tau1, SE2.compose_cartesian_element(cartesian1_1, cartesian1_2), atol = 1e-10)
    assert np.allclose(tau1_hat, SE2.hat(tau1), atol = 1e-10)
    assert np.allclose(tau1, SE2.vee(tau1_hat), atol = 1e-10)
    assert np.allclose(tau1, SE2.Log(M1), atol = 1e-10)
    assert np.allclose(tau1_hat, SE2.log(M1), atol = 1e-10)
    assert np.allclose(right_plus_M1_tau1, left_plus_M1_tau1, atol = 1e-10)
    assert np.allclose(SE2.adjoint(SE2.Exp(tau1)), SE2.jacobian_left(tau1) @ SE2.jacobian_right_inverse(tau1), atol = 1e-10)
    assert np.allclose(SE2.adjoint(SE2.group_composition(M1, M2)), SE2.adjoint(M1) @ SE2.adjoint(M2), atol = 1e-10)
    assert np.allclose(right_plus_M1_tau1, SE2.plus_left(M1, adjoint_M1 @ tau1), atol = 1e-10)
    print("\nAll tests passed for SE2 rigid motion group!")

def SE3_rigid_motion_tests(printing = False):
    tau1 = np.random.rand(6,)
    tau2 = np.random.rand(6,)
    action_vec = np.random.rand(4,)

    # printing
    if printing:
        print("\n- SE3 rigid motion group printings:")
        M1 = SE3.group_element(tau1)
        M2 = SE3.group_element(tau2)
        tau1_hat = SE3.algebra_element(tau1)
        cartesian1 = SE3.decompose_cartesian_element(tau1)
        print(f"tau1:\n {tau1}")
        print(f"tau2:\n {tau2}")
        print(f"action_vec:\n {action_vec}")
        print(f"M1:\n {M1}")
        print(f"M2:\n {M2}")
        print(f"tau1_hat:\n {tau1_hat}")
        print(f"cartesian1:\n {cartesian1}")
        print(f"inverse_M1:\n {SE3.group_inverse(M1)}")
        print(f"action_M1:\n {SE3.group_action(M1, action_vec)}")
        print(f"composition_M1_M2:\n {SE3.group_composition(M1, M2)}")
        print(f"right_plus_M1_tau1:\n {SE3.plus_right(M1, tau1)}")
        print(f"left_plus_M1_tau1:\n {SE3.plus_left(M1, tau1)}")
        print(f"right_minus_M1_M2:\n {SE3.minus_right(M1, M2)}")
        print(f"left_minus_M1_M2:\n {SE3.minus_left(M1, M2)}")
        print(f"adjoint_M1:\n {SE3.adjoint(M1)}")
        print(f"jacobian_inverse_M1:\n {SE3.jacobian_inverse(M1)}")
        print(f"jacobian_composition_1_M1_M2:\n {SE3.jacobian_composition_1(M1, M2)}")
        print(f"jacobian_composition_2_M1_M2:\n {SE3.jacobian_composition_2(M1, M2)}")
        print(f"jacobian_right_tau1:\n {SE3.jacobian_right(tau1)}")
        print(f"jacobian_right_inverse_tau1:\n {SE3.jacobian_right_inverse(tau1)}")
        print(f"jacobian_left_tau1:\n {SE3.jacobian_left(tau1)}")
        print(f"jacobian_left_inverse_tau1:\n {SE3.jacobian_left_inverse(tau1)}")
        print(f"jacobian_plus_right_1_M1_tau1:\n {SE3.jacobian_plus_right_1(M1, tau1)}")
        print(f"jacobian_plus_right_2_M1_tau1:\n {SE3.jacobian_plus_right_2(M1, tau1)}")
        print(f"jacobian_minus_right_1_M1_M2:\n {SE3.jacobian_minus_right_1(M1, M2)}")
        print(f"jacobian_minus_right_2_M1_M2:\n {SE3.jacobian_minus_right_2(M1, M2)}")
        print(f"jacobian_motion_action_1_M1:\n {SE3.jacobian_motion_action_1(M1, action_vec)}")
        print(f"jacobian_motion_action_2_M1:\n {SE3.jacobian_motion_action_2(M1, action_vec)}")

    # testing
    print("\n- SE3 rigid motion group testings:")
    M1 = SE3.group_element(tau1)
    M2 = SE3.group_element(tau2)
    tau1_hat = SE3.algebra_element(tau1)
    cartesian1_1, cartesian1_2, cartesian1_3 = SE3.decompose_cartesian_element(tau1)
    inverse_M1 = SE3.group_inverse(M1)
    action_M1 = SE3.group_action(M1, action_vec)
    composition_M1_M2 = SE3.group_composition(M1, M2)
    right_plus_M1_tau1 = SE3.plus_right(M1, tau1)
    left_plus_M1_tau1 = SE3.plus_left(M1, tau1)
    right_minus_M1_M2 = SE3.minus_right(M1, M2)
    left_minus_M1_M2 = SE3.minus_left(M1, M2)
    adjoint_M1 = SE3.adjoint(M1)
    jacobian_inverse_M1 = SE3.jacobian_inverse(M1)
    jacobian_composition_1_M1_M2 = SE3.jacobian_composition_1(M1, M2)
    jacobian_composition_2_M1_M2 = SE3.jacobian_composition_2(M1, M2)
    jacobian_right_tau1 = SE3.jacobian_right(tau1)
    jacobian_right_inverse_tau1 = SE3.jacobian_right_inverse(tau1)
    jacobian_left_tau1 = SE3.jacobian_left(tau1)
    jacobian_left_inverse_tau1 = SE3.jacobian_left_inverse(tau1)
    jacobian_plus_right_1_M1_tau1 = SE3.jacobian_plus_right_1(M1, tau1)
    jacobian_plus_right_2_M1_tau1 = SE3.jacobian_plus_right_2(M1, tau1)
    jacobian_minus_right_1_M1_M2 = SE3.jacobian_minus_right_1(M1, M2)
    jacobian_minus_right_2_M1_M2 = SE3.jacobian_minus_right_2(M1, M2)
    jacobian_motion_action_1_M1 = SE3.jacobian_motion_action_1(M1, action_vec)
    jacobian_motion_action_2_M1 = SE3.jacobian_motion_action_2(M1, action_vec)
    assert np.allclose(SE3.group_composition(M1, SE3.group_inverse(M1)), SE3.group_composition(SE3.group_inverse(M1), M1), atol = 1e-10)
    assert np.allclose(SE3.group_composition(M1, SE3.group_inverse(M1)), SE3.group_identity(), atol = 1e-10)
    assert np.allclose(SE3.group_action(SE3.group_composition(M1, M2), action_vec), SE3.group_action(M1, SE3.group_action(M2, action_vec)), atol = 1e-10)
    assert np.allclose(M1, SE3.exp(tau1_hat), atol = 1e-10)
    assert np.allclose(M1, SE3.Exp(tau1), atol = 1e-10)
    assert np.allclose(tau1, SE3.compose_cartesian_element(cartesian1_1, cartesian1_2, cartesian1_3), atol = 1e-10)
    assert np.allclose(tau1_hat, SE3.hat(tau1), atol = 1e-10)
    assert np.allclose(tau1, SE3.vee(tau1_hat), atol = 1e-10)
    assert np.allclose(tau1, SE3.Log(M1), atol = 1e-10)
    assert np.allclose(tau1_hat, SE3.log(M1), atol = 1e-10)
    assert np.allclose(right_plus_M1_tau1, left_plus_M1_tau1, atol = 1e-10)
    assert np.allclose(SE3.adjoint(SE3.Exp(tau1)), SE3.jacobian_left(tau1) @ SE3.jacobian_right_inverse(tau1), atol = 1e-10)
    assert np.allclose(SE3.adjoint(SE3.group_composition(M1, M2)), SE3.adjoint(M1) @ SE3.adjoint(M2), atol = 1e-10)
    assert np.allclose(right_plus_M1_tau1, SE3.plus_left(M1, adjoint_M1 @ tau1), atol = 1e-10)
    print("\nAll tests passed for SE3 rigid motion group!")

def T_translation_tests(printing = False):
    n = 3
    t1 = np.random.rand(n,)
    t2 = np.random.rand(n,)
    action_vec = np.random.rand(n + 1,)

    # printing
    if printing:
        print("\n- T translation group printings:")
        T1 = T.group_element(t1)
        T2 = T.group_element(t2)
        t1_hat = T.algebra_element(t1)
        cartesian1 = T.decompose_cartesian_element(t1)
        print(f"t1:\n {t1}")
        print(f"t2:\n {t2}")
        print(f"action_vec:\n {action_vec}")
        print(f"T1:\n {T1}")
        print(f"T2:\n {T2}")
        print(f"t1_hat:\n {t1_hat}")
        print(f"cartesian1:\n {cartesian1}")
        print(f"inverse_T1:\n {T.group_inverse(T1)}")
        print(f"action_T1:\n {T.group_action(T1, action_vec)}")
        print(f"composition_T1_T2:\n {T.group_composition(T1, T2)}")
        print(f"right_plus_T1_t1:\n {T.plus_right(T1, t1)}")
        print(f"left_plus_T1_t1:\n {T.plus_left(T1, t1)}")
        print(f"right_minus_T1_T2:\n {T.minus_right(T1, T2)}")
        print(f"left_minus_T1_T2:\n {T.minus_left(T1, T2)}")
        print(f"adjoint_T1:\n {T.adjoint(T1)}")
        print(f"jacobian_inverse_T1:\n {T.jacobian_inverse(T1)}")
        print(f"jacobian_composition_1_T1_T2:\n {T.jacobian_composition_1(T1, T2)}")
        print(f"jacobian_composition_2_T1_T2:\n {T.jacobian_composition_2(T1, T2)}")
        print(f"jacobian_right_t1:\n {T.jacobian_right(t1)}")
        print(f"jacobian_right_inverse_t1:\n {T.jacobian_right_inverse(t1)}")
        print(f"jacobian_left_t1:\n {T.jacobian_left(t1)}")
        print(f"jacobian_left_inverse_t1:\n {T.jacobian_left_inverse(t1)}")
        print(f"jacobian_plus_right_1_T1_t1:\n {T.jacobian_plus_right_1(T1, t1)}")
        print(f"jacobian_plus_right_2_T1_t1:\n {T.jacobian_plus_right_2(T1, t1)}")
        print(f"jacobian_minus_right_1_T1_T2:\n {T.jacobian_minus_right_1(T1, T2)}")
        print(f"jacobian_minus_right_2_T1_T2:\n {T.jacobian_minus_right_2(T1, T2)}")
        print(f"jacobian_translation_action_1_T1:\n {T.jacobian_translation_action_1(T1, action_vec)}")
        print(f"jacobian_translation_action_2_T1:\n {T.jacobian_translation_action_2(T1, action_vec)}")

    # testing
    print("\n- T translation group testings:")
    T1 = T.group_element(t1)
    T2 = T.group_element(t2)
    t1_hat = T.algebra_element(t1)
    cartesian1 = T.decompose_cartesian_element(t1)
    inverse_T1 = T.group_inverse(T1)
    action_T1 = T.group_action(T1, action_vec)
    composition_T1_T2 = T.group_composition(T1, T2)
    right_plus_T1_t1 = T.plus_right(T1, t1)
    left_plus_T1_t1 = T.plus_left(T1, t1)
    right_minus_T1_T2 = T.minus_right(T1, T2)
    left_minus_T1_T2 = T.minus_left(T1, T2)
    adjoint_T1 = T.adjoint(T1)
    jacobian_inverse_T1 = T.jacobian_inverse(T1)
    jacobian_composition_1_T1_T2 = T.jacobian_composition_1(T1, T2)
    jacobian_composition_2_T1_T2 = T.jacobian_composition_2(T1, T2)
    jacobian_right_t1 = T.jacobian_right(t1)
    jacobian_right_inverse_t1 = T.jacobian_right_inverse(t1)
    jacobian_left_t1 = T.jacobian_left(t1)
    jacobian_left_inverse_t1 = T.jacobian_left_inverse(t1)
    jacobian_plus_right_1_T1_t1 = T.jacobian_plus_right_1(T1, t1)
    jacobian_plus_right_2_T1_t1 = T.jacobian_plus_right_2(T1, t1)
    jacobian_minus_right_1_T1_T2 = T.jacobian_minus_right_1(T1, T2)
    jacobian_minus_right_2_T1_T2 = T.jacobian_minus_right_2(T1, T2)
    jacobian_translation_action_1_T1 = T.jacobian_translation_action_1(T1, action_vec)
    jacobian_translation_action_2_T1 = T.jacobian_translation_action_2(T1, action_vec)
    assert np.allclose(T.group_composition(T1, T.group_inverse(T1)), T.group_composition(T.group_inverse(T1), T1), atol = 1e-10)
    assert np.allclose(T.group_composition(T1, T.group_inverse(T1)), T.group_identity(), atol = 1e-10)
    assert np.allclose(T.group_action(T.group_composition(T1, T2), action_vec), T.group_action(T1, T.group_action(T2, action_vec)), atol = 1e-10)
    assert np.allclose(T1, T.exp(t1_hat), atol = 1e-10)
    assert np.allclose(T1, T.Exp(t1), atol = 1e-10)
    assert np.allclose(t1, T.compose_cartesian_element(cartesian1), atol = 1e-10)
    assert np.allclose(t1_hat, T.hat(t1), atol = 1e-10)
    assert np.allclose(t1, T.vee(t1_hat), atol = 1e-10)
    assert np.allclose(t1, T.Log(T1), atol = 1e-10)
    assert np.allclose(t1_hat, T.log(T1), atol = 1e-10)
    assert np.allclose(right_plus_T1_t1, left_plus_T1_t1, atol = 1e-10)
    assert np.allclose(T.adjoint(T.Exp(t1)), T.jacobian_left(t1) @ T.jacobian_right_inverse(t1), atol = 1e-10)
    assert np.allclose(T.adjoint(T.group_composition(T1, T2)), T.adjoint(T1) @ T.adjoint(T2), atol = 1e-10)
    assert np.allclose(right_plus_T1_t1, T.plus_left(T1, adjoint_T1 @ t1), atol = 1e-10)
    print("\nAll tests passed for T translation group!")


if __name__ == "__main__":
    S1_rotation_tests(printing = False)
    S3_rotation_tests(printing = False)
    SO2_rotation_tests(printing = False)
    SO3_rotation_tests(printing = False)
    SE2_rigid_motion_tests(printing = False)
    SE3_rigid_motion_tests(printing = False)
    T_translation_tests(printing = False)
    print("\n\nAll tests passed for all groups tested!")
