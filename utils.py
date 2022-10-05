from constants import *
import numpy as np
import casadi as ca


# given 1x3 vector, returns 3x3 skew symmetric cross product matrix
def skew_np(s):
    return np.array([[0, -s[2], s[1]], [s[2], 0, -s[0]], [-s[1], s[0], 0]])


# derives a symbolic version of the skew function
def derive_skew_ca():
    s = ca.SX.sym("s", 3)

    skew_sym = ca.SX(3, 3)
    # skew_sym = ca.SX.zeros(3, 3)
    skew_sym[0, 1] = -s[2]
    skew_sym[0, 2] = s[1]
    skew_sym[1, 0] = s[2]
    skew_sym[1, 2] = -s[0]
    skew_sym[2, 0] = -s[1]
    skew_sym[2, 1] = s[0]

    return ca.Function("skew_ca", [s], [skew_sym])


# 2D rotation matrix
def rot_mat_2d_np(th):
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


# given axis and angle, returns 3x3 rotation matrix
def rot_mat_np(s, th):
    # normalize s if isn't already normalized
    norm_s = np.linalg.norm(s)
    assert norm_s != 0.0
    s_normalized = s / norm_s

    # Rodrigues' rotation formula
    skew_s = skew_np(s_normalized)
    return np.eye(3) + np.sin(th) * skew_s + (1.0 - np.cos(th)) * skew_s @ skew_s


# derives a symbolic version of the rotMat function
def derive_rot_mat_ca():
    s = ca.SX.sym("s", 3)
    th = ca.SX.sym("th")
    skew_ca = derive_skew_ca()
    skew_sym = skew_ca(s)

    rot_mat_sym = (
        ca.SX.eye(3) + ca.sin(th) * skew_sym + (1 - ca.cos(th)) * skew_sym @ skew_sym
    )
    return ca.Function("rot_mat_ca", [s, th], [rot_mat_sym])


# given position vector and rotation matrix, returns 4x4 homogeneous
# transformation matrix
def homog_np(p, R):
    return np.block([[R, p[:, np.newaxis]], [np.zeros((1, 3)), 1]])


# derives a symbolic version of the homog function
def derive_homog_ca():
    p = ca.SX.sym("p", 3)
    R = ca.SX.sym("R", 3, 3)
    homog_sym = ca.SX(4, 4)
    homog_sym[:3, :3] = R
    homog_sym[:3, 3] = p
    homog_sym[3, 3] = 1.0
    return ca.Function("homog_ca", [p, R], [homog_sym])


# reverses the direction of the coordinate transformation defined by a 4x4
# homogeneous transformation matrix
def reverse_homog_np(T):
    R = T[:3, :3]
    p = T[:3, 3]
    reverse_homog = np.zeros((4, 4))
    reverse_homog[:3, :3] = R.T
    reverse_homog[:3, 3] = -R.T @ p
    reverse_homog[3, 3] = 1.0
    return reverse_homog


# derives a symbolic function that reverses the direction of the coordinate
# transformation defined by a 4x4 homogeneous transformation matrix
def derive_reverse_homog_ca():
    T = ca.SX.sym("T", 4, 4)
    R = T[:3, :3]
    p = T[:3, 3]
    reverse_homog_sym = ca.SX(4, 4)
    reverse_homog_sym[:3, :3] = R.T
    reverse_homog_sym[:3, 3] = -R.T @ p
    reverse_homog_sym[3, 3] = 1.0
    return ca.Function("reverse_homog_ca", [T], [reverse_homog_sym])


# multiplication between a 4x4 homogenous transformation matrix and 3x1
# position vector, returns 3x1 position
def mult_homog_point_np(T, p):
    p_aug = np.concatenate((p, [1.0]))
    return (T @ p_aug)[:3]


# derives a symbolic version of the mult_homog_point function
def derive_mult_homog_point_ca():
    T = ca.SX.sym("T", 4, 4)
    p = ca.SX.sym("p", 3)
    p_aug = ca.SX.ones(4, 1)
    p_aug[:3] = p
    mult_homog_point_sym = (T @ p_aug)[:3]
    return ca.Function("mult_homog_point_ca", [T, p], [mult_homog_point_sym])


# multiplication between a 4x4 homogenous transformation matrix and 3x1
# force vector, returns 3x1 force
def mult_homog_vec_np(T, f):
    f_aug = np.concatenate((f, [0.0]))
    return (T @ f_aug)[:3]


# derives a symbolic version of the mult_homog_vec function
def derive_mult_homog_vec_ca():
    T = ca.SX.sym("T", 4, 4)
    f = ca.SX.sym("f", 3)
    f_aug = ca.SX.zeros(4, 1)
    f_aug[:3] = f
    mult_homog_vec_sym = (T @ f_aug)[:3]
    return ca.Function("mult_homog_vec_ca", [T, f], [mult_homog_vec_sym])


# generic planar 2 link inverse kinematics implementation
# returns the closest point within the workspace if the requested point is
# outside of it
def planar_IK_np(l1, l2, x, y, elbow_up):
    l = np.sqrt(x**2.0 + y**2.0)
    l = max(abs(l1 - l2), min(l, l1 + l2))

    alpha = np.arctan2(y, x)

    cos_beta = (l**2 + l1**2 - l2**2.0) / (2.0 * l * l1)
    cos_beta = max(-1.0, min(cos_beta, 1.0))
    beta = np.arccos(cos_beta)

    cos_th2_abs = (l**2 - l1**2.0 - l2**2.0) / (2.0 * l1 * l2)
    cos_th2_abs = max(-1.0, min(cos_th2_abs, 1.0))
    th2_abs = np.arccos(cos_th2_abs)

    if elbow_up:
        th1 = alpha - beta
        th2 = th2_abs
    else:
        th1 = alpha + beta
        th2 = -th2_abs

    return th1, th2


# generic planar 2 link jacobian inverse transpose calculation implementation
# end_effector_force = jacobian_inv_tranpose * joint_torque
def planar_jac_inv_transpose_np(l1, l2, th1, th2, tau1, tau2):
    J = np.array(
        [
            [-l1 * np.sin(th1) - l2 * np.sin(th1 + th2), -l2 * np.sin(th1 + th2)],
            [l1 * np.cos(th1) + l2 * np.cos(th1 + th2), l2 * np.cos(th1 + th2)],
        ]
    )
    force = np.linalg.solve(J.T, np.array([tau1, tau2]))
    return force


# generic planar 2 link jacobian transpose calculation implementation
# joint_torque = jacobian_tranpose * end_effector_force
# end_effector_force is force that robot exerts on environment
def planar_jac_transpose_np(l1, l2, th1, th2, f1, f2):
    J = np.array(
        [
            [-l1 * np.sin(th1) - l2 * np.sin(th1 + th2), -l2 * np.sin(th1 + th2)],
            [l1 * np.cos(th1) + l2 * np.cos(th1 + th2), l2 * np.cos(th1 + th2)],
        ]
    )
    tau = J.T @ np.array([f1, f2])
    return tau


# Solo specific functions below

# position of corners of robot, in body frame (so it's a constant)
B_T_Bi = {}
for leg in legs:
    B_T_Bi[leg] = homog_np(B_p_Bi[leg], np.eye(3))


# given numpy trajectory matrix, extract state at timestep k
# note the order argument in reshape, which is necessary to make it consistent
# with casadi's reshape
def extract_state_np(X, U, k):
    p = X[:3, k]
    R_flat = X[3:12, k]
    R = np.reshape(R_flat, (3, 3), order="F")
    pdot = X[12:15, k]
    omega = X[15:18, k]
    p_i = {}
    f_i = {}
    for leg in legs:
        p_i[leg] = U[3 * leg.value : leg.value * 3 + 3, k]
        f_i[leg] = U[12 + 3 * leg.value : 12 + leg.value * 3 + 3, k]
    return p, R, pdot, omega, p_i, f_i


# given casadi trajectory matrix, extract state at timestep k
def extract_state_ca(X, U, k):
    p = X[:3, k]
    R_flat = X[3:12, k]
    R = ca.reshape(R_flat, 3, 3)
    pdot = X[12:15, k]
    omega = X[15:18, k]
    p_i = {}
    f_i = {}
    for leg in legs:
        p_i[leg] = U[3 * leg.value : leg.value * 3 + 3, k]
        f_i[leg] = U[12 + 3 * leg.value : 12 + leg.value * 3 + 3, k]
    return p, R, pdot, omega, p_i, f_i


# given a numpy state, flattens it into the same form as a column of a
# trajectory matrix
def flatten_state_np(p, R, pdot, omega, p_i, f_i):
    R_flat = np.reshape(R, 9, order="F")
    p_i_flat = np.zeros(12)
    f_i_flat = np.zeros(12)
    for leg in legs:
        p_i_flat[3 * leg.value : leg.value * 3 + 3] = p_i[leg]
        f_i_flat[3 * leg.value : leg.value * 3 + 3] = f_i[leg]

    X_k = np.hstack((p, R_flat, pdot, omega))
    U_k = np.hstack((p_i_flat, f_i_flat))

    return X_k, U_k


# inverse kinematics for the solo 8 robot
def solo_IK_np(p, R, p_i, elbow_up_front=True, elbow_up_hind=False):
    T_B = homog_np(p, R)
    rotate_90 = rot_mat_2d_np(np.pi / 2.0)
    q_i = {}
    for leg in legs:
        T_Bi = T_B @ B_T_Bi[leg]
        Bi_T = reverse_homog_np(T_Bi)
        Bi_p_i = mult_homog_point_np(Bi_T, p_i[leg])
        # assert abs(Bi_p_i[1]) < eps # foot should be in shoulder plane
        x_z = rotate_90 @ np.array([Bi_p_i[0], Bi_p_i[2]])
        if leg == legs.FL or leg == legs.FR:
            q1, q2 = planar_IK_np(l_thigh, l_calf, x_z[0], x_z[1], elbow_up_front)
        else:
            q1, q2 = planar_IK_np(l_thigh, l_calf, x_z[0], x_z[1], elbow_up_hind)
        q_i[leg] = np.array([q1, q2])

    return q_i


# jacobian transpose calculation for the solo 8 robot
def solo_jac_transpose_np(p, R, p_i, f_i, elbow_up_front=True, elbow_up_hind=False):
    q_i = solo_IK_np(p, R, p_i, elbow_up_front, elbow_up_hind)
    T_B = homog_np(p, R)
    rotate_90 = rot_mat_2d_np(np.pi / 2.0)
    tau_i = {}
    for leg in legs:
        T_Bi = T_B @ B_T_Bi[leg]
        Bi_T = reverse_homog_np(T_Bi)
        # NOTE: ground reaction force needs to be negated to get force from robot to ground,
        # not from ground to robot
        Bi_f_i = mult_homog_vec_np(Bi_T, -f_i[leg])  # note negative sign
        # assert abs(Bi_f_i[1]) < eps # ground reaction force should be in shoulder plane
        f_xz = rotate_90 @ np.array([Bi_f_i[0], Bi_f_i[2]])
        tau_i[leg] = planar_jac_transpose_np(
            l_thigh, l_calf, q_i[leg][0], q_i[leg][1], f_xz[0], f_xz[1]
        )

    return tau_i


# function commented out since it wasn't updated to account for various knee
# configurations and it isn't used anywhere
# # jacobian inverse transpose calculation for the solo 8 robot
# def solo_jac_inv_transpose_np(p, R, p_i, tau_i):
#     q_i = solo_IK_np(p, R, p_i)
#     T_B = homog_np(p, R)
#     rotate_neg_90 = rot_mat_2d_np(-np.pi / 2.0)
#     f_i = {}
#     for leg in legs:
#         T_Bi = T_B @ B_T_Bi[leg]
#         f_xz = planar_jac_inv_transpose_np(
#             l_thigh, l_calf, q_i[leg][0], q_i[leg][1], tau_i[leg][0], tau_i[leg][1]
#         )
#         Bi_f_i = rotate_neg_90 @ f_xz
#         # note negative sign to convert force from robot to force from ground
#         f_i[leg] = -mult_homog_vec_np(T_Bi, np.array([Bi_f_i[0], 0.0, Bi_f_i[1]]))

#     return f_i


# test functions
if __name__ == "__main__":
    x_axis = np.eye(3)[:, 0]
    y_axis = np.eye(3)[:, 1]
    z_axis = np.eye(3)[:, 2]

    # print("\ntest skew")
    # skew_ca = derive_skew_ca()
    # print(skew_np(np.array([1, 2, 3])))
    # print(skew_ca(np.array([1, 2, 3])))
    # s = ca.SX.sym("s", 3)
    # print(skew_ca(s))

    # print("\ntest rotMat")
    # rot_mat_ca = derive_rot_mat_ca()
    # print(rot_mat_np(x_axis, np.pi / 4))
    # print(rot_mat_ca(x_axis, np.pi / 4))
    # print(rot_mat_np(y_axis, np.pi / 4))
    # print(rot_mat_ca(y_axis, np.pi / 4))
    # print(rot_mat_np(z_axis, np.pi / 4))
    # print(rot_mat_ca(z_axis, np.pi / 4))
    # print(
    #     np.linalg.norm(
    #         rot_mat_np(x_axis, np.pi / 4) @ rot_mat_np(x_axis, np.pi / 4).T - np.eye(3)
    #     )
    # )
    # print(
    #     np.linalg.norm(
    #         rot_mat_np(x_axis, np.pi / 4).T @ rot_mat_np(x_axis, np.pi / 4) - np.eye(3)
    #     )
    # )
    # th = ca.SX.sym("th")
    # print(rot_mat_ca(s, th))
    # print(
    #     np.linalg.norm(
    #         rot_mat_ca(x_axis, np.pi / 4) @ rot_mat_ca(x_axis, np.pi / 4).T - np.eye(3)
    #     )
    # )

    # print("\ntest homog")
    # homog_ca = derive_homog_ca()
    # p = np.array([1, 2, 3])
    # R = rot_mat_np(x_axis, np.pi / 4)
    # print(homog_np(p, R))
    # print(homog_ca(p, R))

    # print("\ntest mult_homog_point")
    # mult_homog_point_ca = derive_mult_homog_point_ca()
    # print(mult_homog_point_np(homog_np(x_axis, R), y_axis))
    # print(mult_homog_point_ca(homog_np(x_axis, R), y_axis))

    # print("\ntest mult_homog_vec")
    # mult_homog_vec_ca = derive_mult_homog_vec_ca()
    # print(mult_homog_vec_np(homog_np(x_axis, R), y_axis))
    # print(mult_homog_vec_ca(homog_np(x_axis, R), y_axis))

    # print("\ntest planar_jac_transpose_np")
    # l1 = l_thigh
    # l2 = l_calf
    # # columns th1, th2, f1, f2
    # test_cases = np.array(
    #     [
    #         [0.0, 0.0, 1.0, 0.0],
    #         [0.0, 0.0, -1.0, 0.0],
    #         [0.0, 0.0, 0.0, 1.0],
    #         [0.0, 0.0, 0.0, -1.0],
    #         [0.0, 90.0, 1.0, 0.0],
    #         [0.0, 90.0, -1.0, 0.0],
    #         [0.0, 90.0, 0.0, 1.0],
    #         [0.0, 90.0, 0.0, -1.0],
    #         [0.0, 180.0, 1.0, 0.0],
    #         [0.0, 180.0, -1.0, 0.0],
    #         [0.0, 180.0, 0.0, 1.0],
    #         [0.0, 180.0, 0.0, -1.0],
    #         [90.0, 0.0, 1.0, 0.0],
    #         [90.0, 0.0, -1.0, 0.0],
    #         [90.0, 0.0, 0.0, 1.0],
    #         [90.0, 0.0, 0.0, -1.0],
    #         [-90.0, 0.0, 1.0, 0.0],
    #         [-90.0, 0.0, -1.0, 0.0],
    #         [-90.0, 0.0, 0.0, 1.0],
    #         [-90.0, 0.0, 0.0, -1.0],
    #         [0.0, 0.0, 1.0, 1.0],
    #     ]
    # )

    # test_cases[:, :2] *= np.pi / 180
    # for idx in range(test_cases.shape[0]):
    #     th1 = test_cases[idx, 0]
    #     th2 = test_cases[idx, 1]
    #     f1 = test_cases[idx, 2]
    #     f2 = test_cases[idx, 3]
    #     tau = planar_jac_transpose_np(l1, l2, th1, th2, f1, f2)
    #     print(th1, th2, f1, f2, tau)

    # reverse_homog_ca = derive_reverse_homog_ca()
    # T = ca.SX.sym("T", 4, 4)
    # print(reverse_homog_ca(T))
    # T = homog_np(p, R)
    # print(T @ reverse_homog_ca(T))
    # print(reverse_homog_ca(T) @ T)

    # print("\ntest extract_state_ca")
    # X = ca.SX.sym("X", 18, 3)
    # U = ca.SX.sym("U", 24, 3)
    # p, R, pdot, omega, p_i, f_i = extract_state_ca(X, U, 0)
    # print("p:", p)
    # print("R:", R)
    # print("pdot:", pdot)
    # print("omega:", omega)
    # for leg in legs:
    #     print("p_i[", leg.value, "]:", p_i[leg])
    #     print("f_i[", leg.value, "]:", f_i[leg])

    # print("\ntest flatten_state_np")
    # p = np.array([1.0, 2.0, 3.0])
    # R = rot_mat_np(np.array([0, 1, 0]), np.pi / 4.0)
    # pdot = np.array([0.4, 0.5, 0.6])
    # omega = np.array([3, 4, 5])
    # p_i = {}
    # f_i = {}
    # for leg in legs:
    #     p_i[leg] = leg.value + np.array([0.7, 0.8, 0.9])
    #     f_i[leg] = leg.value + np.array([0.07, 0.08, 0.09])

    # X_k, U_k = flatten_state_np(p, R, pdot, omega, p_i, f_i)
    # print(X_k)
    # print(U_k)

    # print("\ntest extract_state_np")
    # (
    #     p_extracted,
    #     R_extracted,
    #     pdot_extracted,
    #     omega_extracted,
    #     p_i_extracted,
    #     f_i_extracted,
    # ) = extract_state_np(X_k[:, np.newaxis], U_k[:, np.newaxis], 0)
    # print("p_extracted", p_extracted)
    # print("R_extracted", R_extracted)
    # print("pdot_extracted", pdot_extracted)
    # print("omega_extracted", omega_extracted)
    # print("p_i_extracted", p_i_extracted)
    # print("f_i_extracted", f_i_extracted)

    # calculation used to find suitable value for vertical force upper limit
    force_limit = planar_jac_inv_transpose_np(
        l_thigh, l_calf, np.pi / 4.0, np.pi / 2.0, -2.0, 2.0
    )
    print(force_limit)

    torque_limit = planar_jac_transpose_np(
        l_thigh, l_calf, np.pi / 4.0, np.pi / 2.0, 0.0, 20.0
    )
    print(torque_limit)

    # # calculation used to debug angle wrapping issue in 07-05-180-backflip trajectory
    # p = np.array([-2.22904149e-01, 1.86470565e-14, 3.07146706e-01])
    # R = np.array(
    #     [
    #         [-8.47305219e-01, 3.74376852e-14, -5.31291236e-01],
    #         [-3.93408699e-13, 1.00000000e00, 6.97888883e-13],
    #         [5.31291236e-01, 8.00304925e-13, -8.47305219e-01],
    #     ]
    # )
    # p_i = {}
    # p_i[leg.FL] = np.array([-0.60159766, 0.15, 0.32938683])
    # p_i[leg.FR] = np.array([-0.60159766, -0.15, 0.32938683])
    # p_i[leg.HL] = np.array([-1.90037668e-01, 1.50000000e-01, -9.85315547e-09])
    # p_i[leg.HR] = np.array([-1.90037668e-01, -1.50000000e-01, -9.85315547e-09])
    # p_next = np.array([-2.27567091e-01, 2.96891007e-15, 3.02150509e-01])
    # R_next = np.array(
    #     [
    #         [-8.73350678e-01, 3.99658078e-16, -4.87295136e-01],
    #         [-4.11243103e-13, 1.00000000e00, 7.37879702e-13],
    #         [4.87295136e-01, 8.44789798e-13, -8.73350678e-01],
    #     ]
    # )
    # p_i_next = {}
    # p_i_next[leg.FL] = np.array([-0.60348592, 0.15, 0.29296728])
    # p_i_next[leg.FR] = np.array([-0.60348592, -0.15, 0.29296728])
    # p_i_next[leg.HL] = np.array([-1.90037803e-01, 1.50000000e-01, -9.87121367e-09])
    # p_i_next[leg.HR] = np.array([-1.90037803e-01, -1.50000000e-01, -9.87121367e-09])

    # f_i = {}
    # for leg in legs:
    #     f_i[leg] = np.zeros(3)

    # from draw import *
    # init_fig()
    # draw(p, R, p_i, f_i)
    # draw(p_next, R_next, p_i_next, f_i)
    # plt.show()

    import ipdb

    ipdb.set_trace()
