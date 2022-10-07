from constants import *
from utils import (
    legs,
    derive_skew_ca,
    derive_rot_mat_ca,
    derive_homog_ca,
    derive_reverse_homog_ca,
    derive_mult_homog_point_ca,
    B_T_Bi,
    extract_state_np,
    extract_state_ca,
)
from draw import animate_traj
from generate_reference import generate_reference
import numpy as np
import casadi as ca


def traj_opt(X_ref, U_ref, dt, motion_options={}):
    symmetry = motion_options["symmetry"] if "symmetry" in motion_options else None

    skew_ca = derive_skew_ca()
    rot_mat_ca = derive_rot_mat_ca()
    homog_ca = derive_homog_ca()
    reverse_homog_ca = derive_reverse_homog_ca()
    mult_homog_point_ca = derive_mult_homog_point_ca()

    N = X_ref.shape[1] - 1

    opti = ca.Opti()
    X = opti.variable(18, N + 1)
    U = opti.variable(24, N + 1)
    J = ca.MX(1, 1)

    for k in range(N + 1):
        # extract state
        p, R, pdot, omega, p_i, f_i = extract_state_ca(X, U, k)
        if k != N:
            (
                p_next,
                R_next,
                pdot_next,
                omega_next,
                p_i_next,
                f_i_next,
            ) = extract_state_ca(X, U, k + 1)
        else:
            p_next, R_next, pdot_next, omega_next, p_i_next, f_i_next = (
                None,
                None,
                None,
                None,
                None,
                None,
            )

        # extract reference
        p_ref, R_ref, pdot_ref, omega_ref, p_i_ref, f_i_ref = extract_state_ca(
            X_ref, U_ref, k
        )

        # calculate relative foot locations
        T_B = homog_ca(p, R)
        Bi_p_i = {}
        for leg in legs:
            T_Bi_leg = T_B @ B_T_Bi[leg]
            Bi_T_leg = reverse_homog_ca(T_Bi_leg)
            Bi_p_i[leg] = mult_homog_point_ca(Bi_T_leg, p_i[leg])

        # objective function
        error_p = p - p_ref
        error_p_i = {}
        for leg in legs:
            error_p_i[leg] = p_i[leg] - p_i_ref[leg]
        weighted_error_R = ca.trace(Gp - Gp @ R_ref.T @ R)
        error_pdot = pdot - pdot_ref
        error_omega = omega - R.T @ R_ref @ omega_ref
        error_f_i = {}
        for leg in legs:
            error_f_i[leg] = f_i[leg] - f_i_ref[leg]
        J += ca.dot(Q_p * error_p, error_p)
        for leg in legs:
            J += ca.dot(Q_p_i * error_p_i[leg], error_p_i[leg])
        J += weighted_error_R**2
        J += ca.dot(Q_pdot * error_pdot, error_pdot)
        J += ca.dot(Q_omega * error_omega, error_omega)
        for leg in legs:
            J += ca.dot(Q_f_i * error_f_i[leg], error_f_i[leg])
        if k != N:
            p_i_dot = {}
            for leg in legs:
                p_i_dot[leg] = (p_i_next[leg] - p_i[leg]) / dt
                J += ca.dot(R_p_i_dot * p_i_dot[leg], p_i_dot[leg])

        # dynamics constraints
        f = ca.MX(3, 1)
        tau = ca.MX(3, 1)
        for leg in legs:
            f += f_i[leg]
            tau += ca.cross(p_i[leg] - p, f_i[leg])
        if k != N:
            # 3D dynamics
            opti.subject_to(p_next == p + pdot * dt)
            opti.subject_to(pdot_next == pdot + (f / m + g) * dt)
            opti.subject_to(R_next == R @ rot_mat_ca(omega, dt))
            opti.subject_to(
                omega_next
                == omega + B_I_inv @ (R.T @ tau - skew_ca(omega) @ B_I @ omega) * dt
            )

            # # 2D dynamics constrainted to x-z plane
            # opti.subject_to(p_next[::2] == p[::2] + pdot[::2] * dt)
            # opti.subject_to(pdot_next[::2] == pdot[::2] + (f[::2] / m + g[::2]) * dt)
            # opti.subject_to(R_next == R @ rot_mat_ca(omega, dt))
            # opti.subject_to(omega_next[1] == omega[1] + B_I_inv[1, 1] * tau[1] * dt)
            # opti.subject_to(opti.bounded(-eps, p[1], eps))
            # opti.subject_to(opti.bounded(-eps, pdot[1], eps))
            # opti.subject_to(opti.bounded(-eps, omega[0], eps))
            # opti.subject_to(opti.bounded(-eps, omega[2], eps))

        # kinematics constraints
        for leg in legs:
            # L1 norm constraint in the shoulder plane
            opti.subject_to(
                opti.bounded(-kin_lim, Bi_p_i[leg][0] + Bi_p_i[leg][2], kin_lim)
            )
            opti.subject_to(
                opti.bounded(-kin_lim, Bi_p_i[leg][0] - Bi_p_i[leg][2], kin_lim)
            )
            # y position should be on shoulder plane (within some numerical tolerance)
            opti.subject_to(opti.bounded(-eps, Bi_p_i[leg][1], eps))

        # symmetry constraints
        if symmetry == "sideways":
            opti.subject_to(
                opti.bounded(-eps, Bi_p_i[legs.FL][0] - Bi_p_i[legs.FR][0], eps)
            )
            opti.subject_to(
                opti.bounded(-eps, Bi_p_i[legs.HL][0] - Bi_p_i[legs.HR][0], eps)
            )
            # kinematic constraint already enforces y coord == 0
            opti.subject_to(
                opti.bounded(-eps, Bi_p_i[legs.FL][2] - Bi_p_i[legs.FR][2], eps)
            )
            opti.subject_to(
                opti.bounded(-eps, Bi_p_i[legs.HL][2] - Bi_p_i[legs.HR][2], eps)
            )
            opti.subject_to(opti.bounded(-eps, f_i[legs.FL][0] - f_i[legs.FR][0], eps))
            opti.subject_to(opti.bounded(-eps, f_i[legs.HL][0] - f_i[legs.HR][0], eps))
            opti.subject_to(opti.bounded(-eps, f_i[legs.FL][1] - f_i[legs.FR][1], eps))
            opti.subject_to(opti.bounded(-eps, f_i[legs.HL][1] - f_i[legs.HR][1], eps))
            opti.subject_to(opti.bounded(-eps, f_i[legs.FL][2] - f_i[legs.FR][2], eps))
            opti.subject_to(opti.bounded(-eps, f_i[legs.HL][2] - f_i[legs.HR][2], eps))
        elif symmetry == "diagonal":
            # symmetry constraints
            opti.subject_to(
                opti.bounded(-eps, Bi_p_i[legs.FL][0] - Bi_p_i[legs.HR][0], eps)
            )
            opti.subject_to(
                opti.bounded(-eps, Bi_p_i[legs.HL][0] - Bi_p_i[legs.FR][0], eps)
            )
            # kinematic constraint already enforces y coord == 0
            opti.subject_to(
                opti.bounded(-eps, Bi_p_i[legs.FL][2] - Bi_p_i[legs.HR][2], eps)
            )
            opti.subject_to(
                opti.bounded(-eps, Bi_p_i[legs.HL][2] - Bi_p_i[legs.FR][2], eps)
            )
            # vector bounded by scalar -> multiple scalar constraints
            opti.subject_to(opti.bounded(-eps, f_i[legs.FL][0] - f_i[legs.HR][0], eps))
            opti.subject_to(opti.bounded(-eps, f_i[legs.HL][0] - f_i[legs.FR][0], eps))
            opti.subject_to(opti.bounded(-eps, f_i[legs.FL][1] - f_i[legs.HR][1], eps))
            opti.subject_to(opti.bounded(-eps, f_i[legs.HL][1] - f_i[legs.FR][1], eps))
            opti.subject_to(opti.bounded(-eps, f_i[legs.FL][2] - f_i[legs.HR][2], eps))
            opti.subject_to(opti.bounded(-eps, f_i[legs.HL][2] - f_i[legs.FR][2], eps))

        # friction pyramid constraints
        for leg in legs:
            opti.subject_to(opti.bounded(0.0, f_i[leg][2], f_lim))
            opti.subject_to(
                opti.bounded(-mu * f_i[leg][2], f_i[leg][0], mu * f_i[leg][2])
            )
            opti.subject_to(
                opti.bounded(-mu * f_i[leg][2], f_i[leg][1], mu * f_i[leg][2])
            )

        # contact constraints
        for leg in legs:
            opti.subject_to(p_i[leg][2] >= 0.0)
            opti.subject_to(opti.bounded(-eps, f_i[leg][2] * p_i[leg][2], eps))
            if k != N:
                opti.subject_to(
                    opti.bounded(
                        -eps, f_i[leg][2] * (p_i_next[leg][0] - p_i[leg][0]), eps
                    )
                )
                opti.subject_to(
                    opti.bounded(
                        -eps, f_i[leg][2] * (p_i_next[leg][1] - p_i[leg][1]), eps
                    )
                )

    # apply objective function
    opti.minimize(J)

    # initial conditions constraint
    p_init, R_init, pdot_init, omega_init, p_i_init, f_i_init = extract_state_ca(
        X, U, 0
    )
    (
        p_ref_init,
        R_ref_init,
        pdot_ref_init,
        omega_ref_init,
        p_i_ref_init,
        f_i_ref_init,
    ) = extract_state_np(X_ref, U_ref, 0)
    opti.subject_to(p_init == p_ref_init)
    opti.subject_to(R_init == R_ref_init)
    for leg in legs:
        opti.subject_to(p_i_init[leg] == p_i_ref_init[leg])
    # # turn off below for cyclic constraint (biped step)
    opti.subject_to(pdot_init == np.zeros_like(pdot_ref_init))
    opti.subject_to(omega_init == np.zeros_like(omega_ref_init))
    # note: no initial condition on force

    # # cyclic constraint (for biped-step)
    # p_final, R_final, pdot_final, omega_final, p_i_final, f_i_final = extract_state_ca(
    #     X, U, N
    # )
    # opti.subject_to(ca.sumsqr(p_init - p_final) < eps)
    # opti.subject_to(ca.sumsqr(R_init - R_final) < eps)
    # for leg in legs:
    #     opti.subject_to(ca.sumsqr(p_i_init[leg] - p_i_final[leg]) < eps)
    #     opti.subject_to(ca.sumsqr(f_i_init[leg] - f_i_final[leg]) < eps)
    # opti.subject_to(ca.sumsqr(pdot_init - pdot_final) < eps)
    # opti.subject_to(ca.sumsqr(omega_init - omega_final) < eps)

    # initial solution guess
    opti.set_initial(X, X_ref)
    opti.set_initial(U, U_ref)

    # solve NLP
    p_opts = {}
    s_opts = {"print_level": 5}
    opti.solver("ipopt", p_opts, s_opts)
    sol = opti.solve()

    # extract solution as numpy array
    X_sol = np.array(sol.value(X))
    U_sol = np.array(sol.value(U))

    return X_sol, U_sol


if __name__ == "__main__":
    X_ref, U_ref, dt, motion_options = generate_reference()
    # animate_traj(X_ref, U_ref, dt, motion_options=motion_options)

    X_sol, U_sol = traj_opt(X_ref, U_ref, dt)
    # animate_traj(X_sol, U_sol, dt, motion_options=motion_options)
