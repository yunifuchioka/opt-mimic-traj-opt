from constants import *
from draw import animate_traj
from utils import (
    B_T_Bi,
    legs,
    mult_homog_point_np,
    rot_mat_2d_np,
    rot_mat_np,
    flatten_state_np,
    homog_np,
)
import numpy as np
from numpy import pi
from scipy.interpolate import interp1d
from scipy.interpolate import CubicHermiteSpline


# linearly interpolate x and y, evaluate at t
def linear_interp_t(x, y, t):
    f = interp1d(x, y)
    return f(t)


# interpolate x and y, evaluate at t using cubic splines with zero deriatives
# this creates an interpolation similar to linear interpolation, but with
# smoothed corners
def cubic_interp_t(x, y, t):
    f = CubicHermiteSpline(x, y, np.zeros_like(x))
    return f(t)


# sinusoidal function evaluated at t defined using oscillation period, minimum
# and maximum values
def sinusoid(period, min_val, max_val, t, phase_offset=0):
    return (max_val - min_val) / 2.0 * (
        1 - np.cos(2 * np.pi / period * t + phase_offset)
    ) + min_val


def generate_reference():
    motion_type = "biped-step"

    if motion_type == "stand":
        tf = 10.0
    if motion_type == "squat":
        tf = 10.0
    if motion_type == "trot":
        tf = 10.0
    if motion_type == "bound":
        tf = 10.0
    if motion_type == "pronk":
        tf = 10.0
    if motion_type == "jump":
        tf = 2.0
    elif motion_type == "front-hop":
        tf = 5.0
    elif motion_type == "180-backflip":
        tf = 5.0
    elif motion_type == "180-frontflip":
        tf = 5.0
    if motion_type == "backflip":
        tf = 5.0
    elif motion_type == "back-cartwheel":
        tf = 10.0
    if motion_type == "biped-stand":
        tf = 5.0
    if motion_type == "biped-step":
        tf = 10.0

    N = int(tf * 50)
    dt = tf / (N)
    t_vals = np.linspace(0, tf, N + 1)

    X = np.zeros((18, N + 1))
    U = np.zeros((24, N + 1))

    motion_options = {}

    for k in range(N + 1):
        if motion_type == "stand":
            p = np.array([0.0, 0.0, 0.225])
            R = np.eye(3)
            p_i = {}
            for leg in legs:
                p_i[leg] = B_p_Bi[leg].copy()
            motion_options["elbow_up_front"] = True
            motion_options["elbow_up_hind"] = False
            motion_options["symmetry"] = "sideways"
        if motion_type == "squat":
            if k * dt < 0.5 or k * dt > 9.5:
                body_z = 0.2
            else:
                body_z = sinusoid(
                    period=1.0,
                    min_val=0.15,
                    max_val=0.25,
                    t=t_vals[k],
                    phase_offset=np.pi * 0.5,
                )
            p = np.array([0.0, 0.0, body_z])
            R = np.eye(3)
            p_i = {}
            for leg in legs:
                p_i[leg] = B_p_Bi[leg].copy()
            motion_options["elbow_up_front"] = True
            motion_options["elbow_up_hind"] = False
            motion_options["symmetry"] = "sideways"
        if motion_type == "trot":
            if k * dt < 0.5 or k * dt > 9.5:
                body_x = -0.3
            else:
                body_x = sinusoid(
                    period=9.0,
                    min_val=-0.3,
                    max_val=0.3,
                    t=t_vals[k],
                    phase_offset=-2 * np.pi / 9.0 * 0.5,
                )
            p = np.array([body_x, 0.0, 0.25])
            R = np.eye(3)
            p_i = {}
            for leg in legs:
                p_i[leg] = B_p_Bi[leg].copy()
                p_i[leg][0] += body_x
                if k * dt < 0.5 or k * dt > 9.5:
                    pass
                else:
                    if leg == legs.FL or leg == legs.HR:
                        p_i[leg][2] += max(
                            0.0, sinusoid(0.5, -0.05, 0.05, t_vals[k], pi / 2.0)
                        )
                    else:
                        p_i[leg][2] += max(
                            0.0, sinusoid(0.5, -0.05, 0.05, t_vals[k], 3.0 * pi / 2.0)
                        )
            motion_options["elbow_up_front"] = True
            motion_options["elbow_up_hind"] = False
            motion_options["symmetry"] = "diagonal"
        if motion_type == "bound":
            if k * dt < 0.5 or k * dt > 9.5:
                body_x = -0.3
            else:
                body_x = sinusoid(
                    period=9.0,
                    min_val=-0.3,
                    max_val=0.3,
                    t=t_vals[k],
                    phase_offset=-2 * np.pi / 9.0 * 0.5,
                )
            p = np.array([body_x, 0.0, 0.25])
            R = np.eye(3)
            p_i = {}
            for leg in legs:
                p_i[leg] = B_p_Bi[leg].copy()
                p_i[leg][0] += body_x
                if k * dt < 0.5 or k * dt > 9.5:
                    pass
                else:
                    if leg == legs.FL or leg == legs.FR:
                        p_i[leg][2] += max(
                            0.0, sinusoid(0.5, -0.05, 0.05, t_vals[k], pi / 2.0)
                        )
                    else:
                        p_i[leg][2] += max(
                            0.0, sinusoid(0.5, -0.05, 0.05, t_vals[k], 3.0 * pi / 2.0)
                        )
            motion_options["elbow_up_front"] = True
            motion_options["elbow_up_hind"] = False
            motion_options["symmetry"] = "sideways"
        if motion_type == "pronk":
            if k * dt < 0.5 or k * dt > 9.5:
                body_x = -0.3
            else:
                body_x = sinusoid(
                    period=9.0,
                    min_val=-0.3,
                    max_val=0.3,
                    t=t_vals[k],
                    phase_offset=-2 * np.pi / 9.0 * 0.5,
                )
            p = np.array([body_x, 0.0, 0.25])
            R = np.eye(3)
            p_i = {}
            for leg in legs:
                p_i[leg] = B_p_Bi[leg].copy()
                p_i[leg][0] += body_x
                if k * dt < 0.5 or k * dt > 9.5:
                    pass
                else:
                    p_i[leg][2] += max(
                        0.0, sinusoid(0.5, -0.05, 0.05, t_vals[k], pi / 2.0)
                    )
            motion_options["elbow_up_front"] = True
            motion_options["elbow_up_hind"] = False
            motion_options["symmetry"] = "sideways"
        if motion_type == "jump":
            t_apex = 0.25
            z_apex = np.linalg.norm(g) * t_apex**2 / 2.0
            body_z = cubic_interp_t(
                [0, 0.2 * tf, 0.2 * tf + t_apex, 0.2 * tf + 2 * t_apex, tf],
                [0, 0, z_apex, 0, 0],
                t_vals[k],
            )
            p = np.array([0.0, 0.0, 0.2])
            p[2] += body_z
            R = np.eye(3)
            p_i = {}
            for leg in legs:
                p_i[leg] = B_p_Bi[leg].copy()
                # p_i[leg][2] += body_z
                p_i[leg][2] = max(0.0, p[2] - (l_thigh + l_calf) * 0.95)
            motion_options["elbow_up_front"] = True
            motion_options["elbow_up_hind"] = False
            motion_options["symmetry"] = "sideways"
        elif motion_type == "front-hop":
            body_height = 0.2
            angle = cubic_interp_t(
                [0, 2.0, 2.5, 3.0, tf], [0, 0, np.pi / 4.0, 0, 0], t_vals[k]
            )
            p = np.array([-l_Bx / 2.0, 0.0, body_height])
            p_xz = rot_mat_2d_np(angle) @ np.array([l_Bx / 2.0, 0.0])
            p += np.array([p_xz[0], 0.0, p_xz[1]])
            R = rot_mat_np(np.array([0.0, 1.0, 0.0]), -angle)
            p_i = {}
            T_B = homog_np(p, R)
            for leg in legs:
                if leg == legs.FL or leg == legs.FR:
                    p_Bi = mult_homog_point_np(T_B, B_p_Bi[leg])
                    p_i[leg] = p_Bi.copy()
                    p_i[leg][2] -= body_height
                else:
                    p_i[leg] = B_p_Bi[leg].copy()
            motion_options["elbow_up_front"] = True
            motion_options["elbow_up_hind"] = False
            motion_options["symmetry"] = "sideways"
        elif motion_type == "180-backflip":
            body_height = 0.2
            angle = cubic_interp_t([0, 2.0, 3.2, tf], [0, 0, np.pi, np.pi], t_vals[k])
            p = np.array([-l_Bx / 2.0, 0.0, body_height])
            p_xz = rot_mat_2d_np(angle) @ np.array([l_Bx / 2.0, 0.0])
            p += np.array([p_xz[0], 0.0, p_xz[1]])
            R = rot_mat_np(np.array([0.0, 1.0, 0.0]), -angle)
            p_i = {}
            T_B = homog_np(p, R)
            for leg in legs:
                if leg == legs.FL or leg == legs.FR:
                    p_Bi = mult_homog_point_np(T_B, B_p_Bi[leg])
                    p_i[leg] = p_Bi.copy()
                    # p_i[leg][2] -= body_height
                    p_i[leg][0:3:2] += rot_mat_2d_np(2.0 * angle) @ np.array(
                        [0.0, -body_height]
                    )
                else:
                    p_i[leg] = B_p_Bi[leg].copy()
            motion_options["elbow_up_front"] = True
            motion_options["elbow_up_hind"] = True
            motion_options["symmetry"] = "sideways"
        elif motion_type == "180-frontflip":
            body_height = 0.2
            angle = cubic_interp_t([0, 2.0, 3.2, tf], [0, 0, -np.pi, -np.pi], t_vals[k])
            p = np.array([l_Bx / 2.0, 0.0, body_height])
            p_xz = rot_mat_2d_np(angle) @ np.array([-l_Bx / 2.0, 0.0])
            p += np.array([p_xz[0], 0.0, p_xz[1]])
            R = rot_mat_np(np.array([0.0, 1.0, 0.0]), -angle)
            p_i = {}
            T_B = homog_np(p, R)
            for leg in legs:
                if leg == legs.HL or leg == legs.HR:
                    p_Bi = mult_homog_point_np(T_B, B_p_Bi[leg])
                    p_i[leg] = p_Bi.copy()
                    # p_i[leg][2] -= body_height
                    p_i[leg][0:3:2] += rot_mat_2d_np(2.0 * angle) @ np.array(
                        [0.0, -body_height]
                    )
                else:
                    p_i[leg] = B_p_Bi[leg].copy()
            motion_options["elbow_up_front"] = True
            motion_options["elbow_up_hind"] = True
            motion_options["symmetry"] = "sideways"
        if motion_type == "backflip":
            t_apex = 0.2
            z_apex = np.linalg.norm(g) * t_apex**2 / 2.0
            body_z = linear_interp_t(
                [
                    0.0,
                    1.6,
                    2.0,
                    2.0 + t_apex,
                    2.0 + 2 * t_apex,
                    2.0 + 2 * t_apex + 0.4,
                    tf,
                ],
                [0.0, 0.0, l_Bx / 2.0, l_Bx / 2.0 + z_apex, l_Bx / 2.0, 0.0, 0.0],
                t_vals[k],
            )
            body_x = linear_interp_t(
                [0.0, 1.6, 2.0, 2.0 + 2 * t_apex, 2.0 + 2 * t_apex + 0.4, tf],
                [0.0, 0.0, -l_Bx / 2.0, -3.0 / 2.0 * l_Bx, -2.0 * l_Bx, -2.0 * l_Bx],
                t_vals[k],
            )
            angle = linear_interp_t(
                [0, 1.6, 2.0, 2.0 + 2 * t_apex, 2.0 + 2 * t_apex + 0.4, tf],
                [0.0, 0.0, np.pi / 2.0, 3.0 / 2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi],
                t_vals[k],
            )
            p = np.array([0.0, 0.0, 0.2])
            p[0] += body_x
            p[2] += body_z
            R = rot_mat_np(np.array([0.0, 1.0, 0.0]), -angle)
            p_i = {}
            T_B = homog_np(p, R)
            if t_vals[k] < 2.0:
                for leg in legs:
                    if leg == legs.FL or leg == legs.FR:
                        p_Bi = mult_homog_point_np(T_B, B_p_Bi[leg])
                        p_i[leg] = p_Bi.copy()
                        p_i[leg][0:3:2] += rot_mat_2d_np(2.0 * angle) @ np.array(
                            [0.0, -0.2]
                        )
                    else:
                        p_i[leg] = B_p_Bi[leg].copy()
            elif t_vals[k] > 2.0 + 2 * t_apex:
                for leg in legs:
                    if leg == legs.HL or leg == legs.HR:
                        p_Bi = mult_homog_point_np(T_B, B_p_Bi[leg])
                        p_i[leg] = p_Bi.copy()
                        p_i[leg][0:3:2] += rot_mat_2d_np(2.0 * angle) @ np.array(
                            [0.0, -0.2]
                        )
                    else:
                        p_i[leg] = B_p_Bi[leg].copy()
                        p_i[leg][0] -= 2.0 * l_Bx
            else:
                for leg in legs:
                    B_T_Bi_leg = B_T_Bi[leg].copy()
                    if leg == legs.FR or leg == legs.FL:
                        B_T_Bi_leg[0, 3] += 0.2
                    else:
                        B_T_Bi_leg[0, 3] -= 0.2
                    T_B_i = T_B @ B_T_Bi_leg
                    p_i[leg] = T_B_i[0:3, 3]
            motion_options["elbow_up_front"] = True
            motion_options["elbow_up_hind"] = True
            motion_options["symmetry"] = "sideways"
        elif motion_type == "back-cartwheel":
            body_height = 0.2
            angle = cubic_interp_t(
                [0, 2.0, 3.2, 7.0, 8.2, tf],
                [0, 0, np.pi, np.pi, 2.0 * np.pi, 2.0 * np.pi],
                t_vals[k],
            )
            if t_vals[k] <= tf / 2.0:
                p = np.array([-l_Bx / 2.0, 0.0, body_height])
                p_xz = rot_mat_2d_np(angle) @ np.array([l_Bx / 2.0, 0.0])
                p += np.array([p_xz[0], 0.0, p_xz[1]])
                R = rot_mat_np(np.array([0.0, 1.0, 0.0]), -angle)
                p_i = {}
                T_B = homog_np(p, R)
                for leg in legs:
                    if leg == legs.FL or leg == legs.FR:
                        p_Bi = mult_homog_point_np(T_B, B_p_Bi[leg])
                        p_i[leg] = p_Bi.copy()
                        # p_i[leg][2] -= body_height
                        p_i[leg][0:3:2] += rot_mat_2d_np(2.0 * angle) @ np.array(
                            [0.0, -body_height]
                        )
                    else:
                        p_i[leg] = B_p_Bi[leg].copy()
            else:
                p = np.array([-l_Bx * 3.0 / 2.0, 0.0, body_height])
                p_xz = rot_mat_2d_np(angle - np.pi) @ np.array([l_Bx / 2.0, 0.0])
                p += np.array([p_xz[0], 0.0, p_xz[1]])
                R = rot_mat_np(np.array([0.0, 1.0, 0.0]), -angle)
                p_i = {}
                T_B = homog_np(p, R)
                for leg in legs:
                    if leg == legs.HL or leg == legs.HR:
                        p_Bi = mult_homog_point_np(T_B, B_p_Bi[leg])
                        p_i[leg] = p_Bi.copy()
                        # p_i[leg][2] -= body_height
                        p_i[leg][0:3:2] += rot_mat_2d_np(2.0 * angle) @ np.array(
                            [0.0, -body_height]
                        )
                    elif leg == legs.FL:
                        p_i[leg] = B_p_Bi[legs.HL].copy()
                        p_i[leg][0] -= l_Bx
                    elif leg == legs.FR:
                        p_i[leg] = B_p_Bi[legs.HR].copy()
                        p_i[leg][0] -= l_Bx
            motion_options["elbow_up_front"] = True
            motion_options["elbow_up_hind"] = True
            motion_options["symmetry"] = "sideways"
        if motion_type == "biped-stand":
            p = np.array([0.0, 0.0, l_Bx / 2.0 + (l_thigh + l_calf) * 0.95])
            R = rot_mat_np([0.0, 1.0, 0.0], -np.pi / 2.0)
            T_B = homog_np(p, R)
            p_i = {}
            T_B_i = {}
            for leg in legs:
                T_B_i[leg] = T_B @ B_T_Bi[leg]
                p_i[leg] = T_B_i[leg][0:3, 3]
                if leg == legs.HL or leg == legs.HR:
                    p_i[leg][2] = 0.0
                else:
                    p_i[leg][2] -= (l_thigh + l_calf) * 0.95
            motion_options["elbow_up_front"] = True
            motion_options["elbow_up_hind"] = False
            # no symmetry
        if motion_type == "biped-step":
            p = np.array([0.0, 0.0, l_Bx / 2.0 + (l_thigh + l_calf) * 0.95])
            R = rot_mat_np([0.0, 1.0, 0.0], -np.pi / 2.0)
            T_B = homog_np(p, R)
            p_i = {}
            T_B_i = {}
            for leg in legs:
                T_B_i[leg] = T_B @ B_T_Bi[leg]
                p_i[leg] = T_B_i[leg][0:3, 3]
            # if k * dt < 0.5 or k * dt > tf - 0.5:
            if False:
                for leg in legs:
                    if leg == legs.HL or leg == legs.HR:
                        p_i[leg][2] = 0.0
                    else:
                        p_i[leg][2] -= (l_thigh + l_calf) * 0.95
            else:
                p_i[leg.HL][2] = max(
                    0.0, sinusoid(0.5, -0.05, 0.05, t_vals[k], pi / 2.0)
                )
                p_i[leg.HR][2] = max(
                    0.0, sinusoid(0.5, -0.05, 0.05, t_vals[k], 3.0 * pi / 2.0)
                )
                arm_swing_amplitude = 0.0
                p_i[leg.FL][::2] += rot_mat_2d_np(
                    sinusoid(
                        0.5,
                        -arm_swing_amplitude - pi / 2.0,
                        arm_swing_amplitude - pi / 2.0,
                        t_vals[k],
                        3.0 * pi / 2.0,
                    )
                ) @ np.array([(l_thigh + l_calf) * 0.95, 0])
                p_i[leg.FR][::2] += rot_mat_2d_np(
                    sinusoid(
                        0.5,
                        -arm_swing_amplitude - pi / 2.0,
                        arm_swing_amplitude - pi / 2.0,
                        t_vals[k],
                        pi / 2.0,
                    )
                ) @ np.array([(l_thigh + l_calf) * 0.95, 0])
                motion_options["elbow_up_front"] = True
                motion_options["elbow_up_hind"] = False
                # no symmetry

        pdot = np.array([0.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, 0.0])
        f_i = {}
        for leg in legs:
            f_i[leg] = np.array([0.0, 0.0, 0.0])
            if p_i[leg][2] <= eps:
                # f_i[leg][2] = m * np.linalg.norm(g) / 4.0
                f_i[leg][2] = m * np.linalg.norm(g) # for biped-step
        X[:, k], U[:, k] = flatten_state_np(p, R, pdot, omega, p_i, f_i)

    return X, U, dt, motion_options


if __name__ == "__main__":

    X, U, dt, motion_options = generate_reference()

    animate_traj(X, U, dt, repeat=False, motion_options=motion_options)
