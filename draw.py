from constants import *
from utils import (
    legs,
    rot_mat_2d_np,
    homog_np,
    mult_homog_point_np,
    B_T_Bi,
    extract_state_np,
    solo_IK_np,
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("seaborn")


# draws a coordinate system defined by the 4x4 homogeneous transformation matrix T
def draw_T(T):
    axis_len = 0.1
    origin = T[:3, 3]
    axis_colors = ["r", "g", "b"]
    for axis in range(3):
        axis_head = origin + axis_len * T[:3, axis]
        axis_coords = np.vstack((origin, axis_head)).T

        line = plt.plot([], [])[0]
        line.set_data(axis_coords[0], axis_coords[1])
        line.set_3d_properties(axis_coords[2])
        line.set_color(axis_colors[axis])


def draw(p, R, p_i, f_i, f_len=0.02, motion_options={}):
    elbow_up_front = (
        motion_options["elbow_up_front"] if "elbow_up_front" in motion_options else None
    )
    elbow_up_hind = (
        motion_options["elbow_up_hind"] if "elbow_up_hind" in motion_options else None
    )

    T_B = homog_np(p, R)
    p_Bi = {}
    for leg in legs:
        p_Bi[leg] = mult_homog_point_np(T_B, B_p_Bi[leg])

    # draw body
    body_coords = np.vstack(
        (p_Bi[legs.FL], p_Bi[legs.FR], p_Bi[legs.HR], p_Bi[legs.HL], p_Bi[legs.FL])
    ).T
    line = plt.plot([], [])[0]
    line.set_data(body_coords[0], body_coords[1])
    line.set_3d_properties(body_coords[2])
    line.set_color("b")
    line.set_marker("o")

    # inverse and forward kinematics to extract knee location
    q_i = solo_IK_np(p, R, p_i, elbow_up_front, elbow_up_hind)
    p_knee_i = {}
    p_foot_i = {}
    for leg in legs:
        Bi_xz_knee = rot_mat_2d_np(q_i[leg][0] - np.pi / 2.0) @ np.array([l_thigh, 0.0])
        Bi_xz_foot = Bi_xz_knee + rot_mat_2d_np(
            q_i[leg][0] - np.pi / 2.0 + q_i[leg][1]
        ) @ np.array([l_calf, 0.0])
        Bi_p_knee_i = np.array([Bi_xz_knee[0], 0.0, Bi_xz_knee[1]])
        Bi_p_foot_i = np.array([Bi_xz_foot[0], 0.0, Bi_xz_foot[1]])
        T_Bi = T_B @ B_T_Bi[leg]
        p_knee_i[leg] = mult_homog_point_np(T_Bi, Bi_p_knee_i)
        p_foot_i[leg] = mult_homog_point_np(T_Bi, Bi_p_foot_i)

    # ensure foot positions match the values calculated from IK and FK
    # note that the y position of the legs are allowed to deviate from 0 by
    # amount eps in the kinematics constraint, so we use something larger here
    # to check if the error is "not close to zero"
    for leg in legs:
        assert np.linalg.norm(p_foot_i[leg] - p_i[leg]) < np.sqrt(eps)

    # draw legs
    for leg in legs:
        leg_coords = np.vstack((p_Bi[leg], p_knee_i[leg], p_i[leg])).T
        line = plt.plot([], [])[0]
        line.set_data(leg_coords[0], leg_coords[1])
        line.set_3d_properties(leg_coords[2])
        line.set_color("g")
        line.set_marker("o")

    # draw ground reaction forces
    f_coords = {}
    for leg in legs:
        f_vec = p_i[leg] + f_len * f_i[leg]
        f_coords[leg] = np.vstack((p_i[leg], f_vec)).T
        line = plt.plot([], [])[0]
        line.set_data(f_coords[leg][0], f_coords[leg][1])
        line.set_3d_properties(f_coords[leg][2])
        line.set_color("r")

    draw_T(np.eye(4))
    draw_T(T_B)


def init_fig():
    anim_fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(anim_fig, auto_add_to_figure=False)
    anim_fig.add_axes(ax)
    ax.view_init(azim=-45)
    ax.set_xlim3d([-0.5, 0.5])
    ax.set_ylim3d([-0.5, 0.5])
    ax.set_zlim3d([0, 1])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return anim_fig, ax


def animate_traj(X, U, dt, fname=None, display=True, repeat=True, motion_options={}):
    anim_fig, ax = init_fig()

    def draw_frame(k):
        p, R, pdot, omega, p_i, f_i = extract_state_np(X, U, k)
        while ax.lines:
            ax.lines.pop()
        draw(p, R, p_i, f_i, motion_options=motion_options)

    N = X.shape[1] - 1

    anim = animation.FuncAnimation(
        anim_fig,
        draw_frame,
        frames=N + 1,
        interval=dt * 1000.0,
        repeat=repeat,
        blit=False,
    )

    if fname is not None:
        print("saving animation at videos/" + fname + ".mp4...")
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=int(1 / dt), metadata=dict(artist="Me"), bitrate=1000)
        anim.save("videos/" + fname + ".mp4", writer=writer)
        print("finished saving videos/" + fname + ".mp4")

    if display:
        plt.show()


if __name__ == "__main__":
    from utils import rot_mat_np

    p = np.array([0.0, 0.0, 0.3])
    R = rot_mat_np(np.array([0, 1, 0]), 0.1)
    p_i = {}
    f_i = {}
    for leg in legs:
        p_i[leg] = B_p_Bi[leg]
        f_i[leg] = np.array([0.0, 0.0, 3.0])

    anim_fig, ax = init_fig()
    draw(p=p, R=R, p_i=p_i, f_i=f_i)
    plt.show()
