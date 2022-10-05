import enum
import numpy as np

# enum for the four legs
class legs(enum.Enum):
    FL = 0
    FR = 1
    HL = 2
    HR = 3


# robot physical length paramters
l_Bx = 0.380  # length of body, measured axis to axis at hip (from CAD)
# width between left and right feet (from CAD)
# outer face to face was 310mm, inner face to face was 290mm, thickness of lower leg is 10mm
l_By = 0.3
l_thigh = 0.165  # length of upper leg module measured axis to axis (from CAD)
l_calf = 0.160  # length of lower leg measured axis to axis (from CAD)

# robot inertial paramters
# mass of entire robot, physically measured and consistent with ODRI documentation
# https://github.com/open-dynamic-robot-initiative/open_robot_actuator_hardware/blob/master/mechanics/quadruped_robot_8dof_v2/README.md#description
m = 1.7
# moment of inertia of only the body, taken from ODRI documentation linked below, approximated to be diagonal
# ideally it would be the moment of inertia of the whole robot (to include upper leg motor weights), but I don't have this info available
# https://github.com/open-dynamic-robot-initiative/open_robot_actuator_hardware/blob/master/mechanics/quadruped_robot_8dof_v2/details/quadruped_8dof_v2_inertia.pdf
B_I = np.diag([0.00533767, 0.01314118, 0.01821833])
B_I_inv = np.diag(1 / np.array([0.00533767, 0.01314118, 0.01821833]))

# physical parameters external to robot
g = np.array([0.0, 0.0, -9.81])  # gravity vector
mu = 0.9  # friction coefficient

# position of corners of robot, in body frame (so it's a constant)
B_p_Bi = {}
B_p_Bi[legs.FL] = np.array([l_Bx / 2.0, l_By / 2.0, 0.0])
B_p_Bi[legs.FR] = np.array([l_Bx / 2.0, -l_By / 2.0, 0.0])
B_p_Bi[legs.HL] = np.array([-l_Bx / 2.0, l_By / 2.0, 0.0])
B_p_Bi[legs.HR] = np.array([-l_Bx / 2.0, -l_By / 2.0, 0.0])

# global optimization paramters
eps = 1e-6  # numerical zero threshold

# limit constraint paramters
kin_lim = l_thigh + l_calf  # used for L1 norm kinematic constraint
f_lim = 20.0  # max vertical force in newtons
qdot_lim = 100.0  # joint velocity clip value in export()

# LQR weights
Q_p = np.array([1000.0, 1000.0, 1000.0])
Q_p_i = np.array([500.0, 500.0, 500.0])
Q_R = np.array([100.0, 100.0, 100.0])
Q_pdot = np.array([10.0, 10.0, 10.0])
Q_omega = np.array([1.0, 1.0, 1.0])
Q_f_i = np.array([0.1, 0.1, 0.1])
R_p_i_dot = np.array([1.0, 1.0, 1.0])
# R_p_i_dot = np.array([10.0, 10.0, 10.0]) # jump

# matrix used for rotation matrix cost, calculated from above values
Kp_vec = np.linalg.solve(
    np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]), 4.0 * Q_R
)  # 3 element vector
Gp = sum(Kp_vec) * np.eye(3) - np.diag(Kp_vec)  # 3x3 matrix
