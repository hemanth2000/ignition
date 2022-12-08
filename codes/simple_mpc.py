import casadi as ca
import matplotlib.pyplot as plt
from utils.tiremodel import get_lateral_force, get_longitudinal_force
from utils.trajectory import get_ref_trajectory
from utils.utils import DM2Arr

# Simulation parameters
t_step = 0.05  # Time step in seconds
Nc = 3  # Control horizon : No of time steps optimal control is estimated
Np = 7  # Prediction window
Q = ca.diagcat(200, 75)  # State weighing matrix
R = 150  # Input weighing matrix
speed = 10
PSI = ca.DM([[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]])

# Vehicle parameters
a = 0.3  # Distance between front wheel and COM in meters
b = 0.7  # Distance between rear wheel and COM in meters
m = 2.05  # Mass of car in tons
g = 9.81  # Gravity
I = 3.344  # Moment of inertia
Fz = m * g  # Total normal force from ground
Fzf = b * Fz / (2 * (a + b))  # Normal force on front wheels
Fzr = a * Fz / (2 * (a + b))  # Normal force on rear wheels
mu = 0.85

# Model variables
x1 = ca.SX.sym("x_dot")
x2 = ca.SX.sym("y_dot")
x3 = ca.SX.sym("phi")
x4 = ca.SX.sym("phi_dot")
x5 = ca.SX.sym("X")
x6 = ca.SX.sym("Y")

x = ca.vertcat(x1, x2, x3, x4, x5, x6)  # input vector
u = ca.SX.sym("delta")  # front steering angle

vxf = x1  # front wheel velocity in x direction
vyf = x2 + a * x4  # front wheel velocity in y direction
vxr = x1  # rear wheel velocity in x direction
vyr = x2 - b * x4  # rear wheel velocity in y direction

vlf = vyf * ca.sin(u) + vxf * ca.cos(u)  # front wheel longitudinal velocity
vlr = vxf  # rear wheel longitudinal velocity
vcf = vyf * ca.cos(u) - vxf * ca.sin(u)  # front wheel lateral velocity
vcr = vyf  # rear wheel lateral velocity

alpha_f = ca.atan(vcf / vlf)
alpha_r = ca.atan(vcr / vlr)
slip_ratio_f = 0  # Assumption
slip_ratio_r = 0  # Assumption

Flf = get_longitudinal_force(mu * Fzf, slip_ratio_f)
Fcf = get_lateral_force(mu * Fzf, alpha_f)
Flr = get_longitudinal_force(mu * Fzr, slip_ratio_r)
Fcr = get_lateral_force(mu * Fzr, alpha_r)

Fxf = Flf * ca.cos(u) - Fcf * ca.sin(u)
Fyf = Flf * ca.sin(u) + Fcf * ca.cos(u)
Fxr = Flr
Fyr = Fcr

# Model equations
xdot = ca.vertcat(
    x2 * x4 + 2 * (Fxf + Fxr) / m,
    -1 * x1 * x4 + 2 * (Fyf + Fyr) / m,
    x4,
    2 * (a * Fyf - b * Fyr) / I,
    x1 * ca.cos(x3) - x2 * ca.sin(x3),
    x1 * ca.sin(x3) + x2 * ca.cos(x3),
)

f = ca.Function("f", [x, u], [xdot])

X = ca.SX.sym("X", 6)
P = ca.SX.sym("P", 8)

# Defining variables for NLP
J = 0
g = []

st = X

# Limits
delta_max = 10
delta_dot_max = 0.85

w = []
lbw = []
ubw = []
w0 = []

# # references
# _, Y_ref, phi_ref = get_ref_trajectory(speed, t_step, Np * t_step)
# Formulate the NLP
for k in range(Np):
    if k < Nc:
        Uk = ca.SX.sym("U_" + str(k))
        w += [Uk]
        lbw += [-delta_max]
        ubw += [delta_max]
        w0 += [0]
    J += (ca.vertcat(st[2], st[5]) - P[6:]).T @ Q @ (ca.vertcat(st[2], st[5]) - P[6:]) + Uk.T @ R @ Uk
    k1 = f(st, Uk)
    k2 = f(st + t_step / 2 * k1, Uk)
    k3 = f(st + t_step / 2 * k2, Uk)
    k4 = f(st + t_step / 2 * k3, Uk)
    st = st + (t_step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# Create an NLP solver
prob = {"f": J, "x": ca.vertcat(*w), "g": ca.vertcat(*g)}
solver = ca.nlpsol("solver", "ipopt", prob)
ca.lookupvector
# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=[-ca.inf], ubg=[ca.inf])
w_opt = sol["x"]

# Plotting
u_opt = w_opt
print(u_opt)
x_opt = [[0, 0, 0, 0, 0, 0]]
for k in range(Np):
    Fk = F(x0=x_opt[-1], p=u_opt[k])
    x_opt += [Fk["xf"].full()]
    print(x_opt)

# tgrid = [t_step * k for k in range(Np + 1)]
# plt.figure()
# plt.clf()
# plt.plot()
# plt.plot()
# plt.xlabel("t")
# plt.ylabel()
# plt.grid()
# plt.show()
