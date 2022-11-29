import casadi as ca
import matplotlib.pyplot as plt
from utils.tiremodel import get_lateral_force, get_longitudinal_force

# Simulation parameters
t_step = 0.05  # Time step in seconds
Nc = 3  # Control horizon : No of time steps optimal control is estimated
Np = 7  # Prediction window
Q = ca.DM([[200, 0], [0, 75]])  # State weighing matrix
R = 150  # Input weighing matrix

# Vehicle parameters
a = 0.3  # Distance between front wheel and COM in meters
b = 0.7  # Distance between rear wheel and COM in meters
m = 2.05  # Mass of car in tons
g = 9.81  # Gravity
I = 3.344  # Moment of inertia
Fz = m * g  # Total normal force from ground
Fzf = b * Fz / (2 * (a + b))  # Normal force on front wheels
Fzr = a * Fz / (2 * (a + b))  # Normal force on rear wheels

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

Flf = get_longitudinal_force(Fzf, slip_ratio_f)
Fcf = get_lateral_force(Fzf, alpha_f)
Flr = get_longitudinal_force(Fzr, slip_ratio_r)
Fcr = get_lateral_force(Fzr, alpha_r)

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

# Objective function
L = 1

# Distrete time dynamics
dae = {"x": x, "p": u, "ode": xdot, "quad": L}
opts = {"tf": t_step}
F = ca.integrator("F", "cvodes", dae, opts)

# Defining variables for NLP
w = []
w0 = []
lbw = []
ubw = []
J = 0
g = []
lbg = []
ubg = []

# Limits
delta_max = 10
delta_dot_max = 0.85

# Formulate the NLP
for k in range(Np):
    if k < Nc:
        Uk = ca.SX.sym("U_" + str(k))
        w += [Uk]
        lbw += [-delta_max]
        ubw += [delta_max]
        w0 += [0]

    Fk = F(x0=Xk, p=Uk)
    Xk = Fk["xf"]
    J += Fk["qf"]

    g += [Xk]
    lbg += []
    ubg += []

# Create an NLP solver
prob = {"f": J, "x": ca.vertcat(*w), "g": ca.vertcat(*g)}
solver = ca.nlpsol("solver", "ipopt", prob)

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol["x"]

# Plotting
u_opt = w_opt
x_opt = [[]]
for k in range(Np):
    Fk = F(x0=x_opt[-1], p=u_opt[k])
    x_opt += [Fk["xf"].full()]

tgrid = [t_step * k for k in range(Np + 1)]
plt.figure()
plt.clf()
plt.plot()
plt.plot()
plt.xlabel("t")
plt.ylabel()
plt.grid()
plt.show()
