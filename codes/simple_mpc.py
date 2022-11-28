import casadi as ca
import matplotlib.pyplot as plt

# Simulation parameters
t_sim = 100  # Total simulation time in seconds
t_step = 0.1  # Time step in seconds
control_horizon = 3  # Control horizon : No of time steps optimal control is estimated
prediction_horizon = 10  # Prediction window
N = int(t_sim / t_step)

# Vehicle parameters
a = 0.3  # Distance between front wheel and COM in meters
b = 0.7  # Distance between rear wheel and COM in meters
m = 1  # Mass of car in tons
g = 9.81  # Gravity
I = 2  # Moment of inertia
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

x = ca.vertcat(x1, x2, x3, x4, x5, x6)  # Input vector
u = ca.SX.sym("delta")  # Front steering angle

Fxf = ca.SX.sym("Fxf")
Fxr = ca.SX.sym("Fxr")
Fyf = ca.SX.sym("Fyf")
Fyr = ca.SX.sym("Fyr")

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

# Formulate the NLP
for k in range(N):
    Uk = ca.SX.sym("U_" + str(k))
    w += [Uk]
    lbw += [-ca.inf]
    ubw += [ca.inf]
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
for k in range(N):
    Fk = F(x0=x_opt[-1], p=u_opt[k])
    x_opt += [Fk["xf"].full()]

tgrid = [t_step * k for k in range(N + 1)]
plt.figure()
plt.clf()
plt.plot()
plt.plot()
plt.xlabel("t")
plt.ylabel()
plt.grid()
plt.show()
