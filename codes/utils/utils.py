import casadi as ca
import numpy as np


def DM2Arr(dm):
    return np.array(dm.full())


def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(u[:, 1:], ca.reshape(u[:, -1], -1, 1))

    return t0, next_state, u0
