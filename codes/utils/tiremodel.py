import matplotlib.pyplot as plt
import numpy as np

# Matrix with coefficients
# F_y -> 1st row; M_z -> 2nd row; F_x -> 3rd row


A = np.array(
    [
        [-22.1, 1011, 1078, 1.82, 0.208, 0, -0.354, 0.707, 0.028, 0, 14.8, 0.022, 0],
        [-2.72, -2.28, -1.86, -2.73, 0.11, -0.07, 0.643, -4.04, 0.015, -0.066, 0.945, 0.03, 0.07],
        [-21.3, 1144, 49.6, 226, 0.069, -0.006, 0.056, 0.486, 0, 0, 0, 0, 0],
    ]
)


def get_lateral_force(F_z, alpha, gamma=4, plot=False):
    alpha = alpha / np.pi * 180
    # Lateral Force
    C = 1.3
    D = A[0, 0] * F_z**2 + A[0, 1] * F_z
    BCD = A[0, 2] * np.sin(A[0, 3] * np.arctan(A[0, 4] * F_z))
    B = BCD / (C * D)
    E = A[0, 5] * F_z**2 + A[0, 6] * F_z + A[0, 7]
    S_h = A[0, 8] * gamma
    S_v = (A[0, 9] * F_z**2 + A[0, 10] * F_z) * gamma
    phi = (1 - E) * (alpha + S_h) + E / B * np.arctan(B * (alpha + S_h))
    F_y = D * np.sin(C * np.arctan(B * phi)) + S_v

    if plot:
        plt.figure(1)
        plt.plot(alpha, F_y)
        plt.xlabel("Slip angle (deg)")
        plt.ylabel("Lateral force (N)")
        plt.grid()
        plt.show()

    return F_y


def get_aligning_moment(F_z, alpha, gamma=4, plot=False):
    alpha = alpha / np.pi * 180
    # Aligning Moment
    C = 2.4
    D = A[1, 0] * F_z**2 + A[1, 1] * F_z
    BCD = (A[1, 2] * F_z**2 + A[1, 3] * F_z) / np.exp(A[1, 4] * F_z)
    B = BCD / (C * D)
    E = A[1, 5] * F_z**2 + A[1, 6] * F_z + A[1, 7]
    S_h = A[1, 8] * gamma
    S_v = (A[1, 9] * F_z**2 + A[1, 10] * F_z) * gamma
    phi = (1 - E) * (alpha + S_h) + E / B * np.arctan(B * (alpha + S_h))
    M_z = D * np.sin(C * np.arctan(B * phi)) + S_v
    if plot:
        plt.figure(2)
        plt.plot(alpha, M_z)
        plt.xlabel("Slip angle (deg)")
        plt.ylabel("Aligning Moment (Nm)")
        plt.grid()
        plt.show()
    return M_z


def get_longitudinal_force(F_z, slip_ratio, plot=False):
    # Longitudinal Force
    C = 1.65
    D = A[2, 0] * F_z**2 + A[2, 1] * F_z
    BCD = (A[2, 2] * F_z**2 + A[2, 3] * F_z) / np.exp(A[2, 4] * F_z)
    B = BCD / (C * D)
    E = A[2, 5] * F_z**2 + A[2, 6] * F_z + A[2, 7]

    phi = (1 - E) * slip_ratio + E / B * np.arctan(B * slip_ratio)
    F_x = D * np.sin(C * np.arctan(B * phi))
    if plot:
        plt.figure(3)
        plt.plot(slip_ratio, F_x)
        plt.xlabel("Slip angle (deg)")
        plt.ylabel("Aligning Moment (Nm)")
        plt.grid()
        plt.show()
    return F_x
