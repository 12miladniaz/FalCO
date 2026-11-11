"""BLDC motor design objective, ported line by line from the authors' MATLAB code (PSO-BLDC/Untitled.m).

Decision vector (9 variables; the number of pole pairs P is fixed to 5, since xmin1 = xmax1 = 5 in the
MATLAB code): [Beta, L_m, L_y, L_w, L_g, R_r, L_s, A_c, J_cu].
"""
import math

import numpy as np

P_FIXED = 5
LB = [0.69, 0.012, 0.00582, 0.0033, 0.0009, 0.059, 0.073, 1.9, 5780000.0]
UB = [0.72, 0.014, 0.061, 0.00355, 0.0011, 0.05971, 0.077, 2.0, 5820000.0]
NAMES = ["β", "l_m", "l_y", "l_w", "l_g", "r_r", "l_s", "A_c", "J_cu"]

k_f = 0.7; k_c = 2 / 3; b_r = 1; ro = 1.8e-8
k_h_p = 0.018; k_e_p = 0.00008; n = 1.92; w_p = 0.02; w_v = 2000 / 3; w_c = 0.0125
gamma = 1; ro_m = 7400; ro_w = 8900; ro_y = 7700; c_m_1 = 20
c_m_2 = 1; c_y = 3; c_1 = 0.045; c_2 = 5.42; w_r = 157
n_b = 2; mio_f = 0.0015; f_b = 600; d_i = 0.015; ro_air = 1.093; mio_air = 19.6e-6
k_r = 1


def parts(x):
    Beta, L_m, L_y, L_w, L_g, R_r, L_s, A_c, J_cu = [float(v) for v in x]
    P = P_FIXED
    V_m = Beta * math.pi * L_s * ((2 * R_r * L_m) - (L_m ** 2))
    V_w = math.pi * L_s * ((2 * (R_r + L_g) * L_w) + (L_w ** 2))
    V_t = math.pi * L_s * ((R_r + L_g + L_w + L_y) ** 2)
    Cost_Cte_1 = (c_1 / A_c) + c_2
    cost = c_m_1 * ro_m * V_m + c_m_2 * P + Cost_Cte_1 * k_f * ro_w * V_w + c_y * ro_y * V_t
    A_w = math.pi * L_w * ((2 * R_r) + (2 * L_g) + L_w)
    K_et = 1 + (math.pi * gamma * (R_r + L_g + L_w)) / (P * L_s)
    P_cu = ro * k_f * k_c * K_et * A_w * L_s * (J_cu ** 2)
    V_sy = math.pi * L_s * (2 * (R_r + L_g + L_w) * L_y + L_y ** 2)
    K_l_0 = Beta * P * (L_g + L_w)
    K_l = 1 - (1 / ((R_r / K_l_0) ** 2 + 1))
    LN = math.log((R_r + L_g + L_w) / (R_r - L_m))
    B_sy = (math.pi * K_l * Beta * b_r * L_m) / (2 * P * L_y * LN)
    f = (P * w_r) / (2 * math.pi)
    P_h = k_h_p * ro_y * V_sy * (B_sy ** n) * f
    P_e = (k_e_p * ro_y * V_sy) * (B_sy ** 2) * (f ** 2)
    P_b = (n_b * mio_f * f_b * d_i * w_r) / 2
    Re = (ro_air * w_r * R_r * L_g) / mio_air
    # MATLAB code: `if Re<10000 & Re>500`, which sends Re == 500 exactly to the turbulent formula and
    # creates an artificial minimum on the boundary. The laminar range is taken as 500 <= Re < 10000 here.
    if 500 <= Re < 10000:
        C_f = 0.515 * (((L_g / R_r) ** 0.3) / (Re ** 0.5))
    else:
        C_f = 0.0325 * (((L_g / R_r) ** 0.3) / (Re ** 0.2))
    if Re < 500:
        C_f = 1e10
    P_w = math.pi * k_r * C_f * ro_air * (w_r ** 3) * (R_r ** 4) * L_s
    loss = P_cu + P_h + P_e + P_b + P_w
    return {"cost": cost, "loss": loss, "volume": V_t,
            "F": w_p * loss + w_c * cost + w_v * V_t}


def BLDC(x):
    return parts(x)["F"]


if __name__ == "__main__":
    fc_paper = [0.69, 0.012, 0.00582, 0.0033, 0.0009679529543367227, 0.059, 0.073, 2.0, 5780000.0]
    print(parts(fc_paper))
