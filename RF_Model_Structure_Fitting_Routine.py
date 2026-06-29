"""
RF Model Structure Fitting Routine
Configuration 2 Supplementary Material

Dependencies
------------
This script requires the following Python packages:

- numpy
- matplotlib
- scipy
    - scipy.optimize (curve_fit)
    - scipy.special (Bessel functions)
    - scipy.interpolate (PchipInterpolator)
    - scipy.constants

- ipywidgets
- IPython.display

Tested with Python 3.x
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import jn
from scipy.interpolate import PchipInterpolator
import scipy.constants as const
import ipywidgets as widgets
from IPython.display import clear_output

c = const.c

f = np.array([
    14.232,
    12.694,
    10.186,
    9.673,
    6.585,
    5.0987,
    4.893,
    3.712,
    3.350
]) * 1e9

n = np.array([53, 55, 59, 60, 68, 74, 75, 82, 85])
idx_exp = [1, 2, 4, 5, 7]

lam = c / f
cell_width = 0.0175
cell_length = 0.075


def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


def rmse_np(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def wrap_phase(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


def eta_exp(n):
    return (
        -1.6225e-05 * n**4
        + 4.4086e-03 * n**3
        - 4.4523e-01 * n**2
        + 1.9832e+01 * n
        - 3.2801e+02
    )


def interactive_bessel_fit(s, q, Phi):
    clear_output(wait=True)

    n_sel = n[idx_exp]
    eta_sel = eta_exp(n_sel)

    R1 = cell_width / 2
    R2 = cell_length

    K1R = (2 * np.pi / (s * lam[idx_exp])) * R1
    K2R = (2 * np.pi * q / lam[idx_exp]) * R2

    K1R_interp = PchipInterpolator(n_sel, K1R)
    K2R_interp = PchipInterpolator(n_sel, K2R)

    def bessel_model_mix(n_in, C0, theta0, C1, theta1):
        k1r = K1R_interp(n_in)
        k2r = K2R_interp(n_in)

        theta0 = wrap_phase(theta0)
        theta1 = wrap_phase(theta1)

        J0 = jn(0, k1r)
        J1 = jn(1, k1r)

        E_m0 = C0 * J0 * np.cos(k2r + theta0)
        E_m1 = C1 * J1 * np.cos(k2r + theta1) * np.sin(np.deg2rad(Phi))

        return E_m0 + E_m1

    p0 = [1.0, 0.0, 1.0, 0.0]

    bounds = (
        [0.0, -np.pi, 0.0, -np.pi],
        [np.inf, np.pi, np.inf, np.pi]
    )

    popt, pcov = curve_fit(
        bessel_model_mix,
        n_sel,
        eta_sel,
        p0=p0,
        bounds=bounds
    )

    J_fit = bessel_model_mix(n_sel, *popt)

    r2 = r2_score_np(eta_sel, J_fit)
    rmse = rmse_np(eta_sel, J_fit)

    n_smooth = np.linspace(min(n_sel), max(n_sel), 500)

    eta_smooth = eta_exp(n_smooth)
    J_smooth = bessel_model_mix(n_smooth, *popt)

    C0_raw, theta0_raw, C1_raw, theta1_raw = popt

    C0 = C0_raw
    C1 = C1_raw
    theta0 = wrap_phase(theta0_raw)
    theta1 = wrap_phase(theta1_raw)

    phase_sum = np.abs(theta0) + np.abs(theta1)

    uncertainties = np.sqrt(np.diag(pcov))
    u_C0, u_theta0, u_C1, u_theta1 = uncertainties

    u_phase_sum = np.sqrt(u_theta0**2 + u_theta1**2)

    print(f"C0 = {C0:.4f} ± {u_C0:.4f}")
    print(f"theta0 = {theta0:.4f} ± {u_theta0:.4f}")
    print(f"C1 = {C1:.4f} ± {u_C1:.4f}")
    print(f"theta1 = {theta1:.4f} ± {u_theta1:.4f}")
    print(f"|theta0| + |theta1| = {phase_sum:.4f} ± {u_phase_sum:.4f}")
    print(f"R² = {r2:.4f}")
    print(f"RMSE = {rmse:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(n_smooth, eta_smooth, color='blue', label='Quartic Fit')
    plt.plot(n_smooth, J_smooth, '--', color='orange',
             label=f'Two-term Modal Expansion (s={s:.2f}, q={q:.2f})')
    plt.scatter(n_sel, eta_sel, color='blue')
    plt.scatter(n_sel, J_fit, color='orange')
    plt.xlabel('Principal Quantum Number (n)')
    plt.ylabel(r'$\eta$')
    plt.title('Two-Term Modal Expansion vs Quartic Fit')
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.show()

    difference = J_smooth - eta_smooth

    plt.figure(figsize=(10, 6))
    plt.plot(n_smooth, difference, color='darkred', label='Modal - Quartic')
    plt.axhline(0, color='gray', linestyle='--')
    plt.fill_between(n_smooth, difference, 0, where=(difference > 0),
                     color='green', alpha=0.25, label='Enhancement')
    plt.fill_between(n_smooth, difference, 0, where=(difference < 0),
                     color='blue', alpha=0.25, label='Decrement')
    plt.xlabel('Principal Quantum Number (n)')
    plt.ylabel('Difference')
    plt.title('Difference Between Modal Model and Quartic Fit')
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.show()


slider_s = widgets.FloatSlider(value=0.39, min=0.10, max=2.00, step=0.01, description='s')
slider_q = widgets.FloatSlider(value=0.20, min=0.01, max=1.00, step=0.01, description='q')
slider_Phi = widgets.FloatSlider(value=90.0, min=0.0, max=360.0, step=1.0, description='Phi')

widgets.interact(interactive_bessel_fit, s=slider_s, q=slider_q, Phi=slider_Phi)
