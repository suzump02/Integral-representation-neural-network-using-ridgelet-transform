# file: src/calculator.py
import numpy as np
import mpmath as mp
from tqdm import tqdm

def K_eta_psi(k, psi_hat_func, eta_hat_func):
    """K_{eta,psi} の積分を計算"""
    f_int = lambda z: (mp.conj(psi_hat_func(z, k)) * eta_hat_func(z)) / abs(z)
    val = mp.quad(f_int, [-mp.inf, 0, mp.inf])
    return float(val.real)

def compute_T_ab_grid(k_val, f_func, psi_func, a_points=61, b_points=61):
    """T(a,b)のグリッド計算を実行"""
    a_vals = np.linspace(-30, 30, a_points)
    b_vals = np.linspace(-30, 30, b_points)
    T_values = np.zeros((len(a_vals), len(b_vals)), dtype=np.float64)

    def calculate_T(a, b, k):
        if a == 0: return 0.0
        integrand = lambda x: abs(a) * f_func(x) * mp.conj(psi_func(a * x - b, k))
        val = mp.quad(integrand, [-1, 1])
        return val.real

    for i, a in enumerate(tqdm(a_vals, desc=f"Calculating T(a,b) for k={k_val}")):
        for j, b in enumerate(b_vals):
            try:
                T_values[i, j] = calculate_T(a, b, k_val)
            except ZeroDivisionError:
                print(f"Warning: ZeroDivisionError at k={k_val}, a={a}, b={b}. Setting T=0.")
                T_values[i, j] = 0.0
    return a_vals, b_vals, T_values
