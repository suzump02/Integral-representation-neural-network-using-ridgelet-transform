# ======================================================
# compute_T_ab_batch_fast.py
# Ridgelet型 T(a,b) の高速計算版（mpmath除去、NumPyベクトル化）
# ======================================================

import numpy as np
from tqdm import tqdm
import os
from numpy.polynomial.hermite import hermval

# ------------------------------------------------------
# 基本設定
# ------------------------------------------------------

def f_np(x):
    """元信号 f(x) = sin(2πx)"""
    return np.sin(2 * np.pi * x)

def psi_np(z, k):
    """
    リッジレット関数 ψ(z) = const * H_{2k-1}(z/2) * exp(-z^2 / 4)
    """
    const = - (0.5 ** (2 * k - 1)) / (2.0 * np.sqrt(np.pi))
    x_arg = z / 2.0
    deg = 2 * k - 1
    coeffs = np.zeros(deg + 1)
    coeffs[-1] = 1.0
    H = hermval(x_arg, coeffs)
    return const * H * np.exp(-z**2 / 4.0)

# ------------------------------------------------------
# 高速版 T(a,b) 計算（x離散積分）
# ------------------------------------------------------

def compute_T_ab_grid(k_val=1, a_points=301, b_points=301, Nx=201, x_range=(-1,1), only_grid=False):
    """
    Ridgelet変換 T(a,b) を高速に格子計算する。
    ・x方向は[-1,1]を台形則で積分近似
    ・Hermiteをnumpyで評価
    """
    a_vals = np.linspace(-30, 30, a_points)
    b_vals = np.linspace(-30, 30, b_points)
    if only_grid:
        return a_vals, b_vals, None

    x_grid = np.linspace(x_range[0], x_range[1], Nx)
    dx = x_grid[1] - x_grid[0]
    fx = f_np(x_grid)

    T_values = np.zeros((a_points, b_points), dtype=np.float64)

    # 外側ループ: a のみ
    for i, a in enumerate(tqdm(a_vals, desc=f"Computing T(a,b) for k={k_val}")):
        if abs(a) < 1e-12:
            continue

        # a固定で x-grid に依存する ψ(a*x - b)
        # b を全てまとめてベクトル化して同時計算
        ax = a * x_grid[:, None] - b_vals[None, :]   # shape (Nx, b_points)
        psi_vals = psi_np(ax, k_val)                 # shape (Nx, b_points)
        integrand = fx[:, None] * psi_vals           # f(x) * ψ(a x - b)
        T_values[i, :] = np.trapz(integrand, x_grid, axis=0)

    return a_vals, b_vals, T_values

# ------------------------------------------------------
# メイン処理
# ------------------------------------------------------

if __name__ == '__main__':
    # 高速設定
    k_start = 2
    k_end = 3
    ks_to_compute = range(k_start, k_end + 1)
    a_points = 301   # ← まずはこれでOK
    b_points = 301
    Nx = 201         # ← 積分点

    print(f"[INFO] Compute T(a,b) for k in {list(ks_to_compute)} (fast mode).")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, 'data', 'distribution_T')
    os.makedirs(output_dir, exist_ok=True)

    for k in ks_to_compute:
        print(f"\n--- Processing for k={k} ---")

        output_filename = os.path.join(output_dir, f'T_ab_data_k{k}.npz')
        if os.path.exists(output_filename):
            print(f"[SKIP] File exists: {output_filename}")
            continue

        a_vals, b_vals, T_values = compute_T_ab_grid(
            k_val=k, a_points=a_points, b_points=b_points, Nx=Nx
        )

        np.savez(output_filename, a_vals=a_vals, b_vals=b_vals, T_values=T_values)
        print(f"[DONE] Saved: {output_filename}")

    print("\nAll computations completed successfully.")