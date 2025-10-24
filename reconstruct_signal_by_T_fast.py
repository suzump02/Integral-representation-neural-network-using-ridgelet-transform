# ======================================================
# reconstruct_and_visualize_fast.py
# Ridgelet再構成（NumPyベクトル化版）
# ======================================================

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import csv

def reconstruct_and_plot_for_k_range():
    # --- 設定 ---
    ks_to_process = range(1, 6)
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, 'data', 'distribution_T')
    kernel_filename = os.path.join(project_root, 'data', 'kernel_product_results.csv')
    output_dir = os.path.join(project_root, 'result-figures', 'reconstruct')
    os.makedirs(output_dir, exist_ok=True)

    # --- カーネル定数を読み込み ---
    K_dict = {}
    try:
        with open(kernel_filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                k_val = int(row['k'])
                K_dict[k_val] = float(row['K_eta_psi'])
        print(f"[INFO] Loaded K_eta_psi for {len(K_dict)} k-values from '{kernel_filename}'")
    except FileNotFoundError:
        print(f"[ERROR] '{kernel_filename}' not found.")
        return

    def eta_np(z):
        return np.tanh(z)

    # --- 再構成ループ ---
    for k_val in ks_to_process:
        print(f"\n--- Processing for k={k_val} ---")

        data_filename = os.path.join(data_dir, f'T_ab_data_k{k_val}.npz')
        if not os.path.exists(data_filename):
            print(f"[WARN] Missing: {data_filename}")
            continue

        data = np.load(data_filename)
        a_vals, b_vals, T_values = data['a_vals'], data['b_vals'], data['T_values']
        if k_val not in K_dict:
            print(f"[WARN] Missing K_eta_psi for k={k_val}.")
            continue
        K_eta_psi = K_dict[k_val]

        print(f"[INFO] Reconstructing for k={k_val} with K_eta_psi={K_eta_psi:.4e}")

        # --- 再構成 ---
        x_reconstructed = np.linspace(-1, 1, 400)
        delta_a = a_vals[1] - a_vals[0]
        delta_b = b_vals[1] - b_vals[0]
        g_k = np.zeros_like(x_reconstructed)

        # b方向のベクトル化により高速化
        for i, a in enumerate(tqdm(a_vals, desc=f"Reconstructing k={k_val}")):
            if abs(a) < 1e-9:
                continue
            # ψ積分済みT(a,b) → b依存関数
            T_ab_row = T_values[i, :]  # shape (B,)
            for xi, x in enumerate(x_reconstructed):
                z = a * x - b_vals        # shape (B,)
                eta_vals = eta_np(z)
                g_k[xi] += np.trapz(T_ab_row * eta_vals, b_vals)

        g_k *= delta_a / K_eta_psi

        # --- 可視化 ---
        f_original = np.sin(2 * np.pi * x_reconstructed)
        mse = np.mean((f_original - g_k) ** 2)

        plt.figure(figsize=(10, 6))
        plt.plot(x_reconstructed, f_original, 'k--', label='Original: sin(2πx)')
        plt.plot(x_reconstructed, g_k, 'r-', lw=2, label=f'Reconstructed (k={k_val})')
        plt.title(f"Signal Reconstruction (k={k_val})\nMSE = {mse:.3e}")
        plt.xlabel("x")
        plt.ylabel("Value")
        plt.grid(True, linestyle=":")
        plt.legend()

        plot_filename = os.path.join(output_dir, f'reconstruction_k_{k_val}.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f"[DONE] Saved: {plot_filename}")

    print("\n[INFO] All processing finished.")

if __name__ == "__main__":
    reconstruct_and_plot_for_k_range()