# file: reconstruct_signal_multi_k.py
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import csv

from create_data.compute_T_ab import compute_T_ab_grid

def reconstruct_signal_multi_k():
    """
    複数の k に対して T(a,b) と K_eta_psi を読み込み、
    それぞれの寄与を足し合わせて信号を再構成する。
    """
    mp.mp.dps = 25
    k_list = [1]  # ← ここで足し合わせるkのリストを指定

    # === ステップ1: カーネル定数 K_eta_psi をまとめて読み込み ===
    kernel_filename = os.path.join('data', 'kernel_product_results.csv')
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

    # === ステップ2: 再構成用のx軸設定 ===
    x_reconstructed = np.linspace(-1, 1, 200)
    g_total = np.zeros_like(x_reconstructed, dtype=np.float64)

    # === ステップ3: 各kについて寄与を計算して足し合わせ ===
    for k_val in k_list:
        data_filename = os.path.join('data', 'distribution_T', f'T_ab_data_k{k_val}.npz')

        # T(a,b)の読み込みまたは計算
        if os.path.exists(data_filename):
            print(f"[INFO] Loading T(a,b) for k={k_val}...")
            data = np.load(data_filename)
            a_vals, b_vals, T_values = data['a_vals'], data['b_vals'], data['T_values']
        else:
            print(f"[WARN] '{data_filename}' not found. Computing new data...")
            a_vals, b_vals, T_values = compute_T_ab_grid(k_val=k_val)
            os.makedirs(os.path.dirname(data_filename), exist_ok=True)
            np.savez(data_filename, a_vals=a_vals, b_vals=b_vals, T_values=T_values)

        # K_eta_psi の取得
        if k_val not in K_dict:
            print(f"[WARN] K_eta_psi not found for k={k_val}, skipping...")
            continue
        K_eta_psi = K_dict[k_val]

        print(f"[INFO] Reconstructing for k={k_val} with K_eta_psi={K_eta_psi:.4f}")

        # 再構成の積分
        def eta(z):
            return mp.tanh(z)

        delta_a = a_vals[1] - a_vals[0]
        delta_b = b_vals[1] - b_vals[0]
        g_k = np.zeros_like(x_reconstructed, dtype=np.float64)

        for i_x, x in enumerate(tqdm(x_reconstructed, desc=f"Reconstructing k={k_val}")):
            s = 0.0
            for i_a, a in enumerate(a_vals):
                if abs(a) < 1e-9:
                    continue
                for j_b, b in enumerate(b_vals):
                    T_ab = T_values[i_a, j_b]
                    eta_val = eta(a * x - b)
                    s += T_ab * eta_val * (delta_a * delta_b) / abs(a)
            g_k[i_x] = s / K_eta_psi

        # 各kの寄与を合計
        g_total += g_k

    # === ステップ4: 結果の可視化 ===
    print("\n[INFO] Plotting reconstructed vs original signals...")
    plt.figure(figsize=(12, 7))

    x_original = np.linspace(-1, 1, 400)
    f_original = np.sin(2 * np.pi * x_original)

    plt.plot(x_original, f_original, 'k--', label='Original Signal: sin(2πx)', zorder=1)
    plt.plot(x_reconstructed, g_total, 'r-', linewidth=2, label=f"Reconstructed ∑k={k_list}")
    plt.xlabel("x")
    plt.ylabel("Value")
    plt.title(f"Reconstruction from Multiple k values: k={k_list}")
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.show()


if __name__ == "__main__":
    reconstruct_signal_multi_k()