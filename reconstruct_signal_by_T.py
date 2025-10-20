# file: reconstruct_and_visualize.py
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import csv

def reconstruct_and_plot_for_k_range():
    """
    指定されたkの範囲について、それぞれ独立して信号を再構成し、
    結果を個別のグラフとしてプロット・保存する。
    """
    mp.mp.dps = 100

    # --- 検証したいkの範囲を指定 ---
    ks_to_process = range(1, 6) 

    # --- ステップ1: カーネル定数 K_eta_psi をまとめて読み込み ---
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
        print(f"[ERROR] '{kernel_filename}' not found. Please run compute_kernel_product.py first.")
        return

    # ★★★ 修正箇所 1 ★★★
    # --- ステップ2: 保存用ディレクトリのパスを変更 ---
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_root, 'result-figures', 'reconstruct')
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Plots will be saved to '{output_dir}'")

    # --- ステップ3: 各kについて独立して再構成とプロットを実行 ---
    for k_val in ks_to_process:
        print(f"\n--- Processing for k={k_val} ---")
        
        data_filename = os.path.join('data', 'distribution_T', f'T_ab_data_k{k_val}.npz')

        if not os.path.exists(data_filename):
            print(f"[WARN] Data file '{data_filename}' not found. Skipping k={k_val}.")
            continue
            
        print(f"[INFO] Loading T(a,b) for k={k_val}...")
        data = np.load(data_filename)
        a_vals, b_vals, T_values = data['a_vals'], data['b_vals'], data['T_values']

        if k_val not in K_dict:
            print(f"[WARN] K_eta_psi not found for k={k_val}. Skipping.")
            continue
        K_eta_psi = K_dict[k_val]

        print(f"[INFO] Reconstructing for k={k_val} with K_eta_psi={K_eta_psi:.4e}")

        # 再構成の積分計算
        def eta(z): return mp.tanh(z)
        x_reconstructed = np.linspace(-1, 1, 400)
        g_k = np.zeros_like(x_reconstructed, dtype=np.float64)
        delta_a = a_vals[1] - a_vals[0]
        delta_b = b_vals[1] - b_vals[0]

        for i_x, x in enumerate(tqdm(x_reconstructed, desc=f"Reconstructing k={k_val}")):
            s = 0.0
            for i_a, a in enumerate(a_vals):
                if abs(a) < 1e-9: continue
                for j_b, b in enumerate(b_vals):
                    T_ab = T_values[i_a, j_b]
                    eta_val = eta(a * x - b)
                    s += T_ab * eta_val * (delta_a * delta_b) / abs(a)
            
            if abs(K_eta_psi) > 1e-12:
                g_k[i_x] = s / K_eta_psi
            else:
                g_k[i_x] = s

        # 結果の可視化
        plt.figure(figsize=(10, 6))
        
        f_original = np.sin(2 * np.pi * x_reconstructed)

        plt.plot(x_reconstructed, f_original, 'k--', label='Original Signal: sin(2πx)', zorder=2)
        plt.plot(x_reconstructed, g_k, 'r-', linewidth=2, label=f"Reconstructed (k={k_val})", zorder=3)
        
        mse = np.mean((f_original - g_k)**2)
        
        plt.xlabel("x")
        plt.ylabel("Value")
        plt.title(f"Signal Reconstruction for k = {k_val}\n(MSE = {mse:.4e})")
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.axhline(0, color='black', linewidth=0.5)
        
        # ★★★ 修正箇所 2 ★★★
        # 保存パスは上で定義したoutput_dirをそのまま使用
        plot_filename = os.path.join(output_dir, f'reconstruction_k_{k_val}.png')
        plt.savefig(plot_filename)
        print(f"[INFO] Plot for k={k_val} saved to '{plot_filename}'")
        plt.close()

    print("\n[INFO] All processing finished.")


if __name__ == "__main__":
    reconstruct_and_plot_for_k_range()