# file: reconstruct_signal.py
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import csv

# compute_T_ab.pyからグリッド計算用の関数をインポート
from compute_T_ab import compute_T_ab_grid

def reconstruct_signal_with_K():
    """
    T(a, b)のデータを準備し、K_eta_psiでスケーリングして信号を再構成し、
    プロットします。
    """
    mp.mp.dps = 25

    # === ステップ1: T(a,b)のデータを準備 ===
    
    k_val = 2
    data_filename = os.path.join('data', 'distribution_T', f'T_ab_data_k{k_val}.npz')
    
    if os.path.exists(data_filename):
        print(f"Loading T(a,b) data from '{data_filename}'...")
        data = np.load(data_filename)
        a_vals, b_vals, T_values = data['a_vals'], data['b_vals'], data['T_values']
    else:
        print(f"'{data_filename}' not found. Calculating from scratch...")
        a_vals, b_vals, T_values = compute_T_ab_grid(k_val=k_val)
        output_dir = os.path.dirname(data_filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.savez(data_filename, a_vals=a_vals, b_vals=b_vals, T_values=T_values)
        print(f"T(a,b) data saved to '{data_filename}' for future use.")

    # === ステップ2: K_eta_psi の値を読み込み ===

    K_eta_psi = None
    kernel_filename = os.path.join('data', 'kernel_product_results.csv')
    
    try:
        with open(kernel_filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if int(row['k']) == k_val:
                    K_eta_psi = float(row['K_eta_psi'])
                    break
        if K_eta_psi is not None:
            print(f"Successfully loaded K_eta_psi = {K_eta_psi} for k={k_val} from '{kernel_filename}'")
        else:
            print(f"Warning: K_eta_psi for k={k_val} not found in '{kernel_filename}'.")
            return
    except FileNotFoundError:
        print(f"Error: Kernel file '{kernel_filename}' not found.")
        return
    except (ValueError, KeyError) as e:
        print(f"Error: Could not read data from '{kernel_filename}'. Check file format. Details: {e}")
        return

    # === ステップ3: 再構成公式の左辺 g(x) の計算 ===
    
    print("Reconstructing the signal g(x)...")
    
    def eta(z):
        return mp.tanh(z)

    x_reconstructed = np.linspace(-1, 1, 200)
    g_values = np.zeros_like(x_reconstructed, dtype=np.float64)
    
    delta_a = a_vals[1] - a_vals[0]
    delta_b = b_vals[1] - b_vals[0]

    for i_x, x in enumerate(tqdm(x_reconstructed, desc="Reconstructing g(x)")):
        reconstructed_sum = 0.0
        for i_a, a in enumerate(a_vals):
            if abs(a) < 1e-9:
                continue
            for j_b, b in enumerate(b_vals):
                T_ab = T_values[i_a, j_b]
                eta_val = eta(a * x - b)
                reconstructed_sum += T_ab * eta_val * (delta_a * delta_b) / abs(a)
        g_values[i_x] = reconstructed_sum

    # === ステップ4: プロットによる結果の比較 ===
    
    print("Plotting the original and reconstructed signals...")
    plt.figure(figsize=(12, 7))
    
    x_original = np.linspace(-1, 1, 400)
    f_original = np.sin(2 * np.pi * x_original)
    plt.plot(x_original, f_original, 'k--', label='Original Signal: f(x) = sin(2πx)', zorder=1)
    
    # 読み込んだ K_eta_psi でスケーリング
    if abs(K_eta_psi) > 1e-9:
        g_values_scaled = g_values / K_eta_psi
    else:
        print("Warning: K_eta_psi is close to zero. Cannot scale properly.")
        g_values_scaled = g_values

    plt.plot(x_reconstructed, g_values_scaled, 'r-', label=f'Reconstructed Signal (Scaled by K={K_eta_psi:.4f})', linewidth=2, zorder=2)
    
    plt.xlabel("x")
    plt.ylabel("Value")
    plt.title("Function Reconstruction using Ridgelet Transform")
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.show()

if __name__ == "__main__":
    reconstruct_signal_with_K()
