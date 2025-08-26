# file: analyze_reconstruction_error.py
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import csv
import pandas as pd

# create_dataフォルダのスクリプトから関数をインポート
from create_data.compute_T_ab import compute_T_ab_grid

project_root = os.path.dirname(os.path.abspath(__file__))

def analyze_error_vs_k():
    """
    k=1から10まで、各リッジレット関数による再構成誤差（MSE）を計算し、
    グラフとしてプロットします。
    """
    mp.mp.dps = 25

    # === ステップ1: K_eta_psi の全データを読み込み ===

    kernel_filename = os.path.join('data', 'kernel_product_results.csv')
    try:
        kernel_df = pd.read_csv(kernel_filename, index_col='k')
        print(f"Successfully loaded kernel data from '{kernel_filename}'")
    except FileNotFoundError:
        print(f"Error: Kernel file '{kernel_filename}' not found. Please run compute_kernel_product.py first.")
        return

    # === ステップ2: 各kについて誤差を計算 ===
    
    k_range = range(1, 11)
    error_results = []

    for k_val in k_range:
        print(f"\n--- Processing for k={k_val} ---")
        
        # T(a,b)のデータを準備
        data_filename = os.path.join('data', 'distribution_T', f'T_ab_data_k{k_val}.npz')
        if not os.path.exists(data_filename):
            print(f"Warning: T(a,b) data for k={k_val} not found. Skipping.")
            continue
            
        data = np.load(data_filename)
        a_vals, b_vals, T_values = data['a_vals'], data['b_vals'], data['T_values']
        
        # 対応するK_eta_psiの値を取得
        try:
            K_eta_psi = kernel_df.loc[k_val, 'K_eta_psi']
        except KeyError:
            print(f"Warning: K_eta_psi for k={k_val} not found in CSV. Skipping.")
            continue

        # 信号を再構成
        print(f"Reconstructing signal for k={k_val}...")
        def eta(z): return mp.tanh(z)
        x_reconstructed = np.linspace(-1, 1, 400) # 精度のため少し細かくする
        g_values = np.zeros_like(x_reconstructed, dtype=np.float64)
        delta_a = a_vals[1] - a_vals[0]
        delta_b = b_vals[1] - b_vals[0]

        for i_x, x in enumerate(tqdm(x_reconstructed, desc=f"Reconstructing (k={k_val})")):
            reconstructed_sum = 0.0
            for i_a, a in enumerate(a_vals):
                if abs(a) < 1e-9: continue
                for j_b, b in enumerate(b_vals):
                    T_ab = T_values[i_a, j_b]
                    eta_val = eta(a * x - b)
                    reconstructed_sum += T_ab * eta_val * (delta_a * delta_b) / abs(a)
            g_values[i_x] = reconstructed_sum

        # スケーリングと誤差計算
        if abs(K_eta_psi) > 1e-9:
            g_values_scaled = g_values / K_eta_psi
            f_original = np.sin(2 * np.pi * x_reconstructed)
            mse = np.mean((f_original - g_values_scaled)**2)
            error_results.append({'k': k_val, 'mse': mse})
            print(f"MSE for k={k_val}: {mse:.4e}")
        else:
            print(f"Warning: K_eta_psi for k={k_val} is near zero. Skipping error calculation.")

    # === ステップ3: 誤差をグラフ化 ===
    
    if not error_results:
        print("\nNo results to plot. Exiting.")
        return
        
    print("\nPlotting reconstruction error vs. k...")
    error_df = pd.DataFrame(error_results)

    plt.figure(figsize=(10, 6))
    plt.plot(error_df['k'], error_df['mse'], marker='o', linestyle='-', color='b')
    
    # 縦軸を対数スケールに
    # plt.yscale('log')
    
    plt.xlabel("k (Order of Ridgelet function)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Reconstruction Error vs. Ridgelet Function Order")
    plt.xticks(k_range) # 横軸の目盛りを整数に
    plt.grid(True, which="both", linestyle="--")
    
    # 保存先のディレクトリを指定
    output_dir = os.path.join(project_root, 'result-figures')
    # ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # ファイル名とディレクトリを結合して完全なパスを作成
    output_filename = os.path.join(output_dir, 'reconstruction_error_vs_k.png')
    
    plt.savefig(output_filename)
    print(f"Graph saved as '{output_filename}'")
    
    plt.show()

if __name__ == "__main__":
    analyze_error_vs_k()
