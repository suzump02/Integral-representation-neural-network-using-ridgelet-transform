# file: create_data/compute_kernel_product.py
import mpmath as mp
import csv
import os
from tqdm import tqdm

# tanh(z) のフーリエ変換
def eta_hat(z):
    return -1j * mp.pi / (2 * mp.sinh(mp.pi * z / 2))

# psi_hat(z) の定義
def psi_hat(z, k):
    N = 2 * k - 1 
    phase = (1j) ** N
    return phase * (z ** N) * mp.exp(-(z ** 2))

# K_{eta,psi} の積分計算
def K_eta_psi(k):
    f = lambda z: (mp.conj(psi_hat(z, k)) * eta_hat(z)) / abs(z)
    val = mp.quad(f, [-mp.inf, 0, mp.inf])
    return float(val.real)

if __name__ == "__main__":
    mp.mp.dps = 30
    
    ks_to_compute = range(1, 101)
    results = []
    
    print(f"Calculating K_eta_psi for k = {ks_to_compute.start} to {ks_to_compute.stop - 1}...")

    for k in tqdm(ks_to_compute, desc="Calculating K_eta_psi"):
        val = K_eta_psi(k)
        results.append({'k': k, 'K_eta_psi': val})

    # === 計算結果をCSVファイルに保存 ===
    
    # --- パス設定の変更箇所 ---
    # このスクリプト自身の場所を基準に、プロジェクトのルートディレクトリを取得
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # create_dataの1つ上の階層
    
    # 保存先のディレクトリをプロジェクトルートからのパスとして構築
    output_dir = os.path.join(project_root, 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    # ファイル名とディレクトリを結合して完全なパスを作成
    output_filename = os.path.join(output_dir, 'kernel_product_results.csv')
    
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['k', 'K_eta_psi']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            
        print(f"\nCalculation finished.")
        print(f"Results have been saved to '{output_filename}'")

    except IOError as e:
        print(f"\nAn error occurred while writing to the file: {e}")
