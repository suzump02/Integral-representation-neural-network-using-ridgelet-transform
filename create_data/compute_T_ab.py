# file: create_data/compute_T_ab_batch.py
import numpy as np
import mpmath as mp
from tqdm import tqdm
import os

# -----------------------------------------------------------------
# 再利用可能な関数群
# -----------------------------------------------------------------

def f(x):
    """
    元信号 f(x) = sin(2πx) を定義します。
    """
    return mp.sin(2 * mp.pi * x)

@mp.memoize
def psi(z, k):
    """
    指定された解析式に基づき、リッジレット関数 ψ(z) を定義します。
    """
    const = (-1) * (0.5**(2*k-1))/(2 *mp.sqrt(mp.pi))
    x_arg = z / 2
    hermite_val = mp.hermite(2*k-1, x_arg)
    exp_val = mp.exp(-z**2 / 4)
    return const * hermite_val * exp_val

def calculate_T(a, b, k):
    """
    与えられた (a, b) の点に対してリッジレット変換 T(a, b) を計算します。
    """
    if a == 0:
        return 0.0

    integrand = lambda x: f(x) * mp.conj(psi(a * x - b, k)) 

    val = mp.quad(integrand, [-1, 1])
    return val.real

def compute_T_ab_grid(k_val=1, a_points=601, b_points=601, only_grid=False):
    """
    T(a,b)のグリッド計算をまとめて実行し、結果のグリッドを返します。
    """
    # この関数内ではプログレスバーはtqdm(a_vals)のみに任せる
    a_vals = np.linspace(-30, 30, a_points)
    b_vals = np.linspace(-30, 30, b_points)

    if only_grid:
        return a_vals, b_vals, None

    T_values = np.zeros((len(a_vals), len(b_vals)), dtype=np.float64)

    for i, a in enumerate(tqdm(a_vals, desc=f"Calculating T(a,b) for k={k_val}")):
        for j, b in enumerate(b_vals):
            T_values[i, j] = calculate_T(a, b, k_val)
            
    return a_vals, b_vals, T_values

# -----------------------------------------------------------------
# このファイルを直接実行した場合の動作
# -----------------------------------------------------------------
if __name__ == '__main__':
    mp.mp.dps = 10

    # 計算したいkの範囲を設定
    k_start = 2
    k_end = 3
    ks_to_compute = range(k_start, k_end + 1)
    
    print(f"This script will pre-calculate T(a,b) for k from {k_start} to {k_end}.")
    
    # このスクリプト自身の場所を基準に、プロジェクトのルートディレクトリを取得
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # create_dataの1つ上の階層
    
    # 保存先のディレクトリをプロジェクトルートからのパスとして構築
    output_dir = os.path.join(project_root, 'data', 'distribution_T')
    os.makedirs(output_dir, exist_ok=True)
    
    # ループで各kの値を計算し、個別のファイルに保存
    for k in ks_to_compute:
        print(f"\n--- Processing for k={k} ---")
        
        # kの値を反映したファイル名を生成
        base_filename = f'T_ab_data_k{k}.npz'
        output_filename = os.path.join(output_dir, base_filename)
        
        # 既にファイルが存在する場合はスキップ
        if os.path.exists(output_filename):
            print(f"Data for k={k} already exists. Skipping.")
            continue
            
        # T(a,b)のグリッドを計算
        a_vals, b_vals, T_values = compute_T_ab_grid(k_val=k)
        
        # NumPyの .npz 形式で関連データをまとめて保存
        np.savez(output_filename, a_vals=a_vals, b_vals=b_vals, T_values=T_values)
        
        print(f"\nT(a,b) data for k={k} has been saved to '{output_filename}'")

    print("\nAll calculations finished.")
