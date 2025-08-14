# file: compute_T_ab.py
import numpy as np
import mpmath as mp
from tqdm import tqdm
import os # ディレクトリ操作のためにインポート

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
    n = 2 * k - 1
    const = (1 / (2 * mp.sqrt(mp.pi))) * ((-1)**n) * (0.5**n)
    x_arg = z / 2
    hermite_val = mp.hermite(n, x_arg)
    exp_val = mp.exp(-z**2 / 4)
    return const * hermite_val * exp_val

def calculate_T(a, b, k):
    """
    与えられた (a, b) の点に対してリッジレット変換 T(a, b) を計算します。
    """
    if a == 0:
        return 0.0
    
    # このファイル内で定義された f(x) と ψ(z) を使用
    integrand = lambda x: abs(a) * f(x) * mp.conj(psi(a * x - b, k))
    
    val = mp.quad(integrand, [-1, 1])
    return val.real

def compute_T_ab_grid(k_val=1, a_points=61, b_points=61):
    """
    T(a,b)のグリッド計算をまとめて実行し、結果のグリッドを返します。
    """
    print(f"Calculating T(a, b) grid for k={k_val}...")
    
    a_vals = np.linspace(-30, 30, a_points)
    b_vals = np.linspace(-30, 30, b_points)
    T_values = np.zeros((len(a_vals), len(b_vals)), dtype=np.float64)

    for i, a in enumerate(tqdm(a_vals, desc="Calculating T(a,b)")):
        for j, b in enumerate(b_vals):
            T_values[i, j] = calculate_T(a, b, k_val)
            
    return a_vals, b_vals, T_values

# -----------------------------------------------------------------
# このファイルを直接実行した場合の動作
# -----------------------------------------------------------------
if __name__ == '__main__':
    # mpmathの計算精度を設定
    mp.mp.dps = 25
    
    # 計算したいkの値を設定
    k_to_compute = 2
    
    print(f"This script is a module, but can be run directly to pre-calculate T(a,b) for k={k_to_compute}.")
    
    # T(a,b)のグリッドを計算
    a_vals, b_vals, T_values = compute_T_ab_grid(k_val=k_to_compute)
    
    # 保存先のディレクトリを指定
    output_dir = os.path.join('data', 'distribution_T')
    # ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # kの値を反映したファイル名を生成
    base_filename = f'T_ab_data_k{k_to_compute}.npz'
    # ファイル名とディレクトリを結合して完全なパスを作成
    output_filename = os.path.join(output_dir, base_filename)
    
    # NumPyの .npz 形式で関連データをまとめて保存
    np.savez(output_filename, a_vals=a_vals, b_vals=b_vals, T_values=T_values)
    
    print("\nCalculation finished.")
    print(f"T(a,b) data has been saved to '{output_filename}'")
