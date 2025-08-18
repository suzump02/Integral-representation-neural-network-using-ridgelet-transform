# file: plot_ridgelet_transform_3d.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import mpmath as mp

# create_dataフォルダの中のcompute_T_abからインポート
# compute_T_ab_batch.py内の関数を使うので、ファイル名を合わせる
from create_data.compute_T_ab import compute_T_ab_grid

def plot_T_ab_3d():
    """
    T(a, b) のデータを準備し、3Dサーフェスプロットとして描画します。
    """
    mp.mp.dps = 25

    # === ステップ1: T(a,b)のデータを準備 ===

    k_val = 1

    # --- パス設定の修正箇所 ---
    # このスクリプト自身の場所（プロジェクトルート）を取得
    project_root = os.path.dirname(os.path.abspath(__file__))
    # プロジェクトルートを基準にデータファイルへの完全なパスを構築
    data_filename = os.path.join(project_root, 'data', 'distribution_T', f'T_ab_data_k{k_val}.npz')

    # T(a,b)を保存したファイルがあれば読み込み、なければ計算する
    if os.path.exists(data_filename):
        print(f"Loading T(a,b) data from '{data_filename}'...")
        data = np.load(data_filename)
        a_vals, b_vals, T_values = data['a_vals'], data['b_vals'], data['T_values']
    else:
        print(f"'{data_filename}' not found. Calculating from scratch...")
        # create_data/compute_T_ab_batch.py の関数を呼び出してグリッドを計算
        a_vals, b_vals, T_values = compute_T_ab_grid(k_val=k_val)
        
        # 保存先ディレクトリが存在しない場合に作成
        output_dir = os.path.dirname(data_filename)
        os.makedirs(output_dir, exist_ok=True)
        
        # 計算結果を次回のために正しいパスに保存
        np.savez(data_filename, a_vals=a_vals, b_vals=b_vals, T_values=T_values)
        print(f"Data saved to '{data_filename}' for future use.")

    # === ステップ2: 3Dプロット ===
    
    print("Plotting 3D surface...")
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    B, A = np.meshgrid(b_vals, a_vals)
    surf = ax.plot_surface(B, A, T_values, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Value of T(a, b)')
    ax.set_xlabel('b')
    ax.set_ylabel('a')
    ax.set_zlabel('T(a, b)')
    ax.set_title(f'3D Surface of Ridgelet Transform T(a,b) for k={k_val}')
    
    plt.show()

if __name__ == "__main__":
    plot_T_ab_3d()
