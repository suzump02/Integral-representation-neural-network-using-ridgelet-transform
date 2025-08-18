# file: plt_kernel_results.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np # 計算のためにNumPyをインポート

def plot_kernel_graph():
    """
    kernel_product_results.csvを読み込み、
    kに対するカーネルの値の片対数グラフと理論的な傾向線を描画します。
    """
    
    # --- ファイルパスの設定 ---
    # このスクリプト(plt_kernel_results.py)自身の場所を取得
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # このスクリプトはプロジェクトルートに置かれているため、script_dirがそのままproject_rootになる
    project_root = script_dir
    # プロジェクトルートからの相対パスとして、CSVファイルの完全なパスを構築
    csv_path = os.path.join(project_root, 'data', 'kernel_product_results.csv')

    # --- データの読み込み ---
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data from '{csv_path}'")
    except FileNotFoundError:
        print(f"Error: Data file not found at '{csv_path}'")
        print("Please make sure 'compute_kernel_product.py' has been run and the file exists.")
        return

    # --- グラフの描画 ---
    print("Plotting the graph...")
    plt.figure(figsize=(12, 8))

    # 値が正負に振動するため、色分けしてプロット
    positive_vals = df[df['K_eta_psi'] >= 0]
    negative_vals = df[df['K_eta_psi'] < 0]

    # 正の値をプロット (マーカーサイズを調整)
    plt.scatter(positive_vals['k'], positive_vals['K_eta_psi'], 
                color='blue', label='K > 0 (k is even)', zorder=3, s=20)
    # 負の値をプロット (絶対値をとってプロット、マーカーサイズを調整)
    plt.scatter(negative_vals['k'], abs(negative_vals['K_eta_psi']), 
                color='red', label='|K| where K < 0 (k is odd)', marker='x', zorder=3, s=20)

    # --- 理論的な傾向線の計算と描画 ---
    # k=0.5以下でlogの引数が負になるのを防ぐ
    valid_k = df[df['k'] > 0.5]
    if not valid_k.empty:
        # 理論的な漸近形を計算 (対数領域で)
        k_theory = valid_k['k'] - 0.5
        log_k_theory = k_theory * np.log(k_theory) - k_theory
        
        # 実際のデータの対数を計算
        log_k_data = np.log(np.abs(valid_k['K_eta_psi']))
        
        # データのスケールに合うように、理論曲線を線形フィットさせる
        # log_k_data ≈ slope * log_k_theory + intercept
        slope, intercept = np.polyfit(log_k_theory, log_k_data, 1)
        
        # スケールを合わせた理論曲線を計算 (元のスケールに戻す)
        theory_vals_scaled = np.exp(slope * log_k_theory + intercept)
        
        # 理論曲線を重ねて描画
        plt.plot(valid_k['k'], theory_vals_scaled, color='green', linestyle='-', 
                 label='(k-1/2)log(k-1/2)', zorder=2, linewidth=2.5)

    # 縦軸を対数スケールに設定
    plt.yscale('log')

    # ラベルとタイトル (英語表記)
    plt.xlabel("k")
    plt.ylabel("Absolute value of Kernel |K_eta_psi| (log scale)")
    plt.title("Absolute Kernel Value vs. k")
    
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    
    
    plt.show()


if __name__ == "__main__":
    plot_kernel_graph()
