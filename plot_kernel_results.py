# file: plt_kernel_results.py
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_kernel_graph():
    """
    kernel_product_results.csvを読み込み、
    kに対するカーネルの値の片対数グラフを描画します。
    """
    
    # --- ファイルパスの設定 (修正箇所) ---
    # このスクリプト(plt_kernel_results.py)自身の場所を取得
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # このスクリプトはプロジェクトルートに置かれているため、script_dirがそのままproject_rootになる
    project_root = script_dir
    # プロジェクトルートからの相対パスとして、CSVファイルの完全なパスを構築
    # dataフォルダは 'data/kernel/kernel_product_results.csv' にあるためパスを修正
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

    # 正の値をプロット
    plt.scatter(positive_vals['k'], positive_vals['K_eta_psi'], 
                color='blue', label='K > 0 (k is even)', zorder=3)
    # 負の値をプロット (絶対値をとってプロット)
    plt.scatter(negative_vals['k'], abs(negative_vals['K_eta_psi']), 
                color='red', label='|K| where K < 0 (k is odd)', marker='x', zorder=3)

    # 傾向が分かりやすいように線で結ぶ (絶対値)
    plt.plot(df['k'], abs(df['K_eta_psi']), color='gray', linestyle='--', alpha=0.6, zorder=2)

    # 縦軸を対数スケールに設定
    plt.yscale('log')

    # ラベルとタイトル (英語表記)
    plt.xlabel("k")
    plt.ylabel("Absolute value of Kernel |K_eta_psi| (log scale)")
    plt.title("Absolute Kernel Value vs. k")
    
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    
    # グラフを画像として保存
    output_filename = os.path.join(project_root, 'kernel_plot.png') 
    print(f"Graph saved as '{output_filename}'")
    
    plt.show()


if __name__ == "__main__":
    plot_kernel_graph()
