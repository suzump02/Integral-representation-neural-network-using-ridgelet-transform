# file: compute_kernel_product.py
import mpmath as mp
import csv # CSVファイルへの書き込みのためにインポート
import os  # ディレクトリ操作のためにインポート
from tqdm import tqdm # 進捗バー表示のためにインポート

# tanh(z) のフーリエ変換
def eta_hat(z):
    return -1j * mp.pi / (2 * mp.sinh(mp.pi * z / 2))

# psi_hat(z) の定義
# 引数をkに統一し、不要な行を削除
def psi_hat(z, k):
    # kが奇数か偶数かで位相が変わる
    N = 2 * k - 1 
    phase = (1j) ** N
    return phase * (z ** N) * mp.exp(-(z ** 2))

# K_{eta,psi} の積分計算
def K_eta_psi(k):
    # 被積分関数
    f = lambda z: (mp.conj(psi_hat(z, k)) * eta_hat(z)) / abs(z)
    # 積分を実行し、実部のみを返す
    # 積分は複素数値になる可能性があるため、.realで実部を取得
    val = mp.quad(f, [-mp.inf, 0, mp.inf])
    return float(val.real)

if __name__ == "__main__":
    # mpmathの計算精度を設定
    mp.mp.dps = 30
    
    # 計算するkの範囲
    ks_to_compute = range(1, 51) # 1から50まで計算（範囲は適宜調整してください）
    results = []
    
    print(f"Calculating K_eta_psi for k = {ks_to_compute.start} to {ks_to_compute.stop - 1}...")

    # tqdmを使って進捗を表示しながら計算
    for k in tqdm(ks_to_compute, desc="Calculating K_eta_psi"):
        val = K_eta_psi(k)
        results.append({'k': k, 'K_eta_psi': val})
        # print(f"k={k}, K_eta_psi={val}") # 詳細表示が必要な場合はコメントを外す

    # === 計算結果をCSVファイルに保存 ===
    
    # 保存先のディレクトリを指定
    output_dir = os.path.join('data')
    # ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # ファイル名とディレクトリを結合して完全なパスを作成
    output_filename = os.path.join(output_dir, 'kernel_product_results.csv')
    
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            # 書き込むフィールド（列）の名前を定義
            fieldnames = ['k', 'K_eta_psi']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # ヘッダー（列名）を書き込む
            writer.writeheader()
            # 計算結果のデータを書き込む
            writer.writerows(results)
            
        print(f"\nCalculation finished.")
        print(f"Results have been saved to '{output_filename}'")

    except IOError as e:
        print(f"\nAn error occurred while writing to the file: {e}")
