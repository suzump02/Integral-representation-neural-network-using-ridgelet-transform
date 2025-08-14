# file: plot_K_eta_psi_real.py
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

# Fourier transform of tanh(z)
def eta_hat(z):
    return -1j * mp.pi / (2 * mp.sinh(mp.pi * z / 2))

# psi_hat with phase factor i^(2k-1)
def psi_hat(z, k):
    phase = (1j) ** (2 * k - 1)
    return phase * (z ** (2 * k - 1)) * mp.e ** (-(z ** 2))

# K_{eta,psi} integral (real part only)
def K_eta_psi(k):
    f = lambda z: (mp.conj(psi_hat(z, k)) * eta_hat(z)) / abs(z)
    return float(mp.quad(f, [-mp.inf, 0, mp.inf]).real)  # 実部のみ

if __name__ == "__main__":
    mp.mp.dps = 30  # 高精度計算
    ks = list(range(1, 50))
    values = []

    for k in ks:
        val = K_eta_psi(k)
        values.append(val)
        print(f"k={k}, K_eta_psi={val}")

    # プロット（点＋色分け）
    colors = plt.cm.viridis(np.linspace(0, 1, len(ks)))
    for i, k in enumerate(ks):
        plt.scatter(k, values[i], color=colors[i], label=f"k={k}", s=50)

    plt.figure(figsize=(10, 6)) # プロットサイズを調整

    # プロット（点＋色分け）
    # 負の値を含むため、絶対値を取らずに元の値を直接プロットします
    colors = plt.cm.viridis(np.linspace(0, 1, len(ks)))
    for i, k in enumerate(ks):
        plt.scatter(k, values[i], color=colors[i], label=f"k={k}", s=50, zorder=3)

    # 縦軸を対称対数スケール（symlog）に設定
    # linthreshは、線形スケールとして扱う0周りの範囲の閾値を指定します
    # この値より絶対値が小さい範囲が線形スケールになります
    plt.yscale('symlog', linthresh=1e-5)

    # 0の位置に水平線を引いて、正負を分かりやすくします
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)

    # ラベルとタイトル
    # 以前ご指定いただいたように、ラベルは英語表記にします
    plt.xlabel("k")
    plt.ylabel("Re($K_{\\eta, \\psi}$)")
    plt.title("Real part of $K_{\\eta, \\psi}$ vs k (Symmetric Log Scale)")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, which="both", ls="--")
    plt.show()

