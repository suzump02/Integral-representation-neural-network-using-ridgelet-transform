# file: verify_reconstruction_m1_sin.py
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import hermite
import matplotlib.pyplot as plt

# --- 活性化関数 ---
def eta(z):
    return np.tanh(z)

# --- psi(z) の定義（Fourier逆変換済みの形） ---
def psi(z, k):
    # i^{2k-1} = i^(odd) = ±i
    i_factor = 1j**(2*k - 1)
    coeff = i_factor / (2 * np.sqrt(np.pi)) * (-1)**(2*k - 1) * (0.5)**(2*k - 1)
    Hn = hermite(2*k - 1)  # Hermite 多項式
    return coeff * Hn(z / 2) * np.exp(-z**2 / 4)

# --- f(x) ---
def f(x):
    return np.sin(x)

# --- Ridgelet変換 ---
def ridgelet_transform(a, b, k):
    integrand = lambda x: f(x) * np.conjugate(eta(a*x - b))
    val, _ = quad(integrand, -10, 10, limit=200)
    return val

# --- 再構成 ---
def reconstruct_f(x, k):
    def integrand(b, a):
        return ridgelet_transform(a, b, k) * psi(a*x - b, k) / (a**2)
    val, _ = dblquad(
        integrand,
        -5, 5,      # a の範囲
        lambda _: -10, lambda _: 10  # b の範囲
    )
    return val

if __name__ == "__main__":
    k = 1  # 簡単のため k=1 から
    xs = np.linspace(-3, 3, 50)
    f_true = f(xs)
    f_rec = [reconstruct_f(x, k) for x in xs]

    plt.plot(xs, f_true, label="Original sin(x)", color="blue")
    plt.plot(xs, np.real(f_rec), "o", label="Reconstructed", color="orange")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.title(f"Reconstruction m=1, k={k}")
    plt.show()