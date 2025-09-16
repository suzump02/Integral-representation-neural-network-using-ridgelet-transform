import mpmath as mp

# --- ターゲット関数 f(x) の定義 ---
def sin_2pix(x):
    return mp.sin(2 * mp.pi * x)

# --- 活性化関数 η(z) とそのフーリエ変換 ---
def tanh(z):
    return mp.tanh(z)

def tanh_hat(zeta):
    return -1j * mp.pi / (2 * mp.sinh(mp.pi * zeta / 2))

# --- リッジレット関数 ψ(z) とそのフーリエ変換 ---
@mp.memoize
def psi_odd(z, k):
    n = 2 * k - 1
    const = (1 / (2 * mp.sqrt(mp.pi))) * ((-1)**n) * (0.5**n)
    x_arg = z / 2
    hermite_val = mp.hermite(n, x_arg)
    exp_val = mp.exp(-z**2 / 4)
    return const * hermite_val * exp_val

def psi_odd_hat(zeta, k):
    n = 2 * k - 1
    phase = (1j) ** n
    return phase * (zeta ** n) * mp.exp(-(zeta ** 2))