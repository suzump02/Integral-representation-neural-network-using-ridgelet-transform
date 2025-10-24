import mpmath as mp
import numpy as np

def sanity_check_framework():
    """
    All calculations are brought into a single file to definitively
    diagnose the scaling issue.
    This script measures the scale of each component of the calculation.
    """
    mp.mp.dps = 50
    k = 2
    
    print("\n--- Framework Sanity Check ---")
    print(f"Using k={k}, with precision dps={mp.mp.dps}")

    # --- Step 1: Define all functions in one place ---
    # これにより、ファイル間の定義の不整合を完全に排除します。

    @mp.memoize
    def psi(z, k_local):
        """
        【単一性定義】hat_psiと厳密にフーリエ変換ペアとなる実空間関数 psi
        """
        n_local = 2 * k_local - 1
        # 以前の議論で特定した、数学的に正しい定数を使用
        const = mp.sqrt(mp.pi) * (-1) * (0.5**n_local)
        hermite_val = mp.hermite(n_local, z / 2)
        exp_val = mp.exp(-z**2 / 4)
        return const * hermite_val * exp_val

    @mp.memoize
    def hat_psi(zeta, k_local):
        """
        【単一性定義】Ryanさんのノート(PDF)に基づく純粋なフーリエ空間関数 hat_psi
        """
        n_local = 2 * k_local - 1
        return (1j * zeta)**n_local * mp.exp(-zeta**2)

    def eta(z):
        """
        活性化関数（実空間）
        """
        return mp.tanh(z)

    @mp.memoize
    def hat_eta(zeta):
        """
        活性化関数のフーリエ変換
        """
        return -1j * mp.pi / (2 * mp.sinh(mp.pi * zeta / 2))

    def f(x):
        """
        ターゲット信号
        """
        return mp.sin(2 * mp.pi * x)

    # --- Step 2: Measure the scale of each component ---
    # 各計算ステップで値のスケールを測定し、異常な増幅がどこで起きるかを特定します。

    # Measurement Point 1: Scale of psi(z)
    psi_at_one = psi(1.0, k)
    # ★★★ 修正箇所 ★★★: mpf型をfloat型に変換して表示
    print(f"\n[INFO] 1. Scale of psi(z,k): psi(1, k={k}) = {float(psi_at_one):.6f}")

    # Measurement Point 2: Scale of Kernel K
    integrand_K = lambda zeta: mp.conj(hat_psi(zeta, k)) * hat_eta(zeta) / abs(zeta)
    K_val = mp.quad(integrand_K, [-mp.inf, 0, mp.inf])
    # ★★★ 修正箇所 ★★★: mpf型をfloat型に変換して表示
    print(f"[INFO] 2. Scale of Kernel K: K(k={k}) = {float(K_val.real):.6f}")

    # Measurement Point 3: Scale of T(a,b) at a sample point (a=1, b=0)
    a_sample, b_sample = 1.0, 0.0
    # 現在のs=1フレームワークに従って計算
    integrand_T = lambda x: abs(a_sample) * f(x) * mp.conj(psi(a_sample * x - b_sample, k))
    T_val_sample = mp.quad(integrand_T, [-1, 1])
    # ★★★ 修正箇所 ★★★: mpf型をfloat型に変換して表示
    print(f"[INFO] 3. Scale of T(a,b): T(a=1, b=0, k={k}) = {float(T_val_sample.real):.6f}")

    # Measurement Point 4: Scale of the reconstruction integral g(x) at x=0.25
    # 非常に粗い3x3グリッドを使い、スケールのオーダーを素早く見積もります
    x_sample = 0.25
    a_grid = np.linspace(-30, 30, 3)
    b_grid = np.linspace(-30, 30, 3)
    delta_a = a_grid[1] - a_grid[0]
    delta_b = b_grid[1] - b_grid[0]
    
    reconstructed_sum = 0.0
    for a in a_grid:
        if abs(a) < 1e-9: continue
        for b in b_grid:
            # このチェックでは、T(a,b)がサンプル点の値で一定だと仮定します
            T_ab = T_val_sample.real 
            eta_val = eta(a * x_sample - b)
            reconstructed_sum += T_ab * eta_val * (delta_a * delta_b) / abs(a)
            
    # ★★★ 修正箇所 ★★★: reconstructed_sumも念のためfloat型に変換
    print(f"[INFO] 4. Scale of integral g(x): g(x=0.25, k={k}) with 3x3 grid approx {float(reconstructed_sum):.6f}")
    
    # Measurement Point 5: Final reconstructed value
    final_value = reconstructed_sum / K_val.real
    target_value = f(x_sample)
    # ★★★ 修正箇所 ★★★: mpf型をfloat型に変換して表示
    print(f"[INFO] 5. Final Reconstructed Value (g(x)/K) approx {float(final_value):.6f}")
    print(f"          (Target value f(0.25) should be {float(target_value):.6f})")

if __name__ == "__main__":
    sanity_check_framework()

