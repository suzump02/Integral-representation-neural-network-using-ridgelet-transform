# ======================================================
# train_and_plot_param_dist_a_b.py
# Ridgelet型1層NNのパラメータ(a,b)分布を可視化（解析的T(a,b)と比較可能な範囲設定）
# ======================================================

import os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import colors 
from mpl_toolkits.mplot3d import Axes3D

# === 解析側の a,b 範囲を合わせるために import ===
from create_data.compute_T_ab import compute_T_ab_grid

# =======================
#  設定
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_hidden = 10000
N_epochs = 5000
lr = 1e-4
weight_decay = 1e-6
seed = 123

np.random.seed(seed)
torch.manual_seed(seed)

# =======================
#  データ生成: x ∈ [-1, 1]
# =======================
n_train = 2000
xs = np.linspace(-1, 1, n_train, dtype=np.float32)
ys = np.sin(2 * np.pi * xs).astype(np.float32)
x_train = torch.from_numpy(xs).unsqueeze(1).to(device)
y_train = torch.from_numpy(ys).unsqueeze(1).to(device)

# =======================
#  モデル定義
# =======================
class OneHiddenTanh(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        # a,bの初期範囲を理論グリッドに合わせる
        a_min, a_max = -30.0, 30.0
        b_min, b_max = -30.0, 30.0

        self.a = nn.Parameter(torch.empty(n_hidden, 1).uniform_(a_min, a_max))
        self.b = nn.Parameter(torch.empty(n_hidden).uniform_(b_min, b_max))
        self.c = nn.Parameter(torch.randn(n_hidden, 1) * 0.1)

    def forward(self, x):
        ax = x @ self.a.T
        z = ax - self.b
        act = torch.tanh(z)
        out = act @ self.c
        return out, act

# =======================
#  学習
# =======================
model = OneHiddenTanh(N_hidden).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = nn.MSELoss()

loss_history = []
for epoch in range(1, N_epochs + 1):
    optimizer.zero_grad()
    y_pred, _ = model(x_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 200 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{N_epochs}, loss={loss.item():.6f}")

# =======================
#  パラメータ抽出
# =======================
a_np = model.a.detach().cpu().numpy().reshape(-1)
b_np = model.b.detach().cpu().numpy().reshape(-1)
c_np = model.c.detach().cpu().numpy().reshape(-1)

# =======================
#  出力先
# =======================
project_root = os.path.dirname(os.path.abspath(__file__))
output_root = os.path.join(project_root, "result-figures")
dirs = {
    "fit": os.path.join(output_root, "fit-results"),
    "param": os.path.join(output_root, "param-3d"),
    "density": os.path.join(output_root, "density"),
}
for p in dirs.values():
    os.makedirs(p, exist_ok=True)

# =======================
#  可視化1: sin波フィット
# =======================
model.eval()
with torch.no_grad():
    y_pred_all, _ = model(torch.from_numpy(xs).unsqueeze(1).to(device))
y_pred_all = y_pred_all.cpu().numpy().reshape(-1)

plt.figure(figsize=(8,4))
plt.plot(xs, ys, label='target sin(2πx)', color='tab:blue')
plt.plot(xs, y_pred_all, label='model prediction', color='tab:orange', linewidth=2)
plt.legend()
plt.title('Function fit(N={})'.format(N_hidden))
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig(os.path.join(dirs["fit"], f'fit_plot(N={N_hidden}, N_epoch={N_epochs}, n_train={n_train}).png'), dpi=200)

# =======================
#  可視化2: 損失曲線
# =======================
plt.figure(figsize=(6,3))
plt.plot(loss_history)
plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('MSE (log scale)')
plt.title('Training loss(N={})'.format(N_hidden))
plt.tight_layout()
plt.savefig(os.path.join(dirs["fit"], f'loss_history(N={N_hidden}, N_epoch={N_epochs}, n_train={n_train}).png'), dpi=200)

# =======================
#  可視化3: (a,b,c)の3D散布図
# =======================
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(a_np, b_np, c_np, c=c_np, cmap='coolwarm', s=20)
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('c')
fig.colorbar(p, ax=ax, label='c value')
plt.title('Parameter scatter (a, b, c)(N={})'.format(N_hidden))
plt.tight_layout()
plt.savefig(os.path.join(dirs["param"], f'param_scatter_3d(N={N_hidden}, N_epoch={N_epochs}, n_train={n_train}).png'), dpi=200)

# =======================
#  可視化4: 離散 Ridgelet 密度 T(a,b)
# =======================

# --- 理論側のグリッドを取得して合わせる ---
a_vals_ref, b_vals_ref, _ = compute_T_ab_grid(
    k_val=1,
    a_points=401,
    b_points=401,
    only_grid=True
    )

a_min, a_max = a_vals_ref.min(), a_vals_ref.max()
b_min, b_max = b_vals_ref.min(), b_vals_ref.max()
na, nb = len(a_vals_ref), len(b_vals_ref)

Delta = (a_max - a_min) * (b_max - b_min) / (na * nb)
print(f"[INFO] Using theoretical grid range: a∈[{a_min:.2f},{a_max:.2f}], b∈[{b_min:.2f},{b_max:.2f}]")

# --- Ridgelet 密度を構築 ---
T_samples = c_np / Delta
H, a_edges, b_edges = np.histogram2d(a_np, b_np, bins=[na, nb],
                                     range=[[a_min, a_max], [b_min, b_max]],
                                     weights=T_samples)

Agrid, Bgrid = np.meshgrid(0.5*(a_edges[:-1]+a_edges[1:]),
                           0.5*(b_edges[:-1]+b_edges[1:]))

# --- 検算 ---
print(f"[CHECK] ΣH·Δ ≈ Σc : {H.sum()*Delta:.4e} vs {c_np.sum():.4e}")

# --- 描画 ---
fig, ax = plt.subplots(figsize=(6,5))
# vmin, vmax = np.nanmin(H), np.nanmax(H)
# vabs = max(abs(vmin), abs(vmax))
norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1)

pcm = ax.imshow(H.T, origin='lower', extent=(a_min, a_max, b_min, b_max),
                aspect='auto', cmap='RdBu_r', norm=norm)
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_title('Learned T(a,b) (aligned with analytic grid)(N={})'.format(N_hidden))
plt.colorbar(pcm, ax=ax, label='T (per area)')
plt.tight_layout()
plt.savefig(os.path.join(dirs["density"], f'finite(N={N_hidden}, N_epoch={N_epochs}, n_train={n_train})_T_ab.png'), dpi=200)
print("\n[INFO] All results saved successfully.")