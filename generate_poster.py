import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from svm_utils import (
    generate_concentric_data,
    rbf_kernel_transform,
    train_svm,
    compute_decision_grid,
    compute_3d_decision_surface,
)

X, y = generate_concentric_data(300, 0.08)
gamma = 1.0
C = 10.0
model = train_svm(X, y, C, gamma)
z_values = rbf_kernel_transform(X, gamma)
blue_mask = y == 1
red_mask = y == -1

fig = plt.figure(figsize=(24, 14), facecolor="#1a1a2e")

fig.text(0.5, 0.95, "Support Vector Machine — RBF Kernel Trick",
         fontsize=36, fontweight="bold", color="white", ha="center", va="top")
fig.text(0.5, 0.91, "From Non-linearly Separable 2D Data to Linearly Separable 3D Space",
         fontsize=18, color="#aaaacc", ha="center", va="top")

ax1 = fig.add_subplot(131)
ax1.set_facecolor("#16213e")
xx, yy, Z = compute_decision_grid(model, X, resolution=150)
ax1.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=["#e74c3c33", "#3498db33"], alpha=0.4)
ax1.contour(xx, yy, Z, levels=[0], colors="white", linewidths=2.5)
ax1.scatter(X[blue_mask, 0], X[blue_mask, 1], c="#3498db", edgecolors="white",
            s=40, linewidths=0.5, label="Class +1 (Inner)", zorder=5)
ax1.scatter(X[red_mask, 0], X[red_mask, 1], c="#e74c3c", edgecolors="white",
            s=40, linewidths=0.5, label="Class -1 (Outer)", zorder=5)
sv = model.support_vectors_
ax1.scatter(sv[:, 0], sv[:, 1], s=180, facecolors="none", edgecolors="#f1c40f",
            linewidths=2.5, label="Support Vectors", zorder=10)
ax1.set_xlabel("X₁", fontsize=16, color="white")
ax1.set_ylabel("X₂", fontsize=16, color="white")
ax1.set_title("① 2D Space: Not Linearly Separable", fontsize=18, color="white", fontweight="bold", pad=15)
ax1.tick_params(colors="white")
ax1.legend(fontsize=11, loc="upper right", facecolor="#16213e", edgecolor="white",
           labelcolor="white")
ax1.set_aspect("equal")
ax1.grid(True, alpha=0.15, color="white")
for spine in ax1.spines.values():
    spine.set_color("white")

ax2 = fig.add_subplot(132, projection="3d")
ax2.set_facecolor("#16213e")
ax2.scatter(X[blue_mask, 0], X[blue_mask, 1], z_values[blue_mask],
            c="#3498db", edgecolors="white", s=40, linewidths=0.5, alpha=0.9,
            label="Class +1 (lifted)")
ax2.scatter(X[red_mask, 0], X[red_mask, 1], z_values[red_mask],
            c="#e74c3c", edgecolors="white", s=40, linewidths=0.5, alpha=0.9,
            label="Class -1 (pushed down)")
xx3d, yy3d, zz_rbf, zz_decision = compute_3d_decision_surface(model, gamma, resolution=50)
ax2.plot_surface(xx3d, yy3d, zz_rbf, alpha=0.12, cmap=cm.coolwarm, edgecolor="gray",
                 linewidth=0.2)
z_sep = np.mean(z_values[blue_mask]) * 0.5 + np.mean(z_values[red_mask]) * 0.5
xx_p, yy_p = np.meshgrid(np.linspace(-2, 2, 25), np.linspace(-2, 2, 25))
zz_p = np.full_like(xx_p, z_sep)
ax2.plot_surface(xx_p, yy_p, zz_p, alpha=0.25, color="#2ecc71", edgecolor="#27ae60",
                 linewidth=0.5)
ax2.set_xlabel("X₁", fontsize=14, color="white")
ax2.set_ylabel("X₂", fontsize=14, color="white")
ax2.set_zlabel("Z = exp(-γ‖x‖²)", fontsize=14, color="white")
ax2.set_title("② Kernel Transform: z = exp(-γ‖x‖²)", fontsize=18, color="white",
              fontweight="bold", pad=15)
ax2.tick_params(colors="white")
ax2.legend(fontsize=10, loc="upper left", facecolor="#16213e", edgecolor="white",
           labelcolor="white")
ax2.view_init(elev=25, azim=-60)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.xaxis.pane.set_edgecolor("#333355")
ax2.yaxis.pane.set_edgecolor("#333355")
ax2.zaxis.pane.set_edgecolor("#333355")

ax3 = fig.add_subplot(133, projection="3d")
ax3.set_facecolor("#16213e")
ax3.scatter(X[blue_mask, 0], X[blue_mask, 1], z_values[blue_mask],
            c="#3498db", edgecolors="white", s=50, linewidths=0.5, alpha=0.95)
ax3.scatter(X[red_mask, 0], X[red_mask, 1], z_values[red_mask],
            c="#e74c3c", edgecolors="white", s=50, linewidths=0.5, alpha=0.95)
sv_z = rbf_kernel_transform(sv, gamma)
ax3.scatter(sv[:, 0], sv[:, 1], sv_z, s=180, facecolors="none", edgecolors="#f1c40f",
            linewidths=2.5, zorder=10)
ax3.plot_surface(xx_p, yy_p, zz_p, alpha=0.35, color="#2ecc71", edgecolor="#27ae60",
                 linewidth=0.8)
ax3.set_xlabel("X₁", fontsize=14, color="white")
ax3.set_ylabel("X₂", fontsize=14, color="white")
ax3.set_zlabel("Z", fontsize=14, color="white")
ax3.set_title("③ 3D Space: Linearly Separable", fontsize=18, color="white",
              fontweight="bold", pad=15)
ax3.tick_params(colors="white")
ax3.view_init(elev=20, azim=-45)
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False
ax3.xaxis.pane.set_edgecolor("#333355")
ax3.yaxis.pane.set_edgecolor("#333355")
ax3.zaxis.pane.set_edgecolor("#333355")

fig.text(0.18, 0.03, "Non-linearly separable in 2D", fontsize=14, color="#aaaacc",
         ha="center", style="italic")
fig.text(0.50, 0.03, "RBF kernel maps to 3D feature space", fontsize=14,
         color="#aaaacc", ha="center", style="italic")
fig.text(0.83, 0.03, "Hyperplane separates classes in 3D", fontsize=14,
         color="#aaaacc", ha="center", style="italic")

for x_pos in [0.345, 0.678]:
    fig.text(x_pos, 0.50, "→", fontsize=48, color="#f1c40f", ha="center", va="center",
             fontweight="bold")

plt.subplots_adjust(top=0.85, bottom=0.08, left=0.05, right=0.97, wspace=0.15)
fig.savefig("poster.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("poster.png saved successfully")
plt.close(fig)
