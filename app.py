import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
from matplotlib.animation import FuncAnimation, PillowWriter
from svm_utils import (
    generate_concentric_data,
    rbf_kernel_transform,
    train_svm,
    compute_decision_grid,
    compute_3d_decision_surface,
)

st.set_page_config(page_title="SVM Kernel Trick 3D Visualization", layout="wide")
st.title("Support Vector Machine — Kernel Trick 3D Visualization")

with st.sidebar:
    st.header("Parameters")
    n_samples = st.slider("Number of Samples", min_value=50, max_value=500, value=200, step=10)
    noise = st.slider("Noise Level", min_value=0.01, max_value=0.3, value=0.08, step=0.01)
    gamma = st.slider("γ (Gamma)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    C = st.slider("C (Regularization)", min_value=0.1, max_value=100.0, value=10.0, step=0.1)
    st.divider()
    st.markdown("**RBF Kernel:**")
    st.latex(r"z = e^{-\gamma \|x\|^2}")
    st.divider()
    st.markdown("**Data Legend:**")
    st.markdown("- Blue = Inner circle (Class +1)")
    st.markdown("- Red = Outer circle (Class -1)")

X, y = generate_concentric_data(n_samples, noise)
model = train_svm(X, y, C, gamma)
z_values = rbf_kernel_transform(X, gamma)

tab1, tab2, tab3 = st.tabs(["2D Decision Boundary", "3D Kernel Transform", "Transform Animation"])

with tab1:
    st.subheader("2D Space — Raw Data & SVM Decision Boundary")
    st.markdown("In 2D space, concentric circles cannot be separated by a straight line. The RBF kernel SVM learns a circular decision boundary.")

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    xx, yy, Z = compute_decision_grid(model, X, resolution=150)

    ax1.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=["#FFC0CB", "#ADD8E6"], alpha=0.3)
    ax1.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["red", "black", "blue"], linestyles=["--", "-", "--"], linewidths=[1, 2, 1])

    blue_mask = y == 1
    red_mask = y == -1
    ax1.scatter(X[blue_mask, 0], X[blue_mask, 1], c="dodgerblue", edgecolors="k", s=50, label="Class +1 (inner)", zorder=5)
    ax1.scatter(X[red_mask, 0], X[red_mask, 1], c="tomato", edgecolors="k", s=50, label="Class -1 (outer)", zorder=5)

    sv = model.support_vectors_
    ax1.scatter(sv[:, 0], sv[:, 1], s=200, facecolors="none", edgecolors="k", linewidths=2, label="Support Vectors", zorder=10)

    ax1.set_xlabel("X₁", fontsize=14)
    ax1.set_ylabel("X₂", fontsize=14)
    ax1.set_title(f"2D Decision Boundary (γ={gamma}, C={C})", fontsize=16)
    ax1.legend(fontsize=12, loc="upper right")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Support Vectors", len(model.support_vectors_))
    col2.metric("Training Accuracy", f"{model.score(X, y):.2%}")
    col3.metric("Total Samples", len(X))

with tab2:
    st.subheader("3D Space — Data After RBF Kernel Transform & Decision Hyperplane")
    st.markdown("The RBF kernel maps 2D data into 3D: center points are lifted (z≈1), outer points are pushed down (z≈0), making them separable by a horizontal plane.")

    fig2 = plt.figure(figsize=(12, 9))
    ax2 = fig2.add_subplot(111, projection="3d")

    blue_mask = y == 1
    red_mask = y == -1

    ax2.scatter(X[blue_mask, 0], X[blue_mask, 1], z_values[blue_mask],
                c="dodgerblue", edgecolors="k", s=60, label="Class +1 (inner)", depthshade=True, alpha=0.9)
    ax2.scatter(X[red_mask, 0], X[red_mask, 1], z_values[red_mask],
                c="tomato", edgecolors="k", s=60, label="Class -1 (outer)", depthshade=True, alpha=0.9)

    sv = model.support_vectors_
    sv_z = rbf_kernel_transform(sv, gamma)
    ax2.scatter(sv[:, 0], sv[:, 1], sv_z, s=200, facecolors="none", edgecolors="k", linewidths=2, label="Support Vectors", zorder=10)

    xx3d, yy3d, zz_rbf, zz_decision = compute_3d_decision_surface(model, gamma, resolution=60)

    ax2.plot_surface(xx3d, yy3d, zz_rbf, alpha=0.15, cmap=cm.coolwarm, edgecolor="gray", linewidth=0.3)

    boundary_mask = np.abs(zz_decision) < 0.3
    if np.any(boundary_mask):
        zz_boundary = np.where(boundary_mask, zz_rbf, np.nan)
        ax2.plot_surface(xx3d, yy3d, zz_boundary, alpha=0.6, color="gold", edgecolor="orange", linewidth=0.5)

    z_separation = np.mean(z_values[blue_mask]) * 0.5 + np.mean(z_values[red_mask]) * 0.5
    xx_plane, yy_plane = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))
    zz_plane = np.full_like(xx_plane, z_separation)
    ax2.plot_surface(xx_plane, yy_plane, zz_plane, alpha=0.2, color="green", edgecolor="darkgreen", linewidth=0.5)

    ax2.set_xlabel("X₁", fontsize=12)
    ax2.set_ylabel("X₂", fontsize=12)
    ax2.set_zlabel("Z = exp(-γ‖x‖²)", fontsize=12)
    ax2.set_title(f"3D Kernel Transform (γ={gamma})", fontsize=16)
    ax2.legend(fontsize=10, loc="upper left")
    ax2.view_init(elev=25, azim=-60)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown("**Tip:** The static 3D view is above. Below is an interactive Plotly 3D chart — use your mouse to rotate and zoom.")

    try:
        import plotly.graph_objects as go

        fig_plotly = go.Figure()

        fig_plotly.add_trace(go.Scatter3d(
            x=X[blue_mask, 0], y=X[blue_mask, 1], z=z_values[blue_mask],
            mode="markers",
            marker=dict(size=6, color="dodgerblue", opacity=0.8, line=dict(color="black", width=1)),
            name="Class +1 (inner)",
        ))
        fig_plotly.add_trace(go.Scatter3d(
            x=X[red_mask, 0], y=X[red_mask, 1], z=z_values[red_mask],
            mode="markers",
            marker=dict(size=6, color="tomato", opacity=0.8, line=dict(color="black", width=1)),
            name="Class -1 (outer)",
        ))
        fig_plotly.add_trace(go.Scatter3d(
            x=sv[:, 0], y=sv[:, 1], z=sv_z,
            mode="markers",
            marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(color="black", width=2)),
            name="Support Vectors",
        ))

        xx_flat = xx3d.flatten()
        yy_flat = yy3d.flatten()
        zz_flat = zz_rbf.flatten()
        fig_plotly.add_trace(go.Surface(
            x=xx3d, y=yy3d, z=zz_rbf,
            opacity=0.2,
            colorscale=[[0, "lightblue"], [1, "lightcoral"]],
            showscale=False,
            name="Kernel Surface",
        ))

        fig_plotly.add_trace(go.Surface(
            x=xx_plane, y=yy_plane, z=zz_plane,
            opacity=0.3,
            colorscale=[[0, "green"], [1, "green"]],
            showscale=False,
            name="Decision Hyperplane",
        ))

        fig_plotly.update_layout(
            title=f"Interactive 3D Kernel Transform (γ={gamma})",
            scene=dict(
                xaxis_title="X₁",
                yaxis_title="X₂",
                zaxis_title="Z = exp(-γ‖x‖²)",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
            ),
            width=1000,
            height=700,
        )
        st.plotly_chart(fig_plotly, width="stretch")
    except ImportError:
        st.warning("Plotly is not installed. Run `pip install plotly` to enable interactive 3D.")

with tab3:
    st.subheader("Animation — 2D to 3D Kernel Transform Process")
    st.markdown("Watch how data points are gradually lifted from the 2D plane into 3D space by the RBF kernel, becoming separable by a hyperplane.")

    n_frames = 40

    fig3, ax3 = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "3d"})

    def make_frame(frame_idx):
        ax3.clear()
        progress = frame_idx / (n_frames - 1)
        eased = progress ** 2 * (3 - 2 * progress)

        current_z = z_values * eased

        blue_mask = y == 1
        red_mask = y == -1

        ax3.scatter(X[blue_mask, 0], X[blue_mask, 1], current_z[blue_mask],
                    c="dodgerblue", edgecolors="k", s=50, alpha=0.8, label="Class +1")
        ax3.scatter(X[red_mask, 0], X[red_mask, 1], current_z[red_mask],
                    c="tomato", edgecolors="k", s=50, alpha=0.8, label="Class -1")

        if progress > 0.5:
            plane_alpha = (progress - 0.5) * 2
            z_sep = np.mean(z_values[blue_mask]) * 0.5 + np.mean(z_values[red_mask]) * 0.5
            z_sep_current = z_sep * eased
            xx_p, yy_p = np.meshgrid(np.linspace(-1.5, 1.5, 20), np.linspace(-1.5, 1.5, 20))
            zz_p = np.full_like(xx_p, z_sep_current)
            ax3.plot_surface(xx_p, yy_p, zz_p, alpha=0.3 * plane_alpha, color="green", edgecolor="darkgreen", linewidth=0.3)

        elev = 10 + 20 * eased
        azim = -60 + 30 * eased
        ax3.view_init(elev=elev, azim=azim)

        ax3.set_xlabel("X₁")
        ax3.set_ylabel("X₂")
        ax3.set_zlabel("Z")
        ax3.set_title(f"Kernel Transform Progress: {progress:.0%}", fontsize=14)
        ax3.set_xlim(-2, 2)
        ax3.set_ylim(-2, 2)
        ax3.set_zlim(0, 1.1)
        if progress < 0.3:
            ax3.legend(fontsize=9, loc="upper left")

    frames = []
    for i in range(n_frames):
        make_frame(i)
        buf = io.BytesIO()
        fig3.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        frames.append(buf.getvalue())

    plt.close(fig3)

    gif_buf = io.BytesIO()
    from PIL import Image
    pil_frames = [Image.open(io.BytesIO(f)) for f in frames]
    pil_frames[0].save(
        gif_buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=80,
        loop=0,
    )
    gif_buf.seek(0)
    gif_base64 = base64.b64encode(gif_buf.read()).decode("utf-8")

    st.markdown(
        f'<img src="data:image/gif;base64,{gif_base64}" width="100%">',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("""
### Key Concepts Summary

| Step | Description |
|------|-------------|
| **1. Raw 2D Data** | Blue points in the inner circle, red points in the outer circle — not linearly separable |
| **2. Kernel Transform** | RBF kernel `z = exp(-γ‖x‖²)` lifts center points up and pushes outer points down |
| **3. 3D Separable** | After transform, the two classes are naturally separated along the Z-axis by a horizontal hyperplane |
| **4. Support Vectors** | Points closest to the decision boundary (hollow circles) that determine the hyperplane position |
""")
