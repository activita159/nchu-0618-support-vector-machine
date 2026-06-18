import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_circles


def generate_concentric_data(n_samples=200, noise=0.1, random_state=42):
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=0.4,
        random_state=random_state,
    )
    y = np.where(y == 0, -1, 1)
    return X, y


def rbf_kernel_transform(X, gamma=1.0):
    return np.exp(-gamma * np.sum(X ** 2, axis=1))


def train_svm(X, y, C=1.0, gamma=1.0):
    model = SVC(kernel="rbf", C=C, gamma=gamma)
    model.fit(X, y)
    return model


def compute_decision_grid(model, X, resolution=100):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(grid_points).reshape(xx.shape)
    return xx, yy, Z


def compute_3d_decision_surface(model, gamma, resolution=80):
    x_range = np.linspace(-2, 2, resolution)
    y_range = np.linspace(-2, 2, resolution)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z_decision = model.decision_function(grid_points).reshape(xx.shape)
    Z_rbf = rbf_kernel_transform(grid_points, gamma).reshape(xx.shape)
    return xx, yy, Z_rbf, Z_decision
