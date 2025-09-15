import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from T00_CartPole import remap_angle
from T20_utils import X_Y_dataset_step, simulate_rollout, build_kernel_matrix, find_alpha, label_subplots

# --- Plot settings ---
plt.rcParams.update({'font.size': 14})

# --- Parameters ---
limits = [(-10, 10), (-10, 10), (-np.pi, np.pi), (-10, 10)]
N = 3200  # Number of training samples
M = 500   # Number of basis functions
lambda_reg = 1e-3

# --- Dataset generation ---
X, Y = X_Y_dataset_step(N, limits)

# --- Basis selection & kernel setup ---
indices = np.random.choice(N, M, replace=False)
X_basis = X[indices]
sigma = np.std(X, axis=0)
K_NM = build_kernel_matrix(X, X_basis, sigma)
K_MM = build_kernel_matrix(X_basis, X_basis, sigma)

# --- Model training ---
alpha = find_alpha(K_NM, K_MM, Y, lambda_reg)
Y_pred = K_NM @ alpha

# --- Scatter plot: true vs predicted ---
# labels = [r"$\Delta x$ (m)", r"$\Delta x'$ (m/s)", r"$\Delta \theta$ (rad)", r"$\Delta \theta'$ (rad/s)"]
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.08, hspace=0.35, wspace=0.35)

# for i, ax in enumerate(axes.flat):
#     ax.scatter(Y[:, i], Y_pred[:, i], alpha=0.5)
#     data_min = min(Y[:, i].min(), Y_pred[:, i].min()) * 1.1
#     data_max = max(Y[:, i].max(), Y_pred[:, i].max()) * 1.1
#     ax.plot([data_min, data_max], [data_min, data_max], 'k--')
#     ax.set_xlabel(f'CartPole.py model: {labels[i]}')
#     ax.set_ylabel(f'Non-Linear model: {labels[i]}')

# label_subplots(fig)
# plt.show(block=False)

# --- Contour plots over (x, x_dot) ---
x_vals = np.linspace(-5, 5, 100)
x_dot_vals = np.linspace(-10, 10, 100)
X, X_dot = np.meshgrid(x_vals, x_dot_vals)

zeros = np.zeros_like(X)
dx, dx_dot, dtheta, dtheta_dot = [zeros.copy() for _ in range(4)]
dx_nl_model, dx_dot_nl_model, dtheta_nl_model, dtheta_dot_nl_model = [zeros.copy() for _ in range(4)]

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X0 = np.array([X[i, j], X_dot[i, j], 0, 0])
        _, X_t = simulate_rollout(X0, F=0.0, steps=1)
        dX = X_t[-1] - X0

        dx[i, j], dx_dot[i, j] = dX[0], dX[1]
        dtheta[i, j], dtheta_dot[i, j] = remap_angle(dX[2]), dX[3]

        K = build_kernel_matrix(X0.reshape(1, -1), X_basis, sigma)
        dX_model = (K @ alpha).flatten()
        dx_nl_model[i, j], dx_dot_nl_model[i, j] = dX_model[0], dX_model[1]
        dtheta_nl_model[i, j], dtheta_dot_nl_model[i, j] = remap_angle(dX_model[2]), dX_model[3]

# --- Plot contour comparisons ---
titles = [r"$\Delta x$", r"$\Delta x'$", r"$\Delta \Theta$", r"$\Delta \Theta'$"]
delta_states = [dx, dx_dot, dtheta, dtheta_dot]
delta_states_nl = [dx_nl_model, dx_dot_nl_model, dtheta_nl_model, dtheta_dot_nl_model]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.93, top=0.95, bottom=0.08, hspace=0.35, wspace=0.45)

for i, ax in enumerate(fig.axes):
    cs = ax.contourf(X, X_dot, delta_states_nl[i], levels=50, cmap='magma')
    ax.contour(X, X_dot, delta_states_nl[i], levels=10, colors='black', linewidths=0.5)
    ax.contour(X, X_dot, delta_states[i], levels=10, colors='white', linewidths=0.5)
    fig.colorbar(cs, ax=ax)
    ax.set(title=titles[i], xlabel="x (m)", ylabel="x' (m/s)")

plt.show(block=False)

# --- Contour plots over (theta, theta_dot) ---
theta_vals = np.linspace(-np.pi, np.pi, 100)
theta_dot_vals = np.linspace(-15, 15, 100)
Theta, Theta_dot = np.meshgrid(theta_vals, theta_dot_vals)

zeros = np.zeros_like(Theta)
dx, dx_dot, dtheta, dtheta_dot = [zeros.copy() for _ in range(4)]
dx_nl_model, dx_dot_nl_model, dtheta_nl_model, dtheta_dot_nl_model = [zeros.copy() for _ in range(4)]

for i in range(Theta.shape[0]):
    for j in range(Theta.shape[1]):
        X0 = np.array([0, 0, Theta[i, j], Theta_dot[i, j]])
        _, X_t = simulate_rollout(X0, F=0.0, steps=1)
        dX = X_t[-1] - X0

        dx[i, j], dx_dot[i, j] = dX[0], dX[1]
        dtheta[i, j], dtheta_dot[i, j] = remap_angle(dX[2]), dX[3]

        K = build_kernel_matrix(X0.reshape(1, -1), X_basis, sigma)
        dX_model = (K @ alpha).flatten()
        dx_nl_model[i, j], dx_dot_nl_model[i, j] = dX_model[0], dX_model[1]
        dtheta_nl_model[i, j], dtheta_dot_nl_model[i, j] = remap_angle(dX_model[2]), dX_model[3]

# --- Plot contour comparisons ---
delta_states = [dx, dx_dot, dtheta, dtheta_dot]
delta_states_nl = [dx_nl_model, dx_dot_nl_model, dtheta_nl_model, dtheta_dot_nl_model]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.93, top=0.95, bottom=0.08, hspace=0.4, wspace=0.45)

for i, ax in enumerate(fig.axes):
    cs = ax.contourf(Theta, Theta_dot, delta_states_nl[i], levels=50, cmap='magma')
    ax.contour(Theta, Theta_dot, delta_states_nl[i], levels=10, colors='black', linewidths=0.5)
    ax.contour(Theta, Theta_dot, delta_states[i], levels=10, colors='white', linewidths=0.5)
    fig.colorbar(cs, ax=ax)
    ax.set(title=titles[i], xlabel=r"$\Theta$ (rad)", ylabel=r"$\Theta'$ (rad/s)")

for i, ax in enumerate(fig.axes[:4]):
        ax.text(0.0, 1.0, f"({chr(ord('a') + i)})",
                transform=ax.transAxes + ScaledTranslation(-60 / 72, +7 / 72, fig.dpi_scale_trans), fontsize=16)

plt.show()
