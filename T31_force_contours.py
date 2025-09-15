import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from T00_CartPole import remap_angle, CartPole
from T30_utils import simulate_rollout

plt.rcParams.update({'font.size': 14})

# contour plots of delta states and their relationship to F

# Contours of delta states - F and x
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.93, top=0.95, bottom=0.08, hspace=0.35, wspace=0.45)
F_vals = np.linspace(-40,40, 100)
x_vals = np.linspace(-10, 10, 100)
F, X = np.meshgrid(F_vals, x_vals)
zeros = np.zeros_like(X)
dx, dx_dot, dtheta, dtheta_dot = zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()
for i in range(F.shape[0]):
    for j in range(F.shape[1]):
        X0 = np.array([X[i, j], 0, 0, 0])  # Only state variables
        _, X_t = simulate_rollout(X0, F[i, j], 1)  # Pass force as separate parameter
        dX = X_t[-1] - X0
        dx[i, j] = dX[0]
        dx_dot[i, j] = dX[1]
        dtheta[i, j] = remap_angle(dX[2])
        dtheta_dot[i, j] = dX[3]
titles = [r"$\Delta x$", r"$\Delta x'$", r"$\Delta \Theta$", r"$\Delta \Theta'$"]
delta_states = [dx, dx_dot, dtheta, dtheta_dot]
for i, ax in enumerate(fig.axes):
    cs = ax.contourf(F, X, delta_states[i], levels=50, cmap='magma')
    ax.contour(F, X, delta_states[i], levels=10, colors='white', linewidths=0.5)
    fig.colorbar(cs, ax=ax)
    ax.set(title=titles[i], xlabel="F (N)", ylabel="x (m)")  # Updated labels
    ax.text(0, 1, f"({chr(97 + i)})", transform=ax.transAxes + ScaledTranslation(-55/72, 7/72, fig.dpi_scale_trans), fontsize=16)
plt.show(block=False)

# Contours of delta states - F and x_dot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.93, top=0.95, bottom=0.08, hspace=0.35, wspace=0.45)
F_vals = np.linspace(-40,40, 100)
x_dot_vals = np.linspace(-10, 10, 100)
F, X_dot = np.meshgrid(F_vals, x_dot_vals)
zeros = np.zeros_like(X)
dx, dx_dot, dtheta, dtheta_dot = zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()
for i in range(F.shape[0]):
    for j in range(F.shape[1]):
        X0 = np.array([0, X_dot[i, j], 0, 0])  # Only state variables
        _, X_t = simulate_rollout(X0, F[i, j], 1)  # Pass force as separate parameter
        dX = X_t[-1] - X0
        dx[i, j] = dX[0]
        dx_dot[i, j] = dX[1]
        dtheta[i, j] = remap_angle(dX[2])
        dtheta_dot[i, j] = dX[3]
titles = [r"$\Delta x$", r"$\Delta x'$", r"$\Delta \Theta$", r"$\Delta \Theta'$"]
delta_states = [dx, dx_dot, dtheta, dtheta_dot]
for i, ax in enumerate(fig.axes):
    cs = ax.contourf(F, X, delta_states[i], levels=50, cmap='magma')
    ax.contour(F, X, delta_states[i], levels=10, colors='white', linewidths=0.5)
    fig.colorbar(cs, ax=ax)
    ax.set(title=titles[i], xlabel="F (N)", ylabel="x' (m/s)")  # Updated labels
    ax.text(0, 1, f"({chr(97 + i)})", transform=ax.transAxes + ScaledTranslation(-55/72, 7/72, fig.dpi_scale_trans), fontsize=16)
plt.show(block=False)

# Contours of delta states - F and theta
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.93, top=0.95, bottom=0.08, hspace=0.35, wspace=0.45)
F_vals = np.linspace(-40,40, 100)
theta_vals = np.linspace(-np.pi, np.pi, 100)
F, theta = np.meshgrid(F_vals, theta_vals)
zeros = np.zeros_like(theta)
dx, dx_dot, dtheta, dtheta_dot = zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()
for i in range(F.shape[0]):
    for j in range(F.shape[1]):
        X0 = np.array([0, 0, theta[i, j], 0])  # Only state variables
        _, X_t = simulate_rollout(X0, F[i, j], 1)  # Pass force as separate parameter
        dX = X_t[-1] - X0
        dx[i, j] = dX[0]
        dx_dot[i, j] = dX[1]
        dtheta[i, j] = remap_angle(dX[2])
        dtheta_dot[i, j] = dX[3]
titles = [r"$\Delta x$", r"$\Delta x'$", r"$\Delta \Theta$", r"$\Delta \Theta'$"]
delta_states = [dx, dx_dot, dtheta, dtheta_dot]
for i, ax in enumerate(fig.axes):
    cs = ax.contourf(F, theta, delta_states[i], levels=50, cmap='magma')
    ax.contour(F, theta, delta_states[i], levels=10, colors='white', linewidths=0.5)
    fig.colorbar(cs, ax=ax)
    ax.set(title=titles[i], xlabel="F (N)", ylabel="$\Theta$ (rad)")
    ax.text(0, 1, f"({chr(97 + i)})", transform=ax.transAxes + ScaledTranslation(-55/72, 7/72, fig.dpi_scale_trans), fontsize=16)
plt.show(block=False)

# Contours of delta states - F and theta_dot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.93, top=0.95, bottom=0.08, hspace=0.35, wspace=0.45)
F_vals = np.linspace(-40,40, 100)
theta_dot_vals = np.linspace(-15, 15, 100)
F, theta_dot = np.meshgrid(F_vals, theta_dot_vals)
zeros = np.zeros_like(X)
dx, dx_dot, dtheta, dtheta_dot = zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()
for i in range(F.shape[0]):
    for j in range(F.shape[1]):
        X0 = np.array([0, 0, 0, theta_dot[i, j]])  # Only state variables
        _, X_t = simulate_rollout(X0, F[i, j], 1)  # Pass force as separate parameter
        dX = X_t[-1] - X0
        dx[i, j] = dX[0]
        dx_dot[i, j] = dX[1]
        dtheta[i, j] = remap_angle(dX[2])
        dtheta_dot[i, j] = dX[3]
titles = [r"$\Delta x$", r"$\Delta x'$", r"$\Delta \Theta$", r"$\Delta \Theta'$"]
delta_states = [dx, dx_dot, dtheta, dtheta_dot]
for i, ax in enumerate(fig.axes):
    cs = ax.contourf(F, X, delta_states[i], levels=50, cmap='magma')
    ax.contour(F, X, delta_states[i], levels=10, colors='white', linewidths=0.5)
    fig.colorbar(cs, ax=ax)
    ax.set(title=titles[i], xlabel="F (N)", ylabel="$\Theta'$ (rad/s)")  # Updated labels
    ax.text(0, 1, f"({chr(97 + i)})", transform=ax.transAxes + ScaledTranslation(-55/72, 7/72, fig.dpi_scale_trans), fontsize=16)
plt.show()

# Additional contour plots - theta vs F and theta_dot vs F
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.93, top=0.95, bottom=0.08, hspace=0.35, wspace=0.45)

# First plot: theta vs F (delta_theta_dot)
F_vals = np.linspace(-25, 25, 100)
theta_vals = np.linspace(-np.pi, np.pi, 100)
F, theta = np.meshgrid(F_vals, theta_vals)
zeros = np.zeros_like(theta)
dx, dx_dot, dtheta, dtheta_dot = zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()

for i in range(F.shape[0]):
    for j in range(F.shape[1]):
        X0 = np.array([0, 0, theta[i, j], 0])
        _, X_t = simulate_rollout(X0, F[i, j], 1)
        dX = X_t[-1] - X0
        dx[i, j] = dX[0]
        dx_dot[i, j] = dX[1]
        dtheta[i, j] = remap_angle(dX[2])
        dtheta_dot[i, j] = dX[3]

# Plot delta_theta_dot
cs = axes[1,1].contourf(F, theta, dtheta_dot, levels=50, cmap='magma')
axes[1,1].contour(F, theta, dtheta_dot, levels=10, colors='white', linewidths=0.5)
fig.colorbar(cs, ax=axes[0,0])
axes[1,1].set(title=r"$\Delta \Theta'$", xlabel="F (N)", ylabel="$\Theta$ (rad)")
axes[1,1].text(0, 1, "(a)", transform=axes[0,0].transAxes + ScaledTranslation(-55/72, 7/72, fig.dpi_scale_trans), fontsize=16)

# Plot delta_x_dot
cs = axes[0,1].contourf(F, theta, dx_dot, levels=50, cmap='magma')
axes[0,1].contour(F, theta, dx_dot, levels=10, colors='white', linewidths=0.5)
fig.colorbar(cs, ax=axes[0,1])
axes[0,1].set(title=r"$\Delta x'$", xlabel="F (N)", ylabel="$\Theta$ (rad)")
axes[0,1].text(0, 1, "(b)", transform=axes[0,1].transAxes + ScaledTranslation(-55/72, 7/72, fig.dpi_scale_trans), fontsize=16)

# Second set of plots: theta_dot vs F
F_vals = np.linspace(-25, 25, 100)
theta_dot_vals = np.linspace(-15, 15, 100)
F, theta_dot = np.meshgrid(F_vals, theta_dot_vals)
zeros = np.zeros_like(theta_dot)
dx, dx_dot, dtheta, dtheta_dot = zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()

for i in range(F.shape[0]):
    for j in range(F.shape[1]):
        X0 = np.array([0, 0, 0, theta_dot[i, j]])
        _, X_t = simulate_rollout(X0, F[i, j], 1)
        dX = X_t[-1] - X0
        dx[i, j] = dX[0]
        dx_dot[i, j] = dX[1]
        dtheta[i, j] = remap_angle(dX[2])
        dtheta_dot[i, j] = dX[3]

# Plot delta_theta
cs = axes[1,0].contourf(F, theta_dot, dtheta, levels=50, cmap='magma')
axes[1,0].contour(F, theta_dot, dtheta, levels=10, colors='white', linewidths=0.5)
fig.colorbar(cs, ax=axes[1,0])
axes[1,0].set(title=r"$\Delta \Theta$", xlabel="F (N)", ylabel="$\Theta'$ (rad/s)")
axes[1,0].text(0, 1, "(c)", transform=axes[1,0].transAxes + ScaledTranslation(-55/72, 7/72, fig.dpi_scale_trans), fontsize=16)

# Plot delta_x
cs = axes[0,0].contourf(F, theta_dot, dx, levels=50, cmap='magma')
axes[0,0].contour(F, theta_dot, dx, levels=10, colors='white', linewidths=0.5)
fig.colorbar(cs, ax=axes[1,1])
axes[0,0].set(title=r"$\Delta x$", xlabel="F (N)", ylabel="$\Theta'$ (rad/s)")
axes[0,0].text(0, 1, "(d)", transform=axes[1,1].transAxes + ScaledTranslation(-55/72, 7/72, fig.dpi_scale_trans), fontsize=16)

plt.show()


