# Task 1.2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from T00_CartPole import remap_angle
from T11_dynamic_simulation import simulate_rollout, add_legend

plt.rcParams.update({'font.size': 14})

F = 0.0  # No force applied to cart

#%% Line plots - x and x_dot values
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.2, hspace=0.35, wspace=0.3)

xs = np.linspace(-5, 5, 100)
x_dots = np.linspace(-10, 10, 7)
for x_dot in x_dots:
    dxs, dx_dots, dthetas, dtheta_dots = [], [], [], []
    for x in xs:
        X0 = [x, x_dot, 0.0, 0.0]
        _, X_t = simulate_rollout(X0, F, steps=1)
        dX = X_t[-1] - X0
        dxs.append(dX[0])
        dx_dots.append(dX[1])
        dthetas.append(remap_angle(dX[2]))
        dtheta_dots.append(dX[3])

    delta_states = [dxs, dx_dots, dthetas, dtheta_dots]
    for i,ax in enumerate(fig.axes):
        ax.plot(xs,delta_states[i], label=f"$x'$ = {x_dot:.1f} m/s")

ylabels = ["$\Delta x$", "$\Delta x'$", "$\Delta \\theta$", "$\Delta \\theta'$"]
for i, ax in enumerate(fig.axes):
    ax.set_xlim(xs[0], xs[-1])
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel(ylabels[i])
    ax.grid(True)
    ax.text(0.0, 1.0, f"({chr(ord('a') + i)})",
            transform=(ax.transAxes + ScaledTranslation(-35 / 72, 7 / 72, fig.dpi_scale_trans)), fontsize=16)

add_legend(fig)
plt.show(block=False)

#%% Line plots - theta and theta_dot values
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.2, hspace=0.35, wspace=0.35)

thetas = np.linspace(-np.pi, np.pi, 100)
theta_dots = np.linspace(0, 15, 7)
for theta_dot in theta_dots:
    dxs, dx_dots, dthetas, dtheta_dots = [], [], [], []
    for theta in thetas:
        X0 = [0, 0.0, theta, theta_dot]
        _, X_t = simulate_rollout(X0, F, 1)
        dX=(X_t[-1] - X0)
        dxs.append(dX[0])
        dx_dots.append(dX[1])
        dthetas.append(remap_angle(dX[2]))
        dtheta_dots.append(dX[3])

    delta_states = [dxs, dx_dots, dthetas, dtheta_dots]
    for i,ax in enumerate(fig.axes):
        ax.plot(thetas, delta_states[i], label=f"$\Theta$'={theta_dot:.0f} rad/s")

ylabels = ["$\Delta x$", "$\Delta x'$", "$\Delta \\theta$", "$\Delta \\theta'$"]
for i, ax in enumerate(fig.axes):
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xlabel("$\Theta$ (rad)")
    ax.set_ylabel(ylabels[i])
    ax.grid(True)
    ax.text(0.0, 1.0, f"({chr(ord('a') + i)})",
            transform=(ax.transAxes + ScaledTranslation(-35 / 72, 7 / 72, fig.dpi_scale_trans)), fontsize=16)

add_legend(fig)
plt.show(block=False)

#%% Contours of delta states - x and x_dot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.93, top=0.95, bottom=0.08, hspace=0.35, wspace=0.45)
x_vals = np.linspace(-5, 5, 100)
x_dot_vals = np.linspace(-10, 10, 100)
X, X_dot = np.meshgrid(x_vals, x_dot_vals)
zeros = np.zeros_like(X)
dx, dx_dot, dtheta, dtheta_dot = zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X0 = np.array([X[i, j], X_dot[i, j], 0, 0])
        _, X_t = simulate_rollout(X0, F, 1)
        dX = X_t[-1] - X0
        dx[i, j] = dX[0]
        dx_dot[i, j] = dX[1]
        dtheta[i, j] = remap_angle(dX[2])
        dtheta_dot[i, j] = dX[3]
titles = [r"$\Delta x$", r"$\Delta x'$", r"$\Delta \Theta$", r"$\Delta \Theta'$"]
delta_states = [dx, dx_dot, dtheta, dtheta_dot]
for i, ax in enumerate(fig.axes):
    cs = ax.contourf(X, X_dot, delta_states[i], levels=50, cmap='magma')
    ax.contour(X, X_dot, delta_states[i], levels=10, colors='white', linewidths=0.5)
    fig.colorbar(cs, ax=ax)
    ax.set(title=titles[i], xlabel="x (m)", ylabel="x' (m/s)")
    ax.text(0, 1, f"({chr(97 + i)})", transform=ax.transAxes + ScaledTranslation(-55/72, 7/72, fig.dpi_scale_trans), fontsize=16)
plt.show(block=False)

#%% Contours of delta states - theta and theta_dot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.93, top=0.95, bottom=0.08, hspace=0.4, wspace=0.45)
theta_vals = np.linspace(-np.pi, np.pi, 100)
theta_dot_vals = np.linspace(-15, 15, 100)
Theta, Theta_dot = np.meshgrid(theta_vals, theta_dot_vals)
zeros = np.zeros_like(X)
dx, dx_dot, dtheta, dtheta_dot = zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X0 = np.array([0,0,Theta[i, j], Theta_dot[i, j]])
        _, X_t = simulate_rollout(X0, F, 1)
        dX = X_t[-1] - X0
        dx[i, j] = dX[0]
        dx_dot[i, j] = dX[1]
        dtheta[i, j] = remap_angle(dX[2])
        dtheta_dot[i, j] = dX[3]
titles = [r"$\Delta x$", r"$\Delta x'$", r"$\Delta \Theta$", r"$\Delta \Theta'$"]
delta_states = [dx, dx_dot, dtheta, dtheta_dot]
for i, ax in enumerate(fig.axes):
    cs = ax.contourf(Theta, Theta_dot, delta_states[i], levels=50, cmap='magma')
    ax.contour(Theta, Theta_dot, delta_states[i], levels=10, colors='white', linewidths=0.5)
    fig.colorbar(cs, ax=ax)
    ax.set(title=titles[i], xlabel=r"$\Theta$ (rad)", ylabel=r"$\Theta'$ (rad/s)")
    ax.text(0, 1, f"({chr(97 + i)})", transform=ax.transAxes + ScaledTranslation(-55/72, 7/72, fig.dpi_scale_trans), fontsize=16)
plt.show()
