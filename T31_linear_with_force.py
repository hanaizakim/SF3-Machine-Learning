
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from T30_utils import X_Y_dataset_step_with_F, least_squares_solution, simulate_rollout


plt.rcParams.update({'font.size': 14})
N = 500 # Number of random initial states
n_inputs= 5  # Number of states: x, x', theta, theta', F

force_limits = (-20,20)

#Scatter plot
#===============
# Any initial state
limits = [(-10,10),(-10,10),(-np.pi, np.pi),(-15,15),force_limits]
X,Y= X_Y_dataset_step_with_F(N,limits)
C_T = least_squares_solution(X,Y)
Y_pred = X @ C_T  #Linear model
X_pred = X  + Y_pred #Linear model

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.08, hspace=0.35, wspace=0.35)
for i,ax in enumerate(fig.axes):
    ax.scatter(Y[:, i], Y_pred[:, i], alpha=0.6,label='Linear model')

# Initial state limited to small angles
limits = [(-10,10),(-10,10),(-np.pi/12, np.pi/12),(-3,3),force_limits]
X,Y= X_Y_dataset_step_with_F(N,limits)
C_T_2 = least_squares_solution(X,Y)
Y_pred = X @ C_T_2  #Linear model
X_pred = X  + Y_pred #Linear model

for i,ax in enumerate(fig.axes):
    ax.scatter(Y[:, i], Y_pred[:, i], alpha=0.6,label='Small angle lin model')

# Plot ideal line y=x, label axes and add legend
state_labels = ['$\Delta x$ (m)', "$\Delta x'$ (m/s)", '$\Delta \Theta$ (rad)', "$\Delta \Theta'$ (rad/s)"]
for i,ax in enumerate(fig.axes):
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", color='black')
    ax.set_xlabel(f'Cartpole.py model: {state_labels[i]}')
    ax.set_ylabel(f'Linear model: {state_labels[i]}')
    ax.text(0.0, 1.0, "("+chr(ord('a')+i)+")", transform=(ax.transAxes + ScaledTranslation(-55/72, +7/72, fig.dpi_scale_trans)), fontsize=16)
    ax.legend()

# Contour plots - theta and theta_dot
#======================================
thetas = np.linspace(-np.pi, np.pi, 100)
theta_dots = np.linspace(-15, 15, 100)
Theta, Theta_dot = np.meshgrid(thetas, theta_dots)

X = np.zeros((len(Theta.ravel()), n_inputs))
X[:, 2] = Theta.ravel()
X[:, 3] = Theta_dot.ravel()
Y_pred = X @ C_T  # Predicted deltas using linear model
Y = np.zeros((len(Theta.ravel()), n_inputs))
for i, X0 in enumerate(X):
    _, X_t = simulate_rollout(X0[:-1], F=X0[-1], steps=1)
    Y[i] = np.append(X_t[-1],X0[-1]) - X0

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.08, hspace=0.35, wspace=0.35)
titles = [r"$\Delta x$", r"$\Delta x'$", r"$\Delta \Theta$", r"$\Delta \Theta'$"]
for i, ax in enumerate(fig.axes):
    Z = Y_pred[:, i].reshape(Theta.shape)
    cs = ax.contourf(Theta, Theta_dot, Z, levels=50, cmap='magma')
    ax.contour(Theta, Theta_dot, Z, levels=10, colors='black', linewidths=0.5)
    Z = Y[:, i].reshape(Theta.shape)
    ax.contour(Theta, Theta_dot, Z, levels=10, colors='white', linewidths=0.5)
    fig.colorbar(cs, ax=ax)
    ax.set(title=titles[i], xlabel=r"$\Theta$ (rad)", ylabel=r"$\Theta'$ (rad/s)")
    ax.text(0, 1, f"({chr(97 + i)})", transform=ax.transAxes + ScaledTranslation(-55/72, 7/72, fig.dpi_scale_trans), fontsize=16)
plt.show(block=False)

# Contour plots - x and x_dot
#==============================
xs = np.linspace(-10, 10, 100)
x_dots = np.linspace(-10, 10, 100)
X, X_dot = np.meshgrid(xs, x_dots)

States = np.zeros((len(X.ravel()), n_inputs))
States[:, 0] = X.ravel()
States[:, 1] = X_dot.ravel()
Y_pred = States @ C_T  # Predicted deltas using linear model
Y = np.zeros((len(X.ravel()), n_inputs))
for i, X0 in enumerate(States):
    _, X_t = simulate_rollout(X0[:-1], F=X0[-1], steps=1)
    Y[i] = np.append(X_t[-1],X0[-1]) - X0

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.08, hspace=0.35, wspace=0.4)
titles = [r"$\Delta x$", r"$\Delta x'$", r"$\Delta \Theta$", r"$\Delta \Theta'$"]
for i, ax in enumerate(fig.axes):
    Z = Y_pred[:, i].reshape(X.shape)
    cs = ax.contourf(X, X_dot, Z, levels=50, cmap='magma')
    ax.contour(X, X_dot, Z, levels=10, colors='black', linewidths=0.5)
    Z = Y[:, i].reshape(X.shape)
    ax.contour(X, X_dot, Z, levels=10, colors='white', linewidths=0.5)
    fig.colorbar(cs, ax=ax)
    ax.set(title=titles[i], xlabel=r"$x$ (m)", ylabel=r"$x'$ (m/s)")
    ax.text(0, 1, f"({chr(97 + i)})", transform=ax.transAxes + ScaledTranslation(-55/72, 7/72, fig.dpi_scale_trans), fontsize=16)

plt.show()


