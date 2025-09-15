import numpy as np
import matplotlib.pyplot as plt
from T20_utils import X_Y_dataset_step, simulate_rollout, build_kernel_matrix, find_alpha, plot_rollout, nonlinear_simulate_rollout


# Plot settings
plt.rcParams.update({'font.size': 14})

# Parameters
limits = [(-10, 10), (-10, 10), (-np.pi, np.pi), (-10, 10)]
N = 3200            # Number of training samples
M = 500             # Number of basis functions
lambda_reg = 1e-3   # Regularization

# Generate dataset
X, Y = X_Y_dataset_step(N, limits)

# Select M random basis centers
indices = np.random.choice(N, M, replace=False)
X_basis = X[indices]

# Kernel bandwidth estimate
sigma = np.std(X, axis=0)

# Kernel matrices
K_NM = build_kernel_matrix(X, X_basis, sigma)
K_MM = build_kernel_matrix(X_basis, X_basis, sigma)

# Solve for alpha
alpha = find_alpha(K_NM, K_MM, Y, lambda_reg)

# Simulation rollout comparison
fig_rollout = None
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
tsteps = 300
x_dot_values = np.linspace(-10, 10, len(colors))

for i, x_dot in enumerate(x_dot_values):
    initial_state = np.array([0.0, x_dot, np.pi, 0.0])
    label = f"$x$' = {x_dot:.2f} m/s"

    # True rollout
    time, true_traj = simulate_rollout(initial_state, steps=tsteps)

    # Predicted rollout using nonlinear model
    predicted_traj = nonlinear_simulate_rollout(initial_state, alpha, X_basis, sigma, steps=tsteps)

    # Plotting
    fig_rollout = plot_rollout(time, true_traj, fig=fig_rollout, label=label, color=colors[i])
    fig_rollout = plot_rollout(time, predicted_traj, fig=fig_rollout, style=":", color=colors[i], label=None)

plt.show(block=False)

#%% Varying initial pole angle - theta
fig_rollout = None  # reset figure
colors = ['blue', 'red', 'green', 'purple', 'orange']  # Different color for each angle

for i, theta_deg in enumerate([0, 45, 90, 135, 180]):
    initial_state = np.array([0.0, 0.0, theta_deg / 180 * np.pi, 0.0])
    label = rf"$\theta_0$ = {theta_deg}$^\circ$"

    # True rollout
    time, true_traj = simulate_rollout(initial_state, steps=tsteps)

    # Predicted rollout using nonlinear model
    predicted_traj = nonlinear_simulate_rollout(initial_state, alpha, X_basis, sigma, steps=tsteps)

    # Plotting
    fig_rollout = plot_rollout(time, true_traj, fig=fig_rollout, label=label, color=colors[i])
    fig_rollout = plot_rollout(time, predicted_traj, fig=fig_rollout, style=":", color=colors[i], label=None)

plt.show()

