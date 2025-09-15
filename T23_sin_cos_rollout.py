import numpy as np
import matplotlib.pyplot as plt
from T20_utils import X_Y_dataset_step_jnp, build_kernel_matrix_jnp, find_alpha_jnp, simulate_rollout, nonlinear_simulate_rollout_jnp, plot_rollout, optimise_hyperparams


N = 3200
M = 500
limits = [(-10, 10), (-10, 10), (-np.pi, np.pi), (-15, 15)]

X_train, Y_train = X_Y_dataset_step_jnp(N, limits)
X_basis = X_train[np.random.choice(N, M, replace=False)]

hyperparams = optimise_hyperparams(X_train,Y_train,X_basis,limits=limits, no_init_hparams=50, plot_results=True)
sigma = hyperparams[:-1]
lambda_reg = hyperparams[-1]

print('optimised sigma', sigma)
print('optimised lambda', lambda_reg)

# Build model
K_NM = build_kernel_matrix_jnp(X_train, X_basis, sigma)
K_MM = build_kernel_matrix_jnp(X_basis, X_basis, sigma)
alpha = find_alpha_jnp(K_NM, K_MM, Y_train, lambda_reg)

# Simulation rollout comparison
fig_rollout = None
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
tsteps = 500
x_dot_values = np.linspace(-10, 10, len(colors))

for i, x_dot in enumerate(x_dot_values):
    initial_state = np.array([0.0, x_dot, np.pi, 0.0])
    label = f"$x$' = {x_dot:.2f} m/s"

    # True rollout
    time, true_traj = simulate_rollout(initial_state, steps=tsteps)

    # Predicted rollout using nonlinear model
    predicted_traj = nonlinear_simulate_rollout_jnp(initial_state, alpha, X_basis, sigma, steps=tsteps)

    # Plotting
    fig_rollout = plot_rollout(time, true_traj, fig=fig_rollout, label=label, color=colors[i])
    fig_rollout = plot_rollout(time, predicted_traj, fig=fig_rollout, style=":", color=colors[i], label=None)

plt.show(block=False)

# Varying initial pole angle - theta
fig_rollout = None  # reset figure
colors = ['blue', 'red', 'green', 'purple', 'orange']

for i, theta_deg in enumerate([0, 45, 90, 135, 180]):
    initial_state = np.array([0.0, 0.0, theta_deg / 180 * np.pi, 0.0])
    label = rf"$\theta_0$ = {theta_deg}$^\circ$"

    # True rollout
    time, true_traj = simulate_rollout(initial_state, steps=tsteps)

    # Predicted rollout using nonlinear model
    predicted_traj = nonlinear_simulate_rollout_jnp(initial_state, alpha, X_basis, sigma, steps=tsteps)

    # Plotting
    fig_rollout = plot_rollout(time, true_traj, fig=fig_rollout, label=label, color=colors[i])
    fig_rollout = plot_rollout(time, predicted_traj, fig=fig_rollout, style=":", color=colors[i], label=None)

plt.show()

