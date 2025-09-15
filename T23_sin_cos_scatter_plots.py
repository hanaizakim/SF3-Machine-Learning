import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from T20_utils import X_Y_dataset_step_jnp, objective_with_tracking, grad_wrapper, build_kernel_matrix_jnp, find_alpha_jnp, label_subplots


N = 1000
M = 50
limits = [(-10, 10), (-10, 10), (-np.pi, np.pi), (-15, 15)]

# Generate dataset
X_train, Y_train = X_Y_dataset_step_jnp(N, limits)
indices = np.random.choice(N, M, replace=False)
X_basis = X_train[indices]

# Initialize hyperparameters based on data statistics
est_sigma = jnp.array([
    *jnp.std(X_train[:, :2], axis=0),            # σ1, σ2: position, velocity
    jnp.std(jnp.sin(X_train[:, 2])),             # σ3: sin(angle)
    jnp.std(jnp.cos(X_train[:, 2])),             # σ4: cos(angle)
    jnp.std(X_train[:, 3])                       # σ5: angular velocity
])
est_lambda = 1e-3
init_hyperparams = jnp.concatenate((est_sigma, jnp.array([est_lambda])))

# Set bounds for each hyperparameter (sigma1-5 and lambda)
bounds = [(0.01, 20.0), (0.01, 20.0), (0.01, 2), (0.01, 2), (0.01, 20), (1e-06, 0.1)]

# Create objective and gradient functions with fixed data
objective_fn, trajectory = objective_with_tracking(X_train, X_basis, Y_train)
grad_fn = grad_wrapper(X_train, X_basis, Y_train)

result = minimize(objective_fn, init_hyperparams, jac=grad_fn,
                 bounds=bounds, method="L-BFGS-B",
                 options={'gtol': 1e-6})

opt_params = result.x
opt_mse = result.fun
opt_sigma = opt_params[:-1]  # Now 5 dimensions for sigma
opt_lambda = opt_params[-1]  # Lambda is now the 6th parameter

print("\nFinal Results:")
print(f"Optimized hyperparameters: {opt_params}")
print(f"Final MSE: {opt_mse}")
print(f"Optimized sigma: {opt_sigma}")
print(f"Optimized lambda: {opt_lambda}")

# Build Model with estimated hyperparameters
K_NM_est = build_kernel_matrix_jnp(X_train, X_basis, est_sigma)
K_MM_est = build_kernel_matrix_jnp(X_basis, X_basis, est_sigma)
alpha_est = find_alpha_jnp(K_NM_est, K_MM_est, Y_train, est_lambda)

# Build model with ptimised hyperparameters
K_NM_opt = build_kernel_matrix_jnp(X_train, X_basis, opt_sigma)
K_MM_opt = build_kernel_matrix_jnp(X_basis, X_basis, opt_sigma)
alpha_opt = find_alpha_jnp(K_NM_opt, K_MM_opt, Y_train, opt_lambda)

#########

# Plot optimisation trajectory
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 4))

# Plot MSE over iterations
ax1.semilogy(trajectory['iterations'], trajectory['mse'], 'b-', label='MSE')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('MSE')
ax1.set_title('Optimisation Progress')
ax1.grid(True)
ax1.legend()

# Plot hyperparameters over iterations
hyperparams = np.array(trajectory['hyperparams'])
labels = [r'$\sigma_x$', r"$\sigma_{x'}$", r'$\sigma_{sin_\Theta}$', r'$\sigma_{cos_\Theta}$', r"$\sigma_{\Theta'}$", r'$\lambda$']
for i,label in enumerate(labels):
    ax2.semilogy(trajectory['iterations'], hyperparams[:, i],  label=labels[i])
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Hyperparameter Value')
ax2.set_title('Hyperparameter Evolution')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show(block=False)

##############################################################################################
# Plot predicted dx vs true dx for training data set - estimated and optimised hyperparameters
##############################################################################################

# Predictions on training set
Y_pred_train_est = build_kernel_matrix_jnp(X_train, X_basis, est_sigma) @ alpha_est
Y_pred_train_opt = build_kernel_matrix_jnp(X_train, X_basis, opt_sigma) @ alpha_opt

# Visualisation
labels = [r"$\Delta x$ (m)", r"$\Delta x$' (m/s)", r"$\Delta \theta$ (rad)", r"$\Delta \theta'$ (rad/s)"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.08, hspace=0.35, wspace=0.35)
for i, ax in enumerate(axes.flat):
    # Plot training data
    ax.scatter(Y_train[:, i], Y_pred_train_est[:, i], alpha=0.5, label='Estimated param', color='blue')

    # Plot test data
    ax.scatter(Y_train[:, i], Y_pred_train_opt[:, i], alpha=0.5, label='Optimised param', color='red')

    # Plot diagonal line
    data_min = min(Y_train[:, i].min(), Y_pred_train_opt[:, i].min(), Y_train[:, i].min(), Y_pred_train_opt[:, i].min()) * 1.1
    data_max = max(Y_train[:, i].max(), Y_pred_train_opt[:, i].max(), Y_train[:, i].max(), Y_pred_train_opt[:, i].max()) * 1.1
    ax.plot([data_min, data_max], [data_min, data_max], 'k--')

    ax.set_xlabel(f'CartPole.py model: {labels[i]}')
    ax.set_ylabel(f'Non-Linear model: {labels[i]}')
    ax.grid(True)
    if i == 0:  # Add legend only to first subplot
        ax.legend()

label_subplots(fig)
plt.show(block=False)

#######################################################################################
# Plot predicted dx vs true dx with optimized hyperparameters: training and test data
#######################################################################################

# Generate test data
N_test = 1000
M_test = 500
limits = [(-10, 10), (-10, 10), (-np.pi, np.pi), (-15, 15)]
X_test, Y_test = X_Y_dataset_step_jnp(N_test, limits)

# Predictions on test set
Y_pred_test_est = build_kernel_matrix_jnp(X_test, X_basis, est_sigma) @ alpha_est
Y_pred_test_opt = build_kernel_matrix_jnp(X_test, X_basis, opt_sigma) @ alpha_opt

# Visualization
labels = [r"$\Delta x$ (m)", r"$\Delta x$' (m/s)", r"$\Delta \theta$ (rad)", r"$\Delta \theta'$ (rad/s)"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.08, hspace=0.35, wspace=0.35)

for i, ax in enumerate(axes.flat):
    # Plot training data
    ax.scatter(Y_train[:, i], Y_pred_train_opt[:, i], alpha=0.5, label='Training Data Set', color='blue')

    # Plot test data
    ax.scatter(Y_test[:, i], Y_pred_test_opt[:, i], alpha=0.5, label='Test Data Set', color='red')

    # Plot diagonal line
    data_min = min(Y_test[:, i].min(), Y_pred_test_opt[:, i].min(), Y_test[:, i].min(), Y_pred_test_opt[:, i].min()) * 1.1
    data_max = max(Y_test[:, i].max(), Y_pred_test_opt[:, i].max(), Y_test[:, i].max(), Y_pred_test_opt[:, i].max()) * 1.1
    ax.plot([data_min, data_max], [data_min, data_max], 'k--')

    ax.set_xlabel(f'CartPole.py model: {labels[i]}')
    ax.set_ylabel(f'Non-Linear model: {labels[i]}')
    ax.grid(True)
    if i == 0:  # Add legend only to first subplot
        ax.legend()

label_subplots(fig)
plt.show()
