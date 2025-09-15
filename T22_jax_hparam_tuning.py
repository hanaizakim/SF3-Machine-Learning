import numpy as np
import matplotlib.pyplot as plt
import jax
from functools import partial
from tqdm import tqdm
from scipy.optimize import minimize
from T20_utils import X_Y_dataset_step, objective_fn_params, build_kernel_matrix, find_alpha, label_subplots, add_legend

'''
Key Steps:
----------
1. Generate a training dataset `X, Y` using state transitions from a known simulator.
2. Select a subset of `M` random points as basis centers for kernel computation.
3. Estimate initial hyperparameters:
   - Kernel width (`sigma`) as the standard deviation of input data.
   - Regularization (`lambda`) as a small constant.
4. Fit a baseline model using the estimated hyperparameters.
5. Optimise hyperparameters (`sigma`, `lambda`) using L-BFGS-B and autograd:
   - Objective function: Mean squared error (MSE) between predicted and true dynamics.
   - Gradient computed using JAX.
   - Multiple random initialisations are tried to avoid poor local minima.
6. Evaluate performance:
   - Predict dynamics on both training and independent test datasets.
   - Compute MSE over `n_test_cases` different random test datasets.
7. Visualisation:
   - Predicted vs actual state transitions for training and test data.
   - Comparison of MSE across test cases before and after optimisation
   - Optimisation trajectory: MSE and hyperparameter evolution over iterations.
   - Distribution of final MSE from different initialisations.
'''

plt.rcParams.update({'font.size': 14})
YELLOW = '\033[93m'
WHITE = '\033[0m'

# Parameters
N = 3200   # Number of training samples
M = 50     # Number of basis functions - #Change to 50 for faster execution and better contrast of the 'estimated' vs 'optimised' hyperparameters plots
n_test_cases = 25 # Number of test cases
n_init_hyperparams = 25 # Number of initial hyperparameter sets to try
limits = [(-10, 10), (-10, 10), (-np.pi, np.pi), (-15, 15)]

# Generate training dataset
X, Y = X_Y_dataset_step(N, limits)

# Select M random basis centres from the dataset
indices = np.random.choice(N, M, replace=False)
X_basis = X[indices]

# Estimate initial hyperparameters
sigma = np.std(X, axis=0)
lambda_reg = 1e-3

# Model with estimated (not optimised) hyperparameters to use as a baseline
K_NM = build_kernel_matrix(X, X_basis, sigma)
K_MM = build_kernel_matrix(X_basis, X_basis, sigma)
alpha = find_alpha(K_NM, K_MM, Y, lambda_reg)
Y_pred_approx = K_NM @ alpha

# Optimise hyperparameters
init_hyperparams = np.append(sigma, lambda_reg)
trajectory = {'iterations': [], 'mse': [], 'hyperparams': []}
bounds = [(1e-12, 1e2), (1e-12, 1e2), (1e-12, np.pi), (1e-12, 1e2), (1e-06, 0.1)]
objective_fn = jax.jit(partial(objective_fn_params, X=X, X_basis=X_basis, Y=Y))
grad_fn = jax.grad(objective_fn)

def optimisation_logger(hyperparams):
    mse = float(objective_fn(hyperparams))
    trajectory['iterations'].append(len(trajectory['iterations']))
    trajectory['mse'].append(mse)
    trajectory['hyperparams'].append(np.copy(hyperparams))

result = minimize(objective_fn, init_hyperparams, jac=grad_fn, bounds=bounds,
                method="L-BFGS-B", callback=optimisation_logger, options={'gtol': 1e-6})
opt_params = result.x
opt_mse_from_estimated = result.fun
opt_sigma_from_estimated = opt_params[:4]
opt_lambda_from_estimated = opt_params[4]

print(YELLOW+rf"Optimized hyperparameters starting from estimated values"+WHITE)
print(rf"  - Final MSE: {opt_mse_from_estimated:e}")
print(rf"  - Optimized hyperparameters: sigma={opt_sigma_from_estimated}, lambda={opt_lambda_from_estimated:e}")
print('')

# Explore dependence of initial hyperparameters guess on optimisation results
init_hyperparams_list = np.column_stack((
    np.random.uniform(0.1, 20.0, (n_init_hyperparams, 2)),   # sigma x and sigma x_dot
    np.random.uniform(0.1, np.pi, (n_init_hyperparams, 1)),   # sigma theta
    np.random.uniform(0.1, 20.0, (n_init_hyperparams, 1)),   # sigma theta_dot
    np.power(10, np.random.uniform(-6, -1, n_init_hyperparams))  # lambda values
))
final_mse = np.full(n_init_hyperparams, np.nan)
final_hyperparams = np.full((n_init_hyperparams, len(init_hyperparams)), np.nan)
for i, init_hp in enumerate(tqdm(init_hyperparams_list,desc ="Optimizing hyperparameters", unit=" cases")):
    result = minimize(objective_fn, init_hp, jac=grad_fn, bounds=bounds,
                    method="L-BFGS-B", options={'gtol': 1e-6})
    if result.success:
        final_hyperparams[i] = result.x
        final_mse[i] = result.fun
valid_indices = ~np.isnan(final_mse)
final_hyperparams = final_hyperparams[valid_indices]
final_mse = final_mse[valid_indices]

lowest_mse_index = np.argmin(final_mse)
if final_mse[lowest_mse_index] < opt_mse_from_estimated:
    opt_mse = final_mse[lowest_mse_index]
    opt_sigma = final_hyperparams[lowest_mse_index][:4]
    opt_lambda = final_hyperparams[lowest_mse_index][4]
else:
    opt_mse = opt_mse_from_estimated
    opt_sigma = opt_sigma_from_estimated
    opt_lambda = opt_lambda_from_estimated
print(YELLOW+f"\033[93mOptimized hyperparameters after {n_init_hyperparams} random trials:"+WHITE)
print(rf"  - Final MSE: {opt_mse:e}")
print(rf"  - Optimized hyperparameters: sigma={opt_sigma}, lambda={opt_lambda:e}")
print('')

# Model training data with optimised hyperparameters
K_NM = build_kernel_matrix(X, X_basis, opt_sigma)
K_MM = build_kernel_matrix(X_basis, X_basis, opt_sigma)
alpha_opt = find_alpha(K_NM, K_MM, Y, opt_lambda)
Y_pred_opt = K_NM @ alpha_opt


#Model new test data with optimised hyperparameters (1 case)
X_test, Y_test = X_Y_dataset_step(n_test_cases, limits)
K_NM_test = build_kernel_matrix(X_test, X_basis, opt_sigma)
Y_pred_test = K_NM_test @ alpha_opt

# Model new test data with estimated hyperparameters
mse_test_cases = np.zeros(n_test_cases)
mse_opt_test_cases = np.zeros(n_test_cases)
for i in tqdm(range(n_test_cases), desc="Evaluating test cases", unit=" cases"):
    X_new, Y_new = X_Y_dataset_step(n_test_cases, limits)
    K_NM_new = build_kernel_matrix(X_new, X_basis, sigma)
    Y_new_pred = K_NM_new @ alpha
    mse_test_cases[i] = np.mean((Y_new_pred - Y_new) ** 2)
    K_NM_new = build_kernel_matrix(X_new, X_basis, opt_sigma)
    Y_new_opt = K_NM_new @ alpha_opt
    mse_opt_test_cases[i] = np.mean((Y_new_opt - Y_new) ** 2)

# Plot: = Fit to training data 2x2
#=================================

labels = [r"$\Delta x$ (m)", r"$\Delta x$' (m/s)", r"$\Delta \theta$ (rad)", r"$\Delta \theta'$ (rad/s)"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.15, hspace=0.35, wspace=0.35)

for i, ax in enumerate(axes.flat):
    ax.scatter(Y[:, i], Y_pred_approx[:, i],  color='blue', alpha=0.5, label=r'Training - Estimated $\sigma,\lambda$')
    ax.scatter(Y[:, i], Y_pred_opt[:, i], color='red', alpha=0.5, label=r'Training - Optimised $\sigma,\lambda$')
    ax.scatter(Y_test[:, i], Y_pred_test[:, i], color='yellow', alpha=0.75, label=r'Test - Optimised $\sigma,\lambda$')

    data_min = min(Y[:, i].min(), Y_pred_approx[:, i].min()) * 1.1
    data_max = max(Y[:, i].max(), Y_pred_approx[:, i].max()) * 1.1
    ax.plot([data_min, data_max], [data_min, data_max], 'k--')

    ax.set_xlabel(f'CartPole.py model: {labels[i]}')
    ax.set_ylabel(f'Non-Linear model: {labels[i]}')

add_legend(fig)

label_subplots(fig)
fig.canvas.manager.set_window_title('TRAINING data')
plt.show(block=False)

# Plot: = Fit to training data 1x4
#==================================

labels = [r"$\Delta x$ (m)", r"$\Delta x$' (m/s)", r"$\Delta \theta$ (rad)", r"$\Delta \theta'$ (rad/s)"]

fig, axes = plt.subplots(1, 4, figsize=(12, 4))
fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.3, hspace=0.3, wspace=0.75)

for i, ax in enumerate(axes.flat):
    ax.scatter(Y[:, i], Y_pred_approx[:, i],  color='blue', alpha=0.5, label=r'Training - Estimated $\sigma,\lambda$')
    ax.scatter(Y[:, i], Y_pred_opt[:, i], color='red', alpha=0.5, label=r'Training - Optimised $\sigma,\lambda$')
    ax.scatter(Y_test[:, i], Y_pred_test[:, i], color='yellow', alpha=0.75, label=r'Test - Optimised $\sigma,\lambda$')

    data_min = min(Y[:, i].min(), Y_pred_approx[:, i].min()) * 1.1
    data_max = max(Y[:, i].max(), Y_pred_approx[:, i].max()) * 1.1
    ax.plot([data_min, data_max], [data_min, data_max], 'k--')

    ax.set_xlabel(f'CartPole.py: {labels[i]}')
    ax.set_ylabel(f'Non-Linear: {labels[i]}')

add_legend(fig)

label_subplots(fig)
fig.canvas.manager.set_window_title('TRAINING data')
plt.show(block=False)

# Plot: Fit to test data
#=======================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.08, hspace=0.35, wspace=0.35)

for i, ax in enumerate(axes.flat):
    ax.scatter(Y_test[:, i], Y_pred_test[:, i],  color='red', alpha=0.5, label='Estimated hyperparams')

    data_min = min(Y_test[:, i].min(), Y_pred_test[:, i].min()) * 1.1
    data_max = max(Y_test[:, i].max(), Y_pred_test[:, i].max()) * 1.1
    ax.plot([data_min, data_max], [data_min, data_max], 'k--')

    ax.set_xlabel(f'CartPole.py model: {labels[i]}')
    ax.set_ylabel(f'Non-Linear model: {labels[i]}')
    ax.legend()

label_subplots(fig)
fig.canvas.manager.set_window_title('TEST data')
plt.show(block=False)

# Plot: MSE of test cases
#==============================
fig1, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, n_test_cases + 1), mse_test_cases, 'o', color='blue',label='Estimated hyperparams')
ax.plot(range(1, n_test_cases + 1), mse_opt_test_cases, 'o', color='red', label='Optimized hyperparams')
ax.set_xlabel("Test set")
ax.set_ylabel("Mean Squared Error (MSE)")
ax.grid(True)
ax.legend()

plt.show()