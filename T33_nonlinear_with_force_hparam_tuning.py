import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from scipy.optimize import minimize
from T30_utils import X_Y_dataset_step_with_F, objective_with_tracking_F,grad_wrapper_F,build_kernel_matrix_jnp_F, find_alpha_jnp, optimise_hyperparams
from tqdm import tqdm

# def optimise_hyperparams(X_train,Y_train,X_basis,limits=[(-10, 10), (-10, 10), (-np.pi, np.pi), (-15, 15)], no_init_hparams=50, plot_results=True):

#     N, M = X_train.shape[0], X_basis.shape[0]

#     init_hyperparams_list = []
#     for i in range(no_init_hparams):
#         sigma_xv = np.random.uniform(0.01, 20.0, 2)        # For x, x_dot
#         sigma_stct = np.random.uniform(0.01, 1.0, 2)       # For sin(theta), cos(theta)
#         sigma_tdot = np.random.uniform(0.01, 20.0, 1)      # For theta_dot
#         sigma_F = np.random.uniform(0.01, 20.0, 1)         # For force input
#         lambda_rand = np.random.uniform(1e-06, 0.1)        # Regularization term
#         init_hyperparams = np.concatenate([sigma_xv, sigma_stct, sigma_tdot, sigma_F, [lambda_rand]])
#         init_hyperparams_list.append(init_hyperparams)

#     mse_train_list = []
#     mse_test_list = []
#     output_list = []
#     opt_hyperparams_list = []
#     for init_hyperparams in tqdm(init_hyperparams_list, desc="Optimizing Hyperparameters", unit=" trials"):
#         bounds = [(0.01, 20.0), (0.01, 20.0), (0.01, 1), (0.01, 1), (0.01, 20.0), (1e-06, 0.1), (0.01, 20.0)]

#         # Create objective and gradient functions with fixed data
#         objective_fn, trajectory = objective_with_tracking_F(X_train, X_basis, Y_train)
#         grad_fn = grad_wrapper_F(X_train, X_basis, Y_train)

#         result = minimize(objective_fn, init_hyperparams, jac=grad_fn,
#                         bounds=bounds, method="L-BFGS-B",
#                         options={'gtol': 1e-8})

#         opt_params = result.x
#         opt_mse = float(result.fun)
#         opt_sigma = opt_params[:-1]
#         opt_lambda = opt_params[-1]

#         # sigma = opt_params[:-1]
#         # lambda_reg = opt_params[-1]
#         opt_hyperparams_list.append(result.x)
#         mse_train_list.append(opt_mse)

#         # Build Model with optimised hyperparameters on training set
#         K_NM = build_kernel_matrix_jnp_F(X_train, X_basis, opt_sigma)
#         K_MM = build_kernel_matrix_jnp_F(X_basis, X_basis, opt_sigma)
#         alpha = find_alpha_jnp(K_NM, K_MM, Y_train, opt_lambda)

#         ###### mse for X_test ######
#         X_test, Y_test = X_Y_dataset_step_with_F(N, limits)

#         # Build kernel matrices
#         K_NM_test = build_kernel_matrix_jnp_F(X_test, X_basis, opt_sigma)
#         K_MM_test = build_kernel_matrix_jnp_F(X_basis, X_basis, opt_sigma)

#         Y_pred_test = K_NM_test @ alpha # using alpha from X_train set
#         mse = float(np.mean((Y_pred_test[:-1] - Y_test[:-1])**2)) # MSE for all except force input
#         mse_test_list.append(mse)

#     # Best hyperparameters based on minimum MSE
#     index_min_mse = np.argmin(np.array(mse_train_list)**2 + np.array(mse_test_list)**2)

#     if plot_results:
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.scatter(mse_train_list, mse_test_list, marker='o', color='blue', alpha=0.5)
#         ax.set_xscale('log')
#         ax.set_yscale('log')
#         ax.set_xlim([min(mse_train_list)/1.1, max(mse_train_list)*1.1])
#         ax.set_ylim([min(mse_test_list)/1.1, max(mse_test_list)*1.1])
#         ax.set_xlabel("Training MSE")
#         ax.set_ylabel("Test MSE")
#         ax.grid(True)

#         index_min_mse = np.argmin(np.array(mse_train_list)**2 + np.array(mse_test_list)**2)
#         ax.scatter(mse_train_list[index_min_mse], mse_test_list[index_min_mse], marker='s', color='red', alpha=0.5)

#         plt.show(block=False)

#     return opt_hyperparams_list[index_min_mse]

#Training data
N = 1000  # Number of training samples
M = 400   # Number of basis functions
limits_force = (-20, 20)
limits=[(-10, 10), (-10, 10), (-np.pi, np.pi), (-15, 15), limits_force]
X_train, Y_train = X_Y_dataset_step_with_F(N, limits)
X_basis = X_train[np.random.choice(N, M, replace=False)]
opt_hyperparams = optimise_hyperparams(X_train, Y_train, X_basis, limits=limits, no_init_hparams=50, plot_results=False)
print(f"Optimised hyperparameters: {opt_hyperparams}")
plt.show()