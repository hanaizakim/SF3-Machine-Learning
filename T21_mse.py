import numpy as np
import matplotlib.pyplot as plt
from T00_CartPole import CartPole
from T20_utils import X_Y_dataset_step, simulate_rollout, build_kernel_matrix, find_alpha, label_subplots

# Plot settings
plt.rcParams.update({'font.size': 14})

# Experiment parameters
Ns = [400, 800, 1600, 3200]  # Training sizes
Ms = [10, 20, 40, 80, 160, 320, 640]  # Basis function counts
limits = [(-10, 10), (-10, 10), (-np.pi, np.pi), (-10, 10)]  # State space limits
lambda_reg = 1e-4  # Regularization

# Prepare subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.12, wspace=0.35)

# ---------- Plot 1: MSE vs M for different N ----------
for N in Ns:
    X, Y = X_Y_dataset_step(N, limits)
    sigma = np.std(X, axis=0)
    mse_list = []

    for M in Ms:
        if M > N:
            print(f"Skipping M = {M} for N = {N} (M > N)")
            mse_list.append(np.nan)
            continue

        indices = np.random.choice(N, M, replace=False)
        X_basis = X[indices]

        K_NM = build_kernel_matrix(X, X_basis, sigma)
        K_MM = build_kernel_matrix(X_basis, X_basis, sigma)

        alpha = find_alpha(K_NM, K_MM, Y, lambda_reg)
        Y_pred = K_NM @ alpha

        mse = np.mean((Y_pred - Y) ** 2)
        mse_list.append(mse)

    axes[0].semilogy(Ms, mse_list, 'o-', label=f'N = {N}')

axes[0].set_xlabel('Number of Basis Functions (M)')
axes[0].set_ylabel('Mean Squared Error')
axes[0].legend()
axes[0].grid(True, which="both", linestyle='--', linewidth=0.5)

# ---------- Plot 2: MSE vs lambda for fixed N, varying M ----------
N = max(Ns)
X, Y = X_Y_dataset_step(N, limits)
sigma = np.std(X, axis=0)
lambdas = np.logspace(-6, -1, 10)

for M in Ms:
    indices = np.random.choice(N, M, replace=False)
    X_basis = X[indices]

    K_NM = build_kernel_matrix(X, X_basis, sigma)
    K_MM = build_kernel_matrix(X_basis, X_basis, sigma)

    mse_list = []
    for lambda_val in lambdas:
        alpha = find_alpha(K_NM, K_MM, Y, lambda_val)
        Y_pred = K_NM @ alpha
        mse = np.mean((Y_pred - Y) ** 2)
        mse_list.append(mse)

    axes[1].loglog(lambdas, mse_list, 'o-', label=f'M = {M}')

axes[1].set_xlabel(r'Regularisation ($\lambda$)')
axes[1].set_ylabel('Mean Squared Error')
axes[1].legend(loc='upper left')
axes[1].grid(True, which="both", linestyle='--', linewidth=0.5)

# Display the plots
plt.tight_layout()
plt.show()
