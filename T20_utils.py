import numpy as np
from T00_CartPole import CartPole, remap_angle
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import jax.numpy as jnp
import jax
from scipy.optimize import minimize
from tqdm import tqdm

def X_Y_dataset_step(N, limits):
    X = np.zeros((N, 4))
    Y = np.zeros((N, 4))
    for i in range(N):
        X0 = np.array([np.random.uniform(*limit) for limit in limits])
        _, X_t = simulate_rollout(X0, F=0.0, steps=1)
        X[i] = X0
        Y[i] = X_t[-1] - X0
    return X, Y

def simulate_rollout(initial_state, F=0, Fmax = 20, steps=200):
    """Simulate CartPole rollout given an initial state and force."""
    cartpole = CartPole(visual=False)
    cartpole.setState(initial_state)
    cartpole.max_force = Fmax

    states = [initial_state.copy()]
    for _ in range(steps):
        cartpole.performAction(action=F)
        states.append(cartpole.getState().copy())

    time = np.arange(steps + 1) * cartpole.delta_time
    return time, np.array(states)

def nonlinear_simulate_rollout(initial_state, alpha, X_basis, sigma, steps=200):
    state=initial_state.copy()
    rollout = [state.copy()]
    for _ in range(steps):
        K_NM = build_kernel_matrix(state.reshape(1,-1), X_basis, sigma)
        delta = K_NM @ alpha
        state = state + delta[0]
        state[2] = remap_angle(state[2])
        rollout.append(state.copy())
    return np.array(rollout)

def plot_rollout(time, states_matrix, fig=None, style = "-", color = None, label=None):
    """Plot rollout trajectories: x, x', θ, θ' over time."""
    x, x_dot, theta, theta_dot = (
        states_matrix[:, 0],
        states_matrix[:, 1],
        remap_angle_2pi(states_matrix[:, 2]) * 180 / np.pi,
        states_matrix[:, 3] * 180 / np.pi
    )
    states = [x, x_dot, theta, theta_dot]
    ylabels = ['Cart Position $x$ (m)', 'Cart velocity $x\'$ (m/s)',
               'Pole angle $\\theta$ (deg)', "Pole velocity $\\theta'$ (deg/s)"]

    if fig is None:
        fig, _ = plt.subplots(2, 2, figsize=(12, 8))
        for i, ax in enumerate(fig.axes):
            ax.text(0.0, 1.0, f"({chr(ord('a') + i)})",
                    transform=ax.transAxes + ScaledTranslation(-55 / 72, +7 / 72, fig.dpi_scale_trans), fontsize=16)

    if color is None:
        color = next(iter(plt.rcParams['axes.prop_cycle']))['color']

    for i, ax in enumerate(fig.axes):
        ax.plot(time, states[i], style, color=color, label=label)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabels[i])
        ax.set_xlim(0, time[-1])
        ax.grid(True)
    fig.axes[2].set(ylim=(0, 360), yticks=np.arange(0, 361, 45))

    fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.2, hspace=0.35, wspace=0.3)
    add_legend(fig)
    return fig

def remap_angle_2pi(thetas):
    """Remap angles to the range [0, 2π)."""
    return thetas % (2 * np.pi) if np.isscalar(thetas) else np.array([theta % (2 * np.pi) for theta in thetas])

def add_legend(fig):
    """Clear and add a new legend to the figure."""
    for legend in fig.legends:
        legend.remove()
    fig.legends.clear()

    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        ncol = int((len(labels)+1)/2) if len(labels)>5 else len(labels)
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                   ncol=ncol, fontsize='large', frameon=False)

def build_kernel_matrix(X1, X2, sigma):
    if len(sigma) == 4:
        diffs = X1[:, None, :] - X2[None, :, :]
        diffs[..., 2] = np.sin(diffs[..., 2]/2) # periodic theta
    elif len(sigma) == 5:
        X1_transformed = np.column_stack((X1[:, :2], np.sin(X1[:, 2]), np.cos(X1[:, 2]), X1[:, -1]))
        X2_transformed = np.column_stack((X2[:, :2], np.sin(X2[:, 2]), np.cos(X2[:, 2]), X2[:, -1]))
        diffs = X1_transformed[:, None, :] - X2_transformed[None, :, :]
    else:
        raise ValueError(f"sigma is incorrect: {len(sigma)}")
    K = np.exp(-np.sum((diffs ** 2) / (2 * sigma**2), axis=-1))
    return K

def find_alpha(K_NM, K_MM, Y, lambda_reg):
    A = K_NM.T @ K_NM + lambda_reg * K_MM
    b = K_NM.T @ Y
    alpha = np.linalg.lstsq(A, b, rcond=None)[0]
    return alpha

def label_subplots(fig):
    """Label subplots with letters a, b, c, d."""
    for i, ax in enumerate(fig.axes):
            ax.text(0.0, 1.0, f"({chr(ord('a') + i)})",
                    transform=ax.transAxes + ScaledTranslation(-60 / 72, +7 / 72, fig.dpi_scale_trans), fontsize=16)

#=========================
# JAX
#=========================

def X_Y_dataset_step_jnp(N, limits):
    X, Y = X_Y_dataset_step(N, limits)
    return jnp.array(X), jnp.array(Y)

@jax.jit
def build_kernel_matrix_jnp(X1, X2, sigma):
    if len(sigma) == 4:
        diffs = X1[:, None, :] - X2[None, :, :]
        diffs = diffs.at[..., 2].apply(lambda x: jnp.sin(x/2))  # periodic theta
    elif len(sigma) == 5:
        X1_transformed = transform_states(X1)
        X2_transformed = transform_states(X2)
        diffs = X1_transformed[:, None, :] - X2_transformed[None, :, :]
    else:
        raise ValueError(f"sigma is incorrect: {len(sigma)}")

    scaled_diffs = diffs / sigma
    K = jnp.exp(-0.5 * jnp.sum(scaled_diffs ** 2, axis=-1))
    return K

# Objective function definition
# We need to define objective_fn(hyperparams). However, we also want to pass static parameters X, X_basis, and Y.
# Therefore we define objective_fn_params which takes hyperparameters and additional parameters (X, X_basis, Y)
# and returns the mse we want to minimize.
# We then can define objective_fn as a partial function that binds X, X_basis, and Y:
#    objective_fn = partial(objective_fn_params, X=X, X_basis=X_basis, Y=Y)
# We can also jax.jit the objective function to improve performance:
#    objective_fn = jax.jit(objective_fn_params, X=X, X_basis=X_basis, Y=Y)
def objective_fn_params(hyperparams, X, X_basis, Y):
    X = jnp.array(X)
    X_basis = jnp.array(X_basis)
    Y = jnp.array(Y)

    # Clip hyperparameters to safe ranges to avoid numerical/convergence issues
    sigma = jnp.clip(jnp.array(hyperparams[:4]), 1e-3, 20.0)
    lambda_reg = jnp.clip(hyperparams[4], 1e-6, 0.1)

    K_NM = build_kernel_matrix_jnp(X, X_basis, sigma)
    K_MM = build_kernel_matrix_jnp(X_basis, X_basis, sigma)
    A = K_NM.T @ K_NM + lambda_reg * K_MM
    b = K_NM.T @ Y

    # lstsq causes convergence problems when M is large, so we use solve instead
    # alpha = jnp.linalg.lstsq(A, b, rcond=None)[0]
    alpha = jnp.linalg.solve(A, b)

    Y_pred = K_NM @ alpha
    mse = jnp.mean((Y - Y_pred) ** 2)

    # Ensure finite value returned
    return jnp.where(jnp.isfinite(mse), mse, 1e6)

def transform_states(X):
    # x, x_dot, theta, theta_dot -> x, x_dot, sin(theta), cos(theta), theta_dot
    X_transformed = jnp.column_stack((X[:, :2], jnp.sin(X[:, 2]), jnp.cos(X[:, 2]), X[:, -1]))
    return X_transformed

def find_alpha_jnp(K_NM, K_MM, Y, lambda_reg):
    A = K_NM.T @ K_NM + lambda_reg * K_MM
    b = K_NM.T @ Y
    alpha = jnp.linalg.solve(A, b)
    return alpha

def nonlinear_simulate_rollout_jnp(initial_state, alpha, X_basis, sigma, steps=200):
    state = initial_state.copy()
    rollout = [state.copy()]
    for _ in range(steps):
        K_NM = build_kernel_matrix_jnp(state.reshape(1,-1), X_basis, sigma)
        delta = K_NM @ alpha
        state = state + delta[0]
        state = state.at[2].set(remap_angle(state[2]))
        rollout.append(state.copy())
    return jnp.array(rollout)

@jax.jit
def objective_function(hyperparams, X, X_basis, Y):
    sigma = hyperparams[:-1]
    lambda_reg = hyperparams[-1]

    K_NM = build_kernel_matrix_jnp(X, X_basis, sigma)
    K_MM = build_kernel_matrix_jnp(X_basis, X_basis, sigma)

    alpha = find_alpha_jnp(K_NM, K_MM, Y, lambda_reg)
    Y_pred = K_NM @ alpha
    mse = jnp.mean((Y - Y_pred) ** 2)
    return jnp.where(jnp.isfinite(mse), mse, 1e6)

def objective_with_tracking(X, X_basis, Y):
    trajectory = {
        'iterations': [],
        'hyperparams': [],
        'mse': []
    }

    def objective_fn(hyperparams):
        hyperparams = jnp.array(hyperparams)
        mse = objective_function(hyperparams, X, X_basis, Y)
        trajectory['iterations'].append(len(trajectory['iterations']))
        trajectory['hyperparams'].append(np.array(hyperparams))
        trajectory['mse'].append(float(mse))
        return mse

    return objective_fn, trajectory

grad_objective = jax.jit(jax.grad(objective_function))

def grad_wrapper(X, X_basis, Y):
    def grad_fn(hyperparams):
        hyperparams = jnp.array(hyperparams)
        return grad_objective(hyperparams, X, X_basis, Y)
    return grad_fn




def optimise_hyperparams(X_train,Y_train,X_basis,limits=[(-10, 10), (-10, 10), (-np.pi, np.pi), (-15, 15)], no_init_hparams=50, plot_results=True):

    N, M = X_train.shape[0], X_basis.shape[0]

    init_hyperparams_list = []
    for i in range(no_init_hparams):
        sigma_xv = np.random.uniform(0.01, 20.0, 2)        # For x, x_dot
        sigma_stct = np.random.uniform(0.01, 1.0, 2)       # For sin(theta), cos(theta)
        sigma_tdot = np.random.uniform(0.01, 20.0, 1)      # For theta_dot
        lambda_rand = np.random.uniform(1e-06, 0.1)        # Regularization term
        init_hyperparams = np.concatenate([sigma_xv, sigma_stct, sigma_tdot, [lambda_rand]])
        init_hyperparams_list.append(init_hyperparams)

    mse_train_list = []
    mse_test_list = []
    output_list = []
    opt_hyperparams_list = []
    for init_hyperparams in tqdm(init_hyperparams_list, desc="Optimizing Hyperparameters", unit=" trials"):
        bounds = [(0.01, 20.0), (0.01, 20.0), (0.01, 1), (0.01, 1), (0.01, 20.0), (1e-06, 0.1)]

        # Create objective and gradient functions with fixed data
        objective_fn, trajectory = objective_with_tracking(X_train, X_basis, Y_train)
        grad_fn = grad_wrapper(X_train, X_basis, Y_train)

        result = minimize(objective_fn, init_hyperparams, jac=grad_fn,
                        bounds=bounds, method="L-BFGS-B",
                        options={'gtol': 1e-8})

        opt_params = result.x
        opt_mse = float(result.fun)
        opt_sigma = opt_params[:-1]
        opt_lambda = opt_params[-1]

        # sigma = opt_params[:-1]
        # lambda_reg = opt_params[-1]
        opt_hyperparams_list.append(result.x)
        mse_train_list.append(opt_mse)

        # Build Model with optimised hyperparameters on training set
        K_NM = build_kernel_matrix_jnp(X_train, X_basis, opt_sigma)
        K_MM = build_kernel_matrix_jnp(X_basis, X_basis, opt_sigma)
        alpha = find_alpha_jnp(K_NM, K_MM, Y_train, opt_lambda)

        ###### mse for X_test ######
        X_test, Y_test = X_Y_dataset_step_jnp(N, limits)

        # Build kernel matrices
        K_NM_test = build_kernel_matrix_jnp(X_test, X_basis, opt_sigma)
        K_MM_test = build_kernel_matrix_jnp(X_basis, X_basis, opt_sigma)

        Y_pred_test = K_NM_test @ alpha # using alpha from X_train set
        mse = float(np.mean((Y_pred_test - Y_test)**2))
        mse_test_list.append(mse)

    # Best hyperparameters based on minimum MSE
    index_min_mse = np.argmin(np.array(mse_train_list)**2 + np.array(mse_test_list)**2)

    if plot_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(mse_train_list, mse_test_list, marker='o', color='blue', alpha=0.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([min(mse_train_list)/1.1, max(mse_train_list)*1.1])
        ax.set_ylim([min(mse_test_list)/1.1, max(mse_test_list)*1.1])
        ax.set_xlabel("Training MSE")
        ax.set_ylabel("Test MSE")
        ax.grid(True)

        index_min_mse = np.argmin(np.array(mse_train_list)**2 + np.array(mse_test_list)**2)
        ax.scatter(mse_train_list[index_min_mse], mse_test_list[index_min_mse], marker='s', color='red', alpha=0.5)

        plt.show(block=False)

    return opt_hyperparams_list[index_min_mse]