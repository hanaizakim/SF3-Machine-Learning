import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from T00_jax_CartPole import CartPole, remap_angle
from tqdm import tqdm
import jax.numpy as jnp
import jax
from functools import partial
from scipy.optimize import minimize

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

def find_alpha_jnp(K_NM, K_MM, Y, lambda_reg):
    A = K_NM.T @ K_NM + lambda_reg * K_MM
    b = K_NM.T @ Y
    alpha = jnp.linalg.solve(A, b)
    return alpha

def label_subplots(fig):
    """Label subplots with letters a, b, c, d."""
    for i, ax in enumerate(fig.axes):
            ax.text(0.0, 1.0, f"({chr(ord('a') + i)})",
                    transform=ax.transAxes + ScaledTranslation(-60 / 72, +7 / 72, fig.dpi_scale_trans), fontsize=16)

def least_squares_solution(X,Y):
    # least squares solution to Y = X  C_T
    C_T, *_ = np.linalg.lstsq(X, Y, rcond=None) #C = C_T.T  # shape (4, 4)
    return(C_T)

def linear_simulate_rollout(C_T, initial_state, steps=200):
    # linear model rollout using C_T
    state = initial_state
    rollout = [state]
    for _ in range(steps):
        delta = state @ C_T
        state = state + delta
        state[2] = remap_angle(state[2])  # to avoid divergence
        rollout.append(state.copy())
    return np.array(rollout)


def add_legend(fig):
    """Clear and add a new legend to the figure."""
    for legend in fig.legends:
        legend.remove()
    fig.legends.clear()

    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                   ncol=int((len(labels)+1)/2), fontsize='large', frameon=False)
        
# Generate dataset
def X_Y_dataset_step_with_F(N, limits):
    X = np.zeros((N, len(limits)))
    Y = np.zeros((N, len(limits)))
    for i in range(N):
        X0 = np.array([np.random.uniform(*limit) for limit in limits])
        force = X0[-1]
        _, X_t = simulate_rollout(X0[:-1], F=force, steps=1)
        X[i] = X0
        Y[i] = np.append(X_t[-1], [force]) - X0
    return X, Y

@jax.jit
def build_kernel_matrix_jnp_F(X1, X2, sigma):
    X1_t = transform_states(X1)
    X2_t = transform_states(X2)
    diffs = (X1_t[:, None, :] - X2_t[None, :, :]) / sigma
    return jnp.exp(-0.5 * jnp.sum(diffs ** 2, axis=-1))

def nonlinear_simulate_rollout_jnp_F(initial_state, alpha, X_basis, sigma, steps=200):
    state = initial_state.copy()
    rollout = [state.copy()]
    for _ in range(steps):
        delta = build_kernel_matrix_jnp_F(state.reshape(1, -1), X_basis, sigma) @ alpha
        state = state + delta[0]
        state = state.at[-1].set(initial_state[-1])
        rollout.append(state.copy())
    return jnp.array(rollout)

# Kernel & transformation functions
@jax.jit
def transform_states(X):
    return jnp.column_stack((
        X[:, 0],
        X[:, 1],
        jnp.sin(X[:, 2]),
        jnp.cos(X[:, 2]),
        X[:, 3],
        X[:, 4],
    ))


def optimise_hyperparams(X_train,Y_train,X_basis,limits=[(-10, 10), (-10, 10), (-np.pi, np.pi), (-15, 15)], no_init_hparams=50, plot_results=True):

    N, M = X_train.shape[0], X_basis.shape[0]

    init_hyperparams_list = []
    for i in range(no_init_hparams):
        sigma_xv = np.random.uniform(0.01, 20.0, 2)        # For x, x_dot
        sigma_stct = np.random.uniform(0.01, 1.0, 2)       # For sin(theta), cos(theta)
        sigma_tdot = np.random.uniform(0.01, 20.0, 1)      # For theta_dot
        sigma_F = np.random.uniform(0.01, 20.0, 1)         # For force input
        lambda_rand = np.random.uniform(1e-06, 0.1)        # Regularization term
        init_hyperparams = np.concatenate([sigma_xv, sigma_stct, sigma_tdot, sigma_F, [lambda_rand]])
        init_hyperparams_list.append(init_hyperparams)

    mse_train_list = []
    mse_test_list = []
    output_list = []
    opt_hyperparams_list = []
    for init_hyperparams in tqdm(init_hyperparams_list, desc="Optimizing Hyperparameters", unit=" trials"):
        bounds = [(0.01, 20.0), (0.01, 20.0), (0.01, 1), (0.01, 1), (0.01, 20.0), (1e-06, 0.1), (0.01, 20.0)]

        # Create objective and gradient functions with fixed data
        objective_fn, trajectory = objective_with_tracking_F(X_train, X_basis, Y_train)
        grad_fn = grad_wrapper_F(X_train, X_basis, Y_train)

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
        K_NM = build_kernel_matrix_jnp_F(X_train, X_basis, opt_sigma)
        K_MM = build_kernel_matrix_jnp_F(X_basis, X_basis, opt_sigma)
        alpha = find_alpha_jnp(K_NM, K_MM, Y_train, opt_lambda)

        ###### mse for X_test ######
        X_test, Y_test = X_Y_dataset_step_with_F(N, limits)

        # Build kernel matrices
        K_NM_test = build_kernel_matrix_jnp_F(X_test, X_basis, opt_sigma)
        K_MM_test = build_kernel_matrix_jnp_F(X_basis, X_basis, opt_sigma)

        Y_pred_test = K_NM_test @ alpha # using alpha from X_train set
        mse = float(np.mean((Y_pred_test[:-1] - Y_test[:-1])**2)) # MSE for all except force input
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
#########################################




# Plotting
def plot_trajectory(states, pole_length=0.5, dt=0.01):
    if len(states.shape) == 2:
        states = states[..., jnp.newaxis]

    x = states[:, 0, :]
    theta = states[:, 2, :]
    x_pole = x + pole_length * jnp.sin(theta)
    y_pole = pole_length * jnp.cos(theta)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.grid(True)
    for i in range(states.shape[-1]):
        color = ax.plot(x[:, i], jnp.zeros_like(x[:, i]), label=f'Traj {i+1}')[0].get_color()
        ax.plot(x_pole[:, i], y_pole[:, i], '--', color=color, alpha=0.5)
        ax.plot(x[0, i], 0, 'x', color=color)
        ax.plot(x[-1, i], 0, 'o', color=color)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_aspect('equal')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t = np.arange(states.shape[0]) * dt
    labels = ['x (m)', "x' (m/s)", r'θ (rad)', r"θ' (rad/s)"]
    for i, ax in enumerate(axes.flat):
        ax.plot(t, states[:, i, :])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(labels[i])
        ax.grid(True)
    axes[1][0].set_ylim(-2*np.pi, 2*np.pi)

    for i, ax in enumerate(fig.axes[1:]):
        ax.text(0.0, 1.0, f"({chr(ord('a') + i)})",
                transform=ax.transAxes + ScaledTranslation(-55 / 72, +7 / 72, fig.dpi_scale_trans), fontsize=16)

    plt.tight_layout()
    plt.show()

# Initial conditions
def generate_init_conditions(num, limits):
    return jnp.array([np.random.uniform(low, high, num) for low, high in limits]).T

# Differentiable simulation (JAX-friendly)
@partial(jax.jit, static_argnames=['tsteps', 'max_force'])
def jax_differentiable_sim(x0, policy, tsteps=50, max_force=20):
    cp = CartPole(visual=False)
    cp.max_force = max_force
    def update(carry, _):
        cp.setState(carry)
        a = jnp.dot(policy, carry)
        cp.performAction(a)
        return cp.getState(), (cp.getState(), a)
    _, (X, actions) = jax.lax.scan(update, x0.copy(), jnp.zeros(tsteps))
    return X, actions

# Rollout using actual physics
def rollout(init_conditions, tsteps=100):
    cp = CartPole(visual=False)
    cp.setState(init_conditions)
    states = [cp.getState()]
    for _ in range(tsteps - 1):
        cp.performAction(0.0)
        states.append(cp.getState())
    return jnp.stack(states)

#'''''''''''''''''''''''''''''




# Objective
@jax.jit
def objective_function_F(hyperparams, X, X_basis, Y):
    sigma = hyperparams[:-1]
    lambda_reg = hyperparams[-1]
    K_NM = build_kernel_matrix_jnp_F(X, X_basis, sigma)
    K_MM = build_kernel_matrix_jnp_F(X_basis, X_basis, sigma)
    alpha = find_alpha_jnp(K_NM, K_MM, Y, lambda_reg)
    Y_pred = K_NM @ alpha
    mse = jnp.mean((Y[:, :-1] - Y_pred[:, :-1]) ** 2)  # exclude force
    return jnp.where(jnp.isfinite(mse), mse, 1e6)

def objective_with_tracking_F(X, X_basis, Y):
    trajectory = {'iterations': [], 'hyperparams': [], 'mse': []}
    def objective_fn(hyperparams):
        mse = objective_function_F(jnp.array(hyperparams), X, X_basis, Y)
        i = len(trajectory['iterations'])
        trajectory['iterations'].append(i)
        trajectory['hyperparams'].append(np.array(hyperparams))
        trajectory['mse'].append(float(mse))
        return mse
    return objective_fn, trajectory

# Gradient
grad_objective_F = jax.jit(jax.grad(objective_function_F))
def grad_wrapper_F(X, X_basis, Y):
    def grad_fn(hyperparams):
        return grad_objective_F(jnp.array(hyperparams), X, X_basis, Y)
    return grad_fn






@partial(jax.jit, static_argnames=['tsteps', 'max_force'])
def jax_differentiable_sim_noisy_sensors(x0, policy, tsteps=50, max_force=20, noise_std=0, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)

    cp = CartPole(visual=False)
    cp.max_force = max_force

    def update(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        cp.setState(state)
        noise_scale = 1.0 + jax.random.normal(subkey, shape=state.shape) * noise_std
        state_noisy = state * noise_scale
        a = jnp.dot(policy, state_noisy)
        cp.performAction(a)
        next_state = cp.getState()
        return (next_state, key), (next_state, a)

    carry_final, scan_output = jax.lax.scan(update, (x0.copy(), key), jnp.zeros(tsteps))
    X, actions = scan_output
    return X, actions

@partial(jax.jit, static_argnames=['tsteps', 'max_force'])
def jax_differentiable_sim_noisy_actuator(x0, policy, tsteps=50, max_force=20, noise_std=0, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)

    cp = CartPole(visual=False)
    cp.max_force = max_force

    def update(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        cp.setState(state)
        a = jnp.dot(policy, state)
        noise_scale = 1.0 + jax.random.normal(subkey, shape=a.shape) * noise_std
        cp.performAction(a*noise_scale)
        next_state = cp.getState()
        return (next_state, key), (next_state, a)

    carry_final, scan_output = jax.lax.scan(update, (x0.copy(), key), jnp.zeros(tsteps))
    X, actions = scan_output
    return X, actions

@partial(jax.jit, static_argnames=['tsteps', 'max_force'])
def jax_differentiable_sim_noisy_system(x0, policy, tsteps=50, max_force=20, noise_std=0, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)

    cp = CartPole(visual=False)
    cp.max_force = max_force

    def update(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        cp.setState(state)
        a = jnp.dot(policy, state)
        cp.performAction(a)
        noise_scale = 1.0 + jax.random.normal(subkey, shape=state.shape) * noise_std
        next_state = cp.getState() * noise_scale
        return (next_state, key), (next_state, a)

    carry_final, scan_output = jax.lax.scan(update, (x0.copy(), key), jnp.zeros(tsteps))
    X, actions = scan_output
    return X, actions

@partial(jax.jit, static_argnames=['tsteps', 'max_force'])
def jax_differentiable_sim_noisy_all(x0, policy, tsteps=50, max_force=20, noise_std=[0,0,0], key=None):
    if key is None:
        key = jax.random.PRNGKey(0)

    cp = CartPole(visual=False)
    cp.max_force = max_force

    def update(carry, _):
        state, key = carry
        key, key_sensor, key_actuator, key_system = jax.random.split(key, 4)

        cp.setState(state)

        # Sensor noise
        noise_scale_sensor = 1.0 + jax.random.normal(key_sensor, shape=state.shape) * noise_std[0]
        state_noisy = state * noise_scale_sensor

        # Control with noisy sensors
        a = jnp.dot(policy, state_noisy)

        # Actuator noise
        noise_scale_actuator = 1.0 + jax.random.normal(key_actuator, shape=()) * noise_std[1]
        cp.performAction(a * noise_scale_actuator)

        # System/process noise
        next_state = cp.getState()
        noise_scale_system = 1.0 + jax.random.normal(key_system, shape=next_state.shape) * noise_std[2]
        next_state_noisy = next_state * noise_scale_system

        return (next_state_noisy, key), (next_state_noisy, a)

    carry_final, (X, actions) = jax.lax.scan(update, (x0.copy(), key), jnp.zeros(tsteps))
    return X, actions

@partial(jax.jit, static_argnames=['tsteps', 'max_force'])
def jax_differentiable_sim_sincos(x0, policy, tsteps=50, max_force=20):
    cp = CartPole(visual=False)
    cp.max_force = max_force
    def update(carry, _):
        cp.setState(carry)
        sc = jnp.stack([carry[0], carry[1], jnp.sin(carry[2]), jnp.cos(carry[2]), carry[3]])
        a = jnp.dot(policy, sc)
        cp.performAction(a)
        return cp.getState(), (cp.getState(), a)
    _, (X, actions) = jax.lax.scan(update, x0.copy(), jnp.zeros(tsteps))
    return X, actions

# Loss function for training
def make_loss_fn(init_conditions, sigmas, tsteps, sim_fn):
    @jax.jit
    def loss_fn(policy):
        states = sim_fn(init_conditions, policy, tsteps)
        scaled = states / sigmas[jnp.newaxis, :, jnp.newaxis]
        losses = 1 - jnp.exp(-0.5 * jnp.einsum('aib,aib->ab', scaled, scaled))
        return losses.sum()
    return loss_fn
