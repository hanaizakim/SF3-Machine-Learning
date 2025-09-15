import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import itertools    
from T30_utils import least_squares_solution, linear_simulate_rollout, add_legend, optimise_hyperparams, X_Y_dataset_step_with_F, build_kernel_matrix_jnp_F, nonlinear_simulate_rollout_jnp_F, find_alpha_jnp, simulate_rollout, plot_rollout, label_subplots

# Plot settings
plt.rcParams.update({'font.size': 14})

# Parameters and Data
N = 3200         # Number of training samples
M = 500           # Number of basis functions
limits = [(-10, 10), (-10, 10), (-np.pi, np.pi), (-15, 15), (-20, 20)]

# Dataset generation
X, Y = X_Y_dataset_step_with_F(N, limits)
indices = np.random.choice(N, M, replace=False)
X_basis = X[indices]

# Hyperparameter optimisation
# opt_hyperparams = optimise_hyperparams(X, Y, X_basis, limits=limits, no_init_hparams=50, plot_results=True)
# print(f"Optimized hyperparameters: {opt_hyperparams}")
opt_hyperparams = np.array([14.79945546, 13.61614691,  0.57373077,  0.75401413,  9.62018004, 17.42291767, 0.02945733])
opt_sigma = opt_hyperparams[:-1]
opt_lambda_reg = opt_hyperparams[-1]

# Build model with optimised hyperparameters
K_NM = build_kernel_matrix_jnp_F(X, X_basis, opt_sigma)
K_MM = build_kernel_matrix_jnp_F(X_basis, X_basis, opt_sigma)
alpha_opt = find_alpha_jnp(K_NM, K_MM, Y, opt_lambda_reg)

# Rollout comparison
tsteps = 300
forces=[10,20]
for force in forces:
    initial_state = np.array([0.0, 0.0, 135 / 180 * np.pi, 0.0, force])
    time, true_traj = simulate_rollout(initial_state[:-1], F=initial_state[-1], steps=tsteps)
    nonlinear_opt_traj = nonlinear_simulate_rollout_jnp_F(initial_state, alpha_opt, X_basis, opt_sigma, steps=tsteps)
    # Predicted rollout using linear model
    C_T = least_squares_solution(X, Y)
    linear_traj = linear_simulate_rollout(C_T, initial_state, steps=tsteps)

    # Plotting
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = itertools.cycle(prop_cycle.by_key()['color'])

    fig_rollout = plot_rollout(time, true_traj, label='True', color='black')
    fig_rollout = plot_rollout(time, linear_traj, fig=fig_rollout, style='--', color = next(color_cycle), label='Linear')
    fig_rollout = plot_rollout(time, nonlinear_opt_traj, fig=fig_rollout, style="--", color=next(color_cycle), label='Nonlinear - Opt params')

    # Optional: model prediction vs ground truth (scatter plot)
    Y_lin = X @ C_T  # Linear model prediction
    #Y_nonlin_est = build_kernel_matrix_jnp_F(X, X_basis, sigma) @ alpha
    Y_nonlin_opt = build_kernel_matrix_jnp_F(X, X_basis, opt_sigma) @ alpha_opt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.2, hspace=0.35, wspace=0.3)

    state_labels = ['$\Delta x$ (m)', "$\Delta x'$ (m/s)", '$\Delta \\theta$ (rad)', "$\Delta \\theta'$ (rad/s)"]
    for i, ax in enumerate(fig.axes):
        ax.scatter(Y[:, i], Y_lin[:, i], alpha=0.6,label='Linear model')
        ax.scatter(Y[:, i], Y_nonlin_opt[:, i], alpha=0.6, label='Nonlinear - Opt params')
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", color='black')
        ax.set_xlabel(f"'True' state change: {state_labels[i]}")
        ax.set_ylabel(f'Prediction: {state_labels[i]}')
    add_legend(fig)
    label_subplots(fig)

    # Mixed plot
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = itertools.cycle(prop_cycle.by_key()['color'])
    if force==forces[0]:
        new_fig = plot_rollout(time, true_traj, label='True dynamics', color='black')
        new_fig = plot_rollout(time, linear_traj, fig=new_fig, style='--', color = next(color_cycle), label='Linear model')
        new_fig = plot_rollout(time, nonlinear_opt_traj, fig=new_fig, style="--", color=next(color_cycle), label='Nonlinear model')
    else:
        new_fig.axes[1].clear()
        new_fig.axes[1].plot(time, true_traj[:,0], label='True dynamics', color='black')
        new_fig.axes[1].plot(time, linear_traj[:,0], '--', color = next(color_cycle), label='Linear model')
        new_fig.axes[1].plot(time, nonlinear_opt_traj[:,0], "--", color=next(color_cycle), label='Nonlinear model')
        new_fig.axes[1].grid(True)
        prop_cycle = plt.rcParams['axes.prop_cycle']
        color_cycle = itertools.cycle(prop_cycle.by_key()['color'])
        new_fig.axes[3].clear()
        new_fig.axes[3].plot(time, 180/np.pi*true_traj[:, 2], label='True dynamics', color='black')
        theta=180/np.pi*np.array(linear_traj[:, 2])
        theta[theta<0] += 360
        theta[theta>360] -= 360
        new_fig.axes[3].plot(time, theta, '--', color = next(color_cycle), label='Linear model')
        theta=180/np.pi*np.array(nonlinear_opt_traj[:, 2])
        theta[theta<0] += 360
        theta[theta>360] -= 360
        new_fig.axes[3].plot(time, theta, "--", color=next(color_cycle), label='Nonlinear model')
        new_fig.axes[3].grid(True)
        new_fig.axes[1].set_ylabel(r"Cart position $x$ (m)")
        new_fig.axes[3].set_ylabel(r"Pole angle $\Theta$ (deg)")
        new_fig.axes[3].set(ylim=(0, 360), yticks=np.arange(0, 361, 45))
        new_fig.axes[0].text(0.1, 44.5, "F=10N", bbox=dict(facecolor='white', edgecolor='black'))
        new_fig.axes[2].text(0.1, 20.0, "F=10N", bbox=dict(facecolor='white', edgecolor='black'))
        new_fig.axes[1].text(0.1, 89.0, "F=20N", bbox=dict(facecolor='white', edgecolor='black'))
        new_fig.axes[3].text(0.1, 20.0, "F=20N", bbox=dict(facecolor='white', edgecolor='black'))
        new_fig.axes[1].set_xlabel('Time (s)')
        new_fig.axes[3].set_xlabel('Time (s)')
        new_fig.axes[1].set_xlim(0, time[-1])
        new_fig.axes[3].set_xlim(0, time[-1])
    new_fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.15, hspace=0.35, wspace=0.3)
    state_labels = ['$\Delta x$ (m)', "$\Delta x'$ (m/s)", '$\Delta \\theta$ (rad)', "$\Delta \\theta'$ (rad/s)"]
    for i, ax in enumerate(new_fig.axes):
        if i%2:
            ax.clear()
            ax.scatter(Y[:, i], Y_lin[:, i], alpha=0.6,label='Linear model')
            ax.scatter(Y[:, i], Y_nonlin_opt[:, i], alpha=0.6, label='Nonlinear model')
            ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", color='black')
            ax.set_xlabel(f"'True' state change: {state_labels[i]}")
            ax.set_ylabel(f'Prediction: {state_labels[i]}')
    for legend in new_fig.legends:
        legend.remove()
    new_fig.legends.clear()

    #Legends
    handles, labels = new_fig.axes[0].get_legend_handles_labels()
    new_fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.25, 0.0),
                    ncol=2, fontsize='large', frameon=False)
    handles, labels = new_fig.axes[1].get_legend_handles_labels()
    new_fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.75, 0.0),
                    ncol=1, fontsize='large', frameon=False)
    #Subplot labels (b),(d)
    new_fig.axes[1].text(0.0, 1.0, "(b)",
            transform=new_fig.axes[1].transAxes + ScaledTranslation(-60 / 72, +7 / 72, new_fig.dpi_scale_trans), fontsize=16)
    new_fig.axes[3].text(0.0, 1.0, "(d)",
            transform=new_fig.axes[3].transAxes + ScaledTranslation(-60 / 72, +7 / 72, new_fig.dpi_scale_trans), fontsize=16)


plt.show()
