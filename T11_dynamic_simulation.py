#Task 1.1
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from T00_CartPole import CartPole, remap_angle

def remap_angle_2pi(thetas):
    """Remap angles to the range [0, 2π)."""
    return thetas % (2 * np.pi) if np.isscalar(thetas) else np.array([theta % (2 * np.pi) for theta in thetas])

def simulate_rollout(initial_state, F, steps=200):
    """Simulate CartPole rollout given an initial state and force."""
    cartpole = CartPole(visual=False)
    cartpole.setState(initial_state)

    states = [initial_state]
    for _ in range(steps):
        cartpole.performAction(action=F)
        states.append(cartpole.getState().copy())

    time = np.arange(steps + 1) * cartpole.delta_time
    return time, np.array(states)

def add_legend(fig):
    """Clear and add a new legend to the figure."""
    for legend in fig.legends:
        legend.remove()
    fig.legends.clear()

    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                   ncol=int((len(labels)+1)/2), fontsize='large', frameon=False)

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

def plot_phase_portraits(time, states_matrix, fig=None, style = "-", color=None, label=None):
    """Plot phase portraits for CartPole."""
    x, x_dot, theta, theta_dot = (
        states_matrix[:, 0],
        states_matrix[:, 1],
        remap_angle_2pi(states_matrix[:, 2]) * 180 / np.pi,
        states_matrix[:, 3]
    )

    if fig is None:
        fig, _ = plt.subplots(2, 2, figsize=(12, 8))

    if color is None:
        color = next(iter(plt.rcParams['axes.prop_cycle']))['color']

    xlabels = ['$x$ (m)', r"$\theta$ (deg)", r"$\theta$ (deg)", r"$\theta'$ (rad/s)"]
    ylabels = [r"$x'$ (m/s)", r"$\theta'$ (rad/s)", r"$x$ (m)", r"$x'$ (m/s)"]
    data_pairs = [(x, x_dot), (theta, theta_dot), (theta, x), (theta_dot, x_dot)]
    for i, ax in enumerate(fig.axes):
        ax.plot(*data_pairs[i], style, color=color, label=label)
        ax.set(xlabel=xlabels[i], ylabel=ylabels[i])
        ax.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    add_legend(fig)
    return fig

def plot_report(time, states_matrix, fig=None, style = "-", color=None, label=None):
    """Generate report-style plots with subfigure labels."""
    x, x_dot, theta, theta_dot = (
        states_matrix[:, 0],
        states_matrix[:, 1],
        remap_angle_2pi(states_matrix[:, 2]) * 180 / np.pi,
        states_matrix[:, 3]
    )

    if fig is None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for i, ax in enumerate(axes.flatten()):
            ax.text(0.0, 1.0, f"({chr(ord('a') + i)})",
                    transform=ax.transAxes + ScaledTranslation(-55 / 72, +7 / 72, fig.dpi_scale_trans),
                    fontsize=16)

    if color is None:
        color = next(iter(plt.rcParams['axes.prop_cycle']))['color']

    xlabels = ['Time (s)', 'Time (s)', '$x$ (m)', r'$\theta$ (deg)']
    ylabels = ['$x$ (m)', r'$\theta$ (deg)', "$x'$ (m/s)", r"$\theta'$ (rad/s)"]
    data_pairs = [(time, x), (time, theta), (x, x_dot), (theta, theta_dot)]
    for i, ax in enumerate(fig.axes):
        ax.plot(*data_pairs[i], style, color=color, label=label)
        ax.set(xlabel=xlabels[i], ylabel=ylabels[i])
        ax.grid(True)
    fig.axes[0].set(xlim=(0,np.max(time)))
    fig.axes[1].set(xlim=(0,np.max(time)),yticks=np.arange(0, 361, 90))
    fig.axes[3].set(xlim=(0, 360),xticks=np.arange(0, 361, 90))

    fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.2, hspace=0.35, wspace=0.3)
    add_legend(fig)
    return fig

# =============================== MAIN =================================
if __name__ == "__main__":
    plt.rcParams.update({'font.size': 14})
    
    # Create color maps for different plots
    rollout_colors = plt.cm.viridis(np.linspace(0, 1, 7))  # 7 colors for theta variations
    velocity_colors = plt.cm.plasma(np.linspace(0, 1, 7))   # 7 colors for velocity variations
    angular_colors = plt.cm.magma(np.linspace(0, 1, 6))     # 6 colors for angular velocity variations
    
    # Varying initial pole angles
    fig_rollout, fig_phase, fig_report = None, None, None
    for i, theta_deg in enumerate([0, 30, 60, 90, 120, 150, 180]):
        initial_state = [0.0, 0.0, np.radians(theta_deg), 0.0]
        time, states = simulate_rollout(initial_state, F=0.0)
        label = f"$\\theta_o$ = {theta_deg}$^\\circ$"
        fig_rollout = plot_rollout(time, states, fig=fig_rollout, color=f'C{i}', label=label)
        fig_phase = plot_phase_portraits(time, states, fig=fig_phase, color=f'C{i}', label=label)
        fig_report = plot_report(time, states, fig=fig_report, color=f'C{i}', label=label)
    plt.show(block=False)

    # Varying initial cart velocities
    fig_rollout, fig_phase, fig_report = None, None, None
    for i, x_dot in enumerate(np.linspace(-10, 10, 7)):
        initial_state = [0.0, x_dot, np.pi, 0.0]
        time, states = simulate_rollout(initial_state, F=0.0)
        label = f"$x'$ = {x_dot:.2f} m/s"
        fig_rollout = plot_rollout(time, states, fig=fig_rollout, color=f'C{i}', label=label)
        fig_phase = plot_phase_portraits(time, states, fig=fig_phase, color=f'C{i}', label=label)
        fig_report = plot_report(time, states, fig=fig_report, color=f'C{i}', label=label)
    plt.show(block=False)

    # Varying initial pole angular velocities
    fig_rollout, fig_phase, fig_report = None, None, None
    for i, theta_dot in enumerate(np.linspace(0, 15, 6)):
        initial_state = [0.0, 0.0, np.pi, theta_dot]
        time, states = simulate_rollout(initial_state, F=0.0)
        label = f"$\\theta_o'$ = {theta_dot:.0f} rad/s"
        fig_rollout = plot_rollout(time, states, fig=fig_rollout, color=f'C{i}', label=label)
        fig_phase = plot_phase_portraits(time, states, fig=fig_phase, color=f'C{i}', label=label)
        fig_report = plot_report(time, states, fig=fig_report, color=f'C{i}', label=label)
    plt.show(block=False)

    # Phase plot theta_dot vs x_dot
    plt.figure(figsize=(8,5))
    plt.subplots_adjust(left=0.15, right=0.7, top=0.9, bottom=0.2)
    for i, theta_deg in enumerate([0, 30, 60, 90, 120, 150, 180]):
        initial_state = [0.0, 0.0, np.radians(theta_deg), 0.0]
        time, states = simulate_rollout(initial_state, F=0.0)
        label = f"$\\theta_0$ = {theta_deg}$^\\circ$"
        plt.plot(states[:, 3], states[:, 1], color=f'C{i}', label=label)

    plt.xlabel(r"Pole angular velocity $\theta'$ (rad/s)")
    plt.ylabel(r"Cart velocity $x'$ (m/s)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()
