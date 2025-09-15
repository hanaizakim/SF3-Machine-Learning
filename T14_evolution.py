#Task 1.4
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib.collections import LineCollection
from T00_CartPole import CartPole, remap_angle
from T11_dynamic_simulation import simulate_rollout, add_legend, remap_angle_2pi, plot_rollout, plot_phase_portraits, plot_report
from T13_linear_model import X_Y_dataset_step, least_squares_solution

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

#################
def plot_gradient_line(ax, x, y, cmap='cool'):
    segments = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([segments[:-1], segments[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, len(x)))
    lc.set_array(np.arange(len(x)))
    lc.set_linewidth(2)
    return ax.add_collection(lc)

#######################
if __name__ == "__main__":
    plt.rcParams.update({'font.size': 14})
    # Linear model
    N = 500
    limits = [(-10,10),(-10,10),(-np.pi, np.pi),(-15,15)]
    F = 0.0
    tsteps = 200
    X,Y = X_Y_dataset_step(N, limits)
    C_T = least_squares_solution(X, Y)

    # Create color cycle using tab10 colormap
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    #%% Varying initial cart velocity - x_dot
    fig_rollout, fig_phase, fig_report = None, None, None
    for i, x_dot in enumerate(np.linspace(-10,10,7)):
        initial_state = [0.0, x_dot, np.pi, 0.0] # start hanging down, with velocity

        label = f"$x^\prime$ = {x_dot:.2f}$m/s$"
        time, true_traj = simulate_rollout(initial_state, F, steps=tsteps)
        predicted_traj = linear_simulate_rollout(C_T, initial_state, steps=tsteps)

        # True model
        fig_rollout = plot_rollout(time,true_traj,fig=fig_rollout,color=colors[i % 10],label=label)
        # fig_phase = plot_phase_portraits(time,true_traj,fig=fig_phase,label=label)
        # fig_report = plot_report(time,true_traj,fig=fig_report,label=label)

        # Linear model: Dotted line
        fig_rollout = plot_rollout(time, predicted_traj, fig=fig_rollout, style=":", color=colors[i % 10], label=None)
        # fig_phase = plot_phase_portraits(time,predicted_traj, fig=fig_phase, style=":", color=lastcolor, label=None)
        # fig_report = plot_report(time,predicted_traj, fig=fig_report,style=":",color=lastcolor, label=None)

    plt.show(block=False)

    #%% Varying initial cart angle - theta
    fig_rollout, fig_phase, fig_report = None, None, None
    for i, theta_deg in enumerate([0,45,90,135,180]):
        initial_state = [0.0, 0.0, theta_deg/180*np.pi, 0.0]

        label = f"$\Theta_o$ = {theta_deg}$^\circ$"
        time, true_traj = simulate_rollout(initial_state, F, steps=tsteps)
        predicted_traj = linear_simulate_rollout(C_T, initial_state, steps=tsteps)

        # True model
        fig_rollout = plot_rollout(time,true_traj,fig=fig_rollout,color=colors[i % 10],label=label)
        # fig_phase = plot_phase_portraits(time,true_traj,fig=fig_phase,label=label)
        # fig_report = plot_report(time,true_traj,fig=fig_report,label=label)

        # Linear model: Dotted line
        fig_rollout = plot_rollout(time, predicted_traj, fig=fig_rollout, style=":", color=colors[i % 10], label=None)
        # fig_phase = plot_phase_portraits(time,predicted_traj, fig=fig_phase, style=":", color=lastcolor, label=None)
        # fig_report = plot_report(time,predicted_traj, fig=fig_report,style=":",color=lastcolor, label=None)

    plt.show(block=False)

    #%% Phase plot - theta vs theta_dot
    fig_report = None
    for theta_deg in [135]:
        initial_state = [0.0, 0.0, theta_deg/180*np.pi, 0.0]

        label = f"$\\Theta_o$ = {theta_deg}$^\\circ$"
        time, true_traj = simulate_rollout(initial_state, F, steps=500)
        predicted_traj = linear_simulate_rollout(C_T, initial_state, steps=500)

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        line=plot_gradient_line(ax, remap_angle_2pi(true_traj[:,2])*180/np.pi, true_traj[:,3])
        ax.plot([], [], color=plt.get_cmap('cool')(0.8), label="Cartpole.py model")
        line2=plot_gradient_line(ax, remap_angle_2pi(predicted_traj[:,2])*180/np.pi, predicted_traj[:,3])
        ax.plot([], [], ':', color=plt.get_cmap('cool')(0.8), label="Linear model")
        line2.set_linestyle(':')
        plt.plot(remap_angle_2pi(true_traj[0,2])*180/np.pi, true_traj[0,3], 'x',color='black')
        plt.plot(remap_angle_2pi(true_traj[-1,2])*180/np.pi, true_traj[-1,3], 'o',color='black')
        plt.plot(remap_angle_2pi(predicted_traj[-1,2])*180/np.pi, predicted_traj[-1,3], 's',color='black')
        cbar = plt.colorbar(line, ax=ax)
        cbar.set_label('Time Steps')
        plt.xlim(0, 360)
        ymin, ymax = plt.ylim()
        plt.ylim(-ymax, ymax)
        plt.xlabel(r'$\Theta$ (deg)')
        plt.ylabel(r'$\Theta^\prime$ (rad/s)')
        plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.3)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5),  ncol=1, frameon=False)
        plt.grid(True)

    plt.show()
