import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from matplotlib.colors import LogNorm
from T00_jax_CartPole import CartPole
from T30_utils import plot_trajectory, rollout, generate_init_conditions, label_subplots
from T30_utils import jax_differentiable_sim,jax_differentiable_sim_noisy_sensors, jax_differentiable_sim_noisy_actuator, jax_differentiable_sim_noisy_system, jax_differentiable_sim_noisy_all
from matplotlib.transforms import ScaledTranslation

plt.rcParams.update({'font.size': 14})

# Cartpole noisy model selection
#jax_sim_model=jax_differentiable_sim_noisy_sensors # Cartpole with noisy sensors
#jax_sim_model=jax_differentiable_sim_noisy_actuator # Cartpole with noisy actuator
jax_sim_model=jax_differentiable_sim_noisy_system # Cartpole with noisy system dynamics (wind, etc)
#jax_sim_model=jax_differentiable_sim_noisy_all # Cartpole with noisy sensor, actuator and dynamics (wind, etc)

dt=0.01 # Time step for simulation in cartpole.py

# Policy optimization
n_initial_conditions = 50
tsteps_optimisation = 300
tsteps_visualisation =10000

#limits = jnp.array([(-1,1), (-1, 1), (-np.pi/12, np.pi/12), (-1, 1)])
limits = jnp.array([(-1,1), (-1, 1), (-np.pi/6, np.pi/6), (-1, 1)])
init_conditions=generate_init_conditions(n_initial_conditions,limits).T

sigmas_for_loss_function = jnp.array([2,100,0.1,100])
init_policy = jnp.zeros(4)
init_policy = jnp.array([13.40218004, 15.16610997, 98.29713825, 13.68337699]) # Last known optimal policy obtained with sigmas_for_loss_function = jnp.array([2, 100, 0.1, 100])
#init_policy = jnp.array([36.78181221 , 36.01101153, 180.17619568, 20.76285015])
opt_policy = init_policy

# Loss and gradient functions
def _loss_fn(states):
    scaled_states = states / sigmas_for_loss_function[jnp.newaxis, :, jnp.newaxis]
    losses = 1 - jnp.exp(-jnp.einsum('aib,aib->ab', scaled_states, scaled_states)/(2.0))
    return losses

stds=[0,0.05,0.25]
for std in stds:
    if jax_sim_model==jax_differentiable_sim_noisy_all:
        std=[std,std,std] # If using all noise, scale std by 3 for each noise type
        std_str = f"{std[0]:.2f}, {std[1]:.2f}, {std[2]:.2f}"
    else:
        std_str = f"{std:.2f}"

    if False: # If optimization is to be run
        loss_fn = lambda policy: _loss_fn(jax_sim_model(init_conditions, policy, tsteps=tsteps_optimisation,max_force=20,noise_std=std)[0]).sum()
        grad_fn = jax.grad(loss_fn)

        # Optimization callback
        def callback(policy):
            loss = loss_fn(policy)
            callback.loss_history.append(loss)
            print(f"Iter {callback.i}: loss = {loss}, policy = {policy}")
            callback.i += 1
        callback.i = 1
        callback.loss_history = []

        # Run optimization
        print(f"Initial loss: {loss_fn(init_policy)}")
        opt = minimize(loss_fn, x0=init_policy, method='L-BFGS-B', jac=grad_fn, callback=callback, options={'disp': True, 'gtol': 1e-3})
        opt_policy = opt.x
        print(f"Optimal policy: {opt_policy}")
        print(f"Optimal loss: {loss_fn(opt_policy)}")

        # Plot loss vs. iteration
        plt.figure(figsize=(8, 5))
        plt.plot(callback.loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()

    #Run simulation with the optimal policy
    X_sim, actions = jax_sim_model(init_conditions, opt_policy, tsteps=tsteps_visualisation,max_force=200,noise_std=std)
    losses = _loss_fn(X_sim)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    t = np.arange(0, len(X_sim) * dt, dt)
    plot_data = [
        (X_sim[:, 0], 'x (m)', 'Position x'),
        (X_sim[:, 2], r'$\theta$ (rad)', r'Angle $\theta$'),
        (actions, 'Force (N)', 'Force'),
        (losses, 'Loss', 'Loss')
    ]

    for ax, (y, ylabel, label) in zip(axes.flatten(), plot_data):
        ax.plot(t, y, label=label)
        ax.set_xscale('log')
        ax.set_xlim(0.01, len(X_sim) * dt)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.grid(True)
    fig.canvas.manager.set_window_title(f"Noisy Cartpole - std = {std_str}")

    # Create figure with 1x4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.15, hspace=0.3, wspace=0.75)

    t = np.arange(0, len(X_sim) * dt, dt)
    plot_data = [
        (X_sim[:, 0], 'x (m)', 'Position x'),
        (X_sim[:, 2], r'$\theta$ (rad)', r'Angle $\theta$'),
        (actions, 'Force (N)', 'Force'),
        (losses, 'Loss', 'Loss')
    ]

    for ax, (y, ylabel, label) in zip(axes, plot_data):
        ax.plot(t, y, label=label)
        ax.set_xscale('log')
        ax.set_xlim(0.01, len(X_sim) * dt)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.grid(True)

    fig.canvas.manager.set_window_title(f"Noisy Cartpole - std = {std_str}")

plt.show()