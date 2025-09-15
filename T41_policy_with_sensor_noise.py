import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from T30_utils import generate_init_conditions, jax_differentiable_sim_noisy_sensors, label_subplots


plt.rcParams.update({'font.size': 14})

dt=0.01 # Time step for simulation in cartpole.py

#%% Policy optimization
n_initial_conditions = 50
tsteps = 300

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

if False: # If optimization is to be run
    loss_fn = lambda policy: _loss_fn(jax_differentiable_sim_noisy_sensors(init_conditions, policy, tsteps=500,max_force=20)[0]).sum()
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


stds=[0,0.05,0.25]
for std in stds:

    #Visualization
    X_sim, actions = jax_differentiable_sim_noisy_sensors(init_conditions, opt_policy, tsteps=1000,max_force=200,noise_std=std)

    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    t=np.arange(0, len(X_sim)*dt, dt)
    # Plot x position
    ax1.plot(t,X_sim[:, 0], label='Position x')
    ax1.set_ylabel('x (m)')

    # Plot theta angle
    ax2.plot(t,X_sim[:, 2], label=r'Angle $\theta$')
    ax2.set_ylabel(r'$\theta$ (rad)')

    # Plot actions
    ax3.plot(t,actions, label='Force')
    ax3.set_ylabel('Force (N)')

    # Plot loss
    losses = _loss_fn(X_sim)
    ax4.plot(t,losses, label='Loss')
    ax4.set_ylabel('Loss')

    for ax in fig.axes:
        ax.set_xlim(0.01, len(X_sim)*dt)
        ax.set_xlabel('Time (s)')
        ax.grid(True)
        ax.set_xscale('log')
    label_subplots(fig)



stds=[0,0.05,0.25]
for std in stds:

    #Visualization
    X_sim, actions = jax_differentiable_sim_noisy_sensors(init_conditions, opt_policy, tsteps=1000,max_force=200,noise_std=std)

    # Create figure with 1x4 subplots
    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(12, 4))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.15, hspace=0.3, wspace=0.75)

    t=np.arange(0, len(X_sim)*dt, dt)
    # Plot x position
    ax1.plot(t,X_sim[:, 0], label='Position x')
    ax1.set_ylabel('x (m)')

    # Plot theta angle
    ax2.plot(t,X_sim[:, 2], label=r'Angle $\theta$')
    ax2.set_ylabel(r'$\theta$ (rad)')

    # Plot actions
    ax3.plot(t,actions, label='Force')
    ax3.set_ylabel('Force (N)')

    # Plot loss
    losses = _loss_fn(X_sim)
    ax4.plot(t,losses, label='Loss')
    ax4.set_ylabel('Loss')

    for ax in fig.axes:
        ax.set_xlim(0.01, len(X_sim)*dt)
        ax.set_xlabel('Time (s)')
        ax.grid(True)
        ax.set_xscale('log')
    label_subplots(fig)


plt.show()