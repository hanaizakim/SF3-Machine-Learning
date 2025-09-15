import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from matplotlib.colors import LogNorm
from T00_jax_CartPole import CartPole
from T30_utils import plot_trajectory, rollout, generate_init_conditions, jax_differentiable_sim, label_subplots
from matplotlib.transforms import ScaledTranslation

plt.rcParams.update({'font.size': 14})

# #%% Simulate CartPole system using JAX
# # Define a set of initial conditions for the CartPole system
# n_initial_conditions = 3
# limits = jnp.array([(0,0), (-2, 2), (-np.pi/12, np.pi/12), (-5, 5)])
# init_conditions=generate_init_conditions(n_initial_conditions,limits)
# states=rollout(init_conditions, tsteps=100)
# plot_trajectory(states)

# #%% Create a  jax differentiable function to implement control policies
# # Check that jax_differentiable_sim works as expected, i.e. same trajectory as rollout when no control input is applied
# policy= jnp.array([0.0, 0.0, 0.0, 0.0]) # no control input
# X_no_control,_ = jax_differentiable_sim(init_conditions, policy, tsteps=100)
# X_roll_out = rollout(init_conditions, tsteps=100)
# print("Jax_differentiable_sim working as expected:", jnp.allclose(X_no_control, X_roll_out,atol=1e-3, rtol=1e-3))

####################################################

dt=0.01 # Time step for simulation in cartpole.py

# Policy optimization
n_initial_conditions = 50
tsteps = 300

#limits = jnp.array([(-1,1), (-1, 1), (-np.pi/12, np.pi/12), (-1, 1)])
limits = jnp.array([(-1,1), (-1, 1), (-np.pi/6, np.pi/6), (-1, 1)])
init_conditions=generate_init_conditions(n_initial_conditions,limits).T

sigmas_for_loss_function = jnp.array([2,100,0.1,100])
init_policy = jnp.zeros(4)
init_policy = jnp.array([13.40218004, 15.16610997, 98.29713825, 13.68337699]) # Last known optimal policy obtained with sigmas_for_loss_function = jnp.array([2, 100, 0.1, 100])
opt_policy = init_policy

# Loss and gradient functions
def _loss_fn(states):
    scaled_states = states / sigmas_for_loss_function[jnp.newaxis, :, jnp.newaxis]
    losses = 1 - jnp.exp(-jnp.einsum('aib,aib->ab', scaled_states, scaled_states)/(2.0))
    return losses

if False: # If optimization is to be run
    loss_fn = lambda policy: _loss_fn(jax_differentiable_sim(init_conditions, policy, tsteps=500,max_force=20)[0]).sum()
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

#Visualization
X_sim, actions = jax_differentiable_sim(init_conditions, opt_policy, tsteps=1000,max_force=200)

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


tsteps=100
angles = jnp.array([0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8])
init_cond = jnp.vstack([jnp.array([0.0,0.0,angle,0.0]) for angle in angles])
init_conditions = jnp.array(init_cond).T
forces= [1, 5, 25, 100]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.25, hspace=0.4, wspace=0.3)
t= np.arange(0, tsteps*dt, dt)
for i,force in enumerate(forces):
    X_sim, actions = jax_differentiable_sim(init_conditions, opt_policy, tsteps=tsteps,max_force=force)
    losses=_loss_fn(X_sim)

    ax=fig.axes[i]
    ax.plot(np.array(t),np.array(losses))
    ax.set_title(f'Max Force = {force} N')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Loss')
    ax.grid(True)
    ax.set_xlim(0, t[-1])
    ax.text(0.0, 1.0, f"({chr(97 + i)})", transform=ax.transAxes + ScaledTranslation(-55/72, +7/72, fig.dpi_scale_trans), fontsize=16)

legend_labels = [f'θ₀ = {angle:.2f} rad' for angle in angles]
fig.legend(legend_labels,
           loc='lower center', bbox_to_anchor=(0.5, 0.0),
           ncol=3, frameon=False, fontsize='large')
plt.show()