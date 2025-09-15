import numpy as np
import matplotlib.pyplot as plt
from T20_utils import simulate_rollout, plot_rollout

plt.rcParams.update({'font.size': 14})

# Create colors to cycle through
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # get default color cycle

forces=[20,200,2000]
fig_rollout, fig_phase, fig_report = None, None, None
for i, force in enumerate(forces):
    initial_state = [0.0, 0.0, np.radians(180), 0.0]  # Only state variables
    time, states = simulate_rollout(initial_state, F=force, Fmax=20+10*force, steps=80)
    label = f"$F$ = {force} N"
    fig_rollout = plot_rollout(time, states, fig=fig_rollout, color=colors[i % 10], label=label)
plt.show()

