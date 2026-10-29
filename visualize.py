import numpy as np
import matplotlib.pyplot as plt
from ctrnn import Genome
from environment import GradientField, run_trial

field = GradientField(source=(0, 0), peak=1.0, spread=40.0)
best_vec = np.load("best_genome_v2.npy")
best = Genome(vector=best_vec)

history = np.load("fitness_history_v2.npy")

rng = np.random.default_rng(42)
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# --- Left: trajectories of evolved worm from several random starts ---
ax = axes[0]
xs = np.linspace(-100, 100, 200)
ys = np.linspace(-100, 100, 200)
X, Y = np.meshgrid(xs, ys)
C = field.peak * np.exp(-((X - field.source[0])**2 + (Y - field.source[1])**2) / (2 * field.spread**2))
ax.contourf(X, Y, C, levels=20, cmap="YlOrRd", alpha=0.6)
ax.plot(*field.source, "k*", markersize=20, label="chemical source")

colors = plt.cm.viridis(np.linspace(0, 1, 6))
for i in range(6):
    angle = rng.uniform(0, 2 * np.pi)
    radius = rng.uniform(30, 60)
    start_pos = (radius * np.cos(angle), radius * np.sin(angle))
    start_heading = rng.uniform(0, 2 * np.pi)
    traj, final_dist = run_trial(best, field, start_pos, start_heading, n_steps=1200)
    ax.plot(traj[:, 0], traj[:, 1], color=colors[i], lw=1.5)
    ax.plot(traj[0, 0], traj[0, 1], "o", color=colors[i], markersize=6)

ax.set_title("Evolved worm trajectories\n(dots = start, star = chemical source)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
ax.legend(loc="upper right")

# --- Right: fitness over generations ---
ax2 = axes[1]
gens = np.arange(len(history))
best_hist = history[:, 0]
mean_hist = history[:, 1]
ax2.plot(gens, best_hist, label="best genome", color="firebrick")
ax2.plot(gens, mean_hist, label="population mean", color="steelblue", alpha=0.7)
ax2.axhline(0, color="gray", lw=0.8, linestyle="--")
ax2.set_xlabel("generation")
ax2.set_ylabel("fitness (fractional distance reduction)")
ax2.set_title("Evolution of the klinotaxis circuit")
ax2.legend()

plt.tight_layout()
plt.savefig("chemotaxis_results.png", dpi=140)
print("saved plot")
