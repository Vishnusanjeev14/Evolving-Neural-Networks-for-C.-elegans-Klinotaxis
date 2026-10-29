"""
Review-3 results & analysis.

Run AFTER evolve_v2.py has printed DONE.

Produces:
  fig1_trajectories.png   - evolved worm paths up the gradient
  fig2_learning.png       - training vs held-out fitness across generations
  fig3_benchmark.png      - evolved circuit vs hand-coded baseline vs random
  fig4_robustness.png     - performance under sensory noise / altered gradients
  results_table.txt       - the numbers, for the report

Depends on the ORIGINAL ctrnn.py / environment.py (unchanged) plus
evolve_v2.py. Uses MOTOR = [4, 5] to match ctrnn.py exactly.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ctrnn import Genome, N_NEURONS, MOTOR
from environment import GradientField, run_trial, run_trials_batch, SPEED, MAX_TURN_RATE
from evolve_v2 import make_trial_conditions, fitness, N_STEPS, HELDOUT_SEED, N_HELDOUT

LIGHT_BG = "#FFFFFF"
field = GradientField(source=(0, 0), peak=1.0, spread=40.0)
best = Genome(vector=np.load("best_genome_v2.npy"))
history = np.load("fitness_history_v2.npy")
val_history = np.load("val_history_v2.npy")

heldout = make_trial_conditions(np.random.default_rng(HELDOUT_SEED), n=N_HELDOUT)
lines = []


def log(s):
    print(s)
    lines.append(s)


# ---------------------------------------------------------------- baselines
def klinokinesis_fitness(conditions, n_steps=N_STEPS):
    """Hand-coded reference controller: turn while concentration falls,
    go straight while it rises. Gives the evolved circuit something
    meaningful to be compared against."""
    dt = 0.1
    scores = []
    for (sp, sh) in conditions:
        pos = np.array(sp, float)
        h = sh
        prev = field.concentration(pos)
        for _ in range(n_steps):
            c = field.concentration(pos)
            dc = (c - prev) / dt
            prev = c
            h += (1.0 if dc < -1e-6 else 0.0) * MAX_TURN_RATE
            pos = pos + SPEED * dt * np.array([np.cos(h), np.sin(h)])
        r0 = np.linalg.norm(np.array(sp) - field.source)
        scores.append((r0 - np.linalg.norm(pos - field.source)) / r0)
    return float(np.mean(scores))


def random_fitness(conditions, n=15):
    rng = np.random.default_rng(7)
    return float(np.mean([fitness(Genome.random(rng), field, conditions) for _ in range(n)]))


evolved = fitness(best, field, heldout)
hand = klinokinesis_fitness(heldout)
rand = random_fitness(heldout)

log("=" * 62)
log("RESULT 1 - HELD-OUT PERFORMANCE (40 unseen start conditions)")
log("=" * 62)
log(f"  Random untrained CTRNN   : {rand: .3f}")
log(f"  Hand-coded klinokinesis  : {hand: .3f}")
log(f"  EVOLVED CTRNN (ours)     : {evolved: .3f}")
log(f"  Improvement over baseline: {evolved - hand:+.3f}")
log("")

# ------------------------------------------------- fig 1: trajectories
rng = np.random.default_rng(3)
fig, ax = plt.subplots(figsize=(7, 6.5), facecolor=LIGHT_BG)
ax.set_facecolor(LIGHT_BG)
xs = np.linspace(-80, 80, 250)
X, Y = np.meshgrid(xs, xs)
C = field.peak * np.exp(-(X**2 + Y**2) / (2 * field.spread**2))
ax.contourf(X, Y, C, levels=25, cmap="YlOrRd", alpha=0.55)
ax.plot(0, 0, "k*", markersize=22, label="chemical source", zorder=5)

colors = plt.cm.viridis(np.linspace(0, 0.9, 8))
for i in range(8):
    a = rng.uniform(0, 2 * np.pi)
    r = rng.uniform(30, 60)
    sp = (r * np.cos(a), r * np.sin(a))
    sh = rng.uniform(0, 2 * np.pi)
    traj, fd = run_trial(best, field, sp, sh, n_steps=N_STEPS)
    ax.plot(traj[:, 0], traj[:, 1], color=colors[i], lw=1.6, alpha=0.9)
    ax.plot(traj[0, 0], traj[0, 1], "o", color=colors[i], markersize=8,
            markeredgecolor="k", markeredgewidth=0.6)
ax.set_title("Evolved klinotaxis: worm trajectories\n(circles = start, star = source)",
             fontsize=12)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
ax.legend(loc="upper right", framealpha=0.95)
plt.tight_layout()
plt.savefig("fig1_trajectories.png", dpi=150, facecolor=LIGHT_BG)
plt.close()

# arrival statistics over the full held-out set
starts = [c[0] for c in heldout]
headings = [c[1] for c in heldout]
r0 = np.linalg.norm(np.array(starts) - field.source, axis=1)
fd = run_trials_batch(best, field, starts, headings, n_steps=N_STEPS)
log("=" * 62)
log("RESULT 2 - ARRIVAL STATISTICS (40 held-out trials)")
log("=" * 62)
log(f"  Mean start distance      : {r0.mean():.1f}")
log(f"  Mean final distance      : {fd.mean():.1f}")
log(f"  Trials ending within 10u : {int((fd < 10).sum())}/40  ({100*(fd<10).mean():.0f}%)")
log(f"  Trials ending within 20u : {int((fd < 20).sum())}/40  ({100*(fd<20).mean():.0f}%)")
log(f"  Trials that got closer   : {int((fd < r0).sum())}/40  ({100*(fd<r0).mean():.0f}%)")
log("")

# ------------------------------------------------- fig 2: learning curve
fig, ax = plt.subplots(figsize=(8, 5), facecolor=LIGHT_BG)
ax.set_facecolor(LIGHT_BG)
gens = np.arange(len(history))
ax.plot(gens, history[:, 0], color="#C0392B", alpha=0.45, lw=1, label="training best (per-gen)")
ax.plot(gens, history[:, 1], color="#5DADE2", alpha=0.5, lw=1, label="population mean")
ax.plot(val_history[:, 0], val_history[:, 1], color="#1A5276", marker="o",
        markersize=5, lw=2.5, label="HELD-OUT (unseen conditions)")
ax.axhline(hand, color="#7D3C98", ls="--", lw=1.8, label=f"hand-coded baseline ({hand:.2f})")
ax.axhline(0, color="gray", ls=":", lw=1)
ax.set_xlabel("generation")
ax.set_ylabel("fitness (fractional distance reduction)")
ax.set_title(f"Learning curve: held-out fitness rises to {evolved:.2f}", fontsize=12)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("fig2_learning.png", dpi=150, facecolor=LIGHT_BG)
plt.close()

# ------------------------------------------------- fig 3: benchmark bars
fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=LIGHT_BG)
ax.set_facecolor(LIGHT_BG)
names = ["Random\nCTRNN", "Hand-coded\nklinokinesis", "Evolved CTRNN\n(this work)"]
vals = [rand, hand, evolved]
bars = ax.bar(names, vals, color=["#AEB6BF", "#A569BD", "#2E86C1"], width=0.55,
              edgecolor="black", linewidth=0.7)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
            ha="center", fontsize=11, fontweight="bold")
ax.axhline(0, color="gray", lw=1)
ax.set_ylabel("held-out fitness")
ax.set_title("Evolved circuit vs baselines", fontsize=12)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig("fig3_benchmark.png", dpi=150, facecolor=LIGHT_BG)
plt.close()

# ------------------------------------------------- fig 4: robustness
log("=" * 62)
log("RESULT 3 - ROBUSTNESS (generalisation beyond training conditions)")
log("=" * 62)

# (a) sensory noise
def noisy_fitness(genome, conditions, noise, n_steps=N_STEPS, seed=1):
    rng = np.random.default_rng(seed)
    dt = 0.1
    g = genome
    B = len(conditions)
    pos = np.array([c[0] for c in conditions], float)
    heading = np.array([c[1] for c in conditions], float)
    y = np.zeros((B, N_NEURONS))
    r0 = np.linalg.norm(pos - field.source, axis=1)
    prev = field.peak * np.exp(-np.sum(pos**2, axis=1) / (2 * field.spread**2))
    for _ in range(n_steps):
        c = field.peak * np.exp(-np.sum(pos**2, axis=1) / (2 * field.spread**2))
        dc = (c - prev) / dt
        prev = c
        dc = dc + rng.normal(0, noise, B)
        I = np.zeros((B, N_NEURONS))
        I[:, 0] = g.gain[0] * np.clip(dc, 0, None)
        I[:, 1] = g.gain[1] * np.clip(-dc, 0, None)
        z = 1 / (1 + np.exp(-(y + g.bias)))
        y = y + dt * ((-y + z @ g.W + I) / g.tau)
        zo = 1 / (1 + np.exp(-(y + g.bias)))
        turn = np.tanh(2.0 * (zo[:, MOTOR[0]] - zo[:, MOTOR[1]]))
        heading = heading + turn * MAX_TURN_RATE
        pos = pos + SPEED * dt * np.stack([np.cos(heading), np.sin(heading)], 1)
    fdist = np.linalg.norm(pos - field.source, axis=1)
    return float(np.mean((r0 - fdist) / r0))

noises = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
noise_scores = [noisy_fitness(best, heldout, nz) for nz in noises]
log("  (a) Sensory noise on dC/dt:")
for nz, s in zip(noises, noise_scores):
    log(f"        noise sd={nz:<6} -> fitness {s: .3f}")

# (b) unseen gradient spreads
spreads = [20, 30, 40, 50, 70]
spread_scores = []
for sp_ in spreads:
    f2 = GradientField(source=(0, 0), peak=1.0, spread=sp_)
    starts = [c[0] for c in heldout]
    hd = [c[1] for c in heldout]
    rr = np.linalg.norm(np.array(starts) - f2.source, axis=1)
    ff = run_trials_batch(best, f2, starts, hd, n_steps=N_STEPS)
    spread_scores.append(float(np.mean((rr - ff) / rr)))
log("  (b) Unseen gradient widths (trained only on spread=40):")
for sp_, s in zip(spreads, spread_scores):
    tag = "  <- trained" if sp_ == 40 else ""
    log(f"        spread={sp_:<4} -> fitness {s: .3f}{tag}")

# (c) unseen start distances
dists = [(30, 60), (60, 90), (90, 120)]
dist_scores = []
for (lo, hi) in dists:
    rng2 = np.random.default_rng(99)
    conds = []
    for _ in range(40):
        a = rng2.uniform(0, 2*np.pi); r = rng2.uniform(lo, hi)
        conds.append(((r*np.cos(a), r*np.sin(a)), rng2.uniform(0, 2*np.pi)))
    dist_scores.append(fitness(best, field, conds))
log("  (c) Unseen start distances (trained only on 30-60):")
for (lo, hi), s in zip(dists, dist_scores):
    tag = "  <- trained" if lo == 30 else ""
    log(f"        start r={lo}-{hi:<4} -> fitness {s: .3f}{tag}")
log("")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), facecolor=LIGHT_BG)
for a in axes:
    a.set_facecolor(LIGHT_BG)
    a.grid(alpha=0.25)
axes[0].plot(noises, noise_scores, "o-", color="#2E86C1", lw=2)
axes[0].set_xlabel("sensory noise (sd)"); axes[0].set_ylabel("fitness")
axes[0].set_title("(a) Robustness to sensory noise")
axes[1].bar([str(s) for s in spreads], spread_scores, color="#48C9B0",
            edgecolor="black", linewidth=0.6)
axes[1].set_xlabel("gradient spread"); axes[1].set_title("(b) Unseen gradient widths")
axes[2].bar([f"{lo}-{hi}" for lo, hi in dists], dist_scores, color="#F5B041",
            edgecolor="black", linewidth=0.6)
axes[2].set_xlabel("start distance"); axes[2].set_title("(c) Unseen start distances")
plt.tight_layout()
plt.savefig("fig4_robustness.png", dpi=150, facecolor=LIGHT_BG)
plt.close()

log("=" * 62)
log("FIGURES WRITTEN: fig1_trajectories.png, fig2_learning.png,")
log("                 fig3_benchmark.png, fig4_robustness.png")
log("=" * 62)

open("results_table.txt", "w").write("\n".join(lines))
print("\nwrote results_table.txt")