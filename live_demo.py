"""
Live demo of the evolved (v2) worm. Run from the worm folder:   python live_demo.py
Optional:  python live_demo.py --save demo.gif   (writes a GIF instead of opening a window)
Uses best_genome_v2.npy with the original ctrnn.py / environment.py. Nothing is retrained.
"""
import sys
import numpy as np
import matplotlib
if "--save" in sys.argv: matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ctrnn import Genome, Worm
from environment import GradientField, SPEED, MAX_TURN_RATE

DT, N_STEPS, PER_FRAME, N_WORMS = 0.1, 1200, 8, 5
BG, INK = "#FAF5E9", "#1F3A4D"
field = GradientField(source=(0, 0), peak=1.0, spread=40.0)
genome = Genome(vector=np.load("best_genome_v2.npy"))
rng = np.random.default_rng(7)

def simulate(start, heading):
    w = Worm(genome, dt=DT); pos = np.array(start, float); prev = field.concentration(pos)
    P, C, Z = [pos.copy()], [prev], [np.full(6, 0.5)]
    for _ in range(N_STEPS):
        c = field.concentration(pos); dc = (c - prev) / DT; prev = c
        turn = w.step(dc); heading += turn * MAX_TURN_RATE
        pos = pos + SPEED * DT * np.array([np.cos(heading), np.sin(heading)])
        P.append(pos.copy()); C.append(c); Z.append(1 / (1 + np.exp(-(w.y + genome.bias))))
    return np.array(P), np.array(C), np.array(Z)

runs = []
for i in range(N_WORMS):
    a, r = rng.uniform(0, 2*np.pi), rng.uniform(30, 60)
    runs.append(simulate((r*np.cos(a), r*np.sin(a)), rng.uniform(0, 2*np.pi)))
cols = plt.cm.tab10(np.arange(N_WORMS))

fig = plt.figure(figsize=(13, 6.5), facecolor=BG)
ax = fig.add_axes([0.04, 0.08, 0.46, 0.84], facecolor=BG)
xs = np.linspace(-80, 80, 200); X, Y = np.meshgrid(xs, xs)
ax.contourf(X, Y, np.exp(-(X**2+Y**2)/(2*40.0**2)), 20, cmap="YlOrRd", alpha=0.55)
ax.plot(0, 0, "k*", ms=18); ax.set_aspect("equal"); ax.set_xlim(-80, 80); ax.set_ylim(-80, 80)
ax.set_title("Evolved worms climbing the chemical gradient (star = source)", color=INK)
trails = [ax.plot([], [], color=c, lw=1.6)[0] for c in cols]
heads = [ax.plot([], [], "o", color=c, ms=8, mec="k")[0] for c in cols]
ax_c = fig.add_axes([0.57, 0.60, 0.40, 0.30], facecolor="white")
ax_n = fig.add_axes([0.57, 0.12, 0.40, 0.38], facecolor="white")
ax_c.set_title("Worm 1: chemical concentration at its nose", color=INK, fontsize=11)
ax_n.set_title("Worm 1: neuron activity  (sensory ASEL/ASER, motor SMB_D/SMB_V)", color=INK, fontsize=11)
ax_c.set_xlim(0, N_STEPS); ax_c.set_ylim(0, 1.05); ax_n.set_xlim(0, N_STEPS); ax_n.set_ylim(-0.02, 1.02)
ax_n.set_xlabel("timestep", color=INK)
lc, = ax_c.plot([], [], color="#B5532E", lw=1.5)
names = [("ASEL (rising)", 0, "#1B7A80"), ("ASER (falling)", 1, "#B87613"), ("SMB_D", 4, "#2F7A4A"), ("SMB_V", 5, "#7A3F8F")]
ln = [ax_n.plot([], [], lw=1.3, label=n, color=c)[0] for n, _, c in names]
ax_n.legend(loc="upper right", fontsize=9, ncol=2)
txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", color=INK, fontsize=11)

def update(f):
    t = min((f + 1) * PER_FRAME, N_STEPS)
    for i, (P, C, Z) in enumerate(runs):
        trails[i].set_data(P[:t+1, 0], P[:t+1, 1]); heads[i].set_data([P[t, 0]], [P[t, 1]])
    P, C, Z = runs[0]
    lc.set_data(range(t+1), C[:t+1])
    for l, (_, k, _) in zip(ln, names): l.set_data(range(t+1), Z[:t+1, k])
    d = np.mean([np.linalg.norm(r[0][t]) for r in runs])
    txt.set_text(f"step {t}/{N_STEPS}   mean distance to source: {d:.1f}")
    return trails + heads + [lc] + ln + [txt]

frames = N_STEPS // PER_FRAME
anim = FuncAnimation(fig, update, frames=frames, interval=30, blit=False, repeat=False)
if "--save" in sys.argv:
    out = sys.argv[sys.argv.index("--save") + 1]
    anim.save(out, writer="pillow", fps=25, dpi=60); print("saved", out)
else:
    plt.show()