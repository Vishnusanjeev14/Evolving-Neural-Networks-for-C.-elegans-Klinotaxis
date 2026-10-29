"""
2D environment for the C. elegans chemotaxis simulation.

The chemical field is a Gaussian centered at a source point, similar to
the NaCl radial gradient assays used in real C. elegans chemotaxis
experiments (e.g. Pierce-Shimomura et al. 1999).
"""

import numpy as np
from ctrnn import Worm

SPEED = 1.0          # constant forward crawl speed (arbitrary units/timestep)
MAX_TURN_RATE = 0.15  # radians/timestep, caps how sharply the worm can turn


class GradientField:
    def __init__(self, source=(0.0, 0.0), peak=1.0, spread=40.0):
        self.source = np.array(source, dtype=np.float64)
        self.peak = peak
        self.spread = spread

    def concentration(self, pos):
        d2 = np.sum((np.asarray(pos) - self.source) ** 2)
        return self.peak * np.exp(-d2 / (2 * self.spread ** 2))


def run_trial(genome, field, start_pos, start_heading, n_steps=600, dt=0.1):
    """
    Simulate one worm from start_pos/start_heading for n_steps.
    Returns trajectory (n_steps+1, 2) and final distance to source.
    """
    worm = Worm(genome, dt=dt)
    pos = np.array(start_pos, dtype=np.float64)
    heading = start_heading
    traj = np.zeros((n_steps + 1, 2))
    traj[0] = pos

    prev_c = field.concentration(pos)
    for t in range(n_steps):
        c = field.concentration(pos)
        dc = (c - prev_c) / dt
        prev_c = c

        turn = worm.step(dc)
        heading += turn * MAX_TURN_RATE
        pos = pos + SPEED * dt * np.array([np.cos(heading), np.sin(heading)])
        traj[t + 1] = pos

    final_dist = np.linalg.norm(pos - field.source)
    return traj, final_dist


def run_trials_batch(genome, field, start_positions, start_headings, n_steps=500, dt=0.1):
    """
    Vectorized version: simulates B trials of the SAME genome in parallel
    (different starting positions/headings) as one batched numpy
    computation instead of a Python loop per trial. Much faster for
    evolution, where each genome needs several trials for a stable
    fitness estimate.

    Returns final distances to source, shape (B,).
    """
    g = genome
    B = len(start_positions)
    pos = np.array(start_positions, dtype=np.float64)          # (B,2)
    heading = np.array(start_headings, dtype=np.float64)        # (B,)
    y = np.zeros((B, 6))                                        # (B,6) neuron states

    d2_0 = np.sum((pos - field.source) ** 2, axis=1)
    prev_c = field.peak * np.exp(-d2_0 / (2 * field.spread ** 2))

    for t in range(n_steps):
        d2 = np.sum((pos - field.source) ** 2, axis=1)
        c = field.peak * np.exp(-d2 / (2 * field.spread ** 2))
        dc = (c - prev_c) / dt
        prev_c = c

        I = np.zeros((B, 6))
        I[:, 0] = g.gain[0] * np.clip(dc, 0, None)
        I[:, 1] = g.gain[1] * np.clip(-dc, 0, None)

        z = 1.0 / (1.0 + np.exp(-(y + g.bias)))       # (B,6)
        dydt = (-y + z @ g.W + I) / g.tau
        y = y + dt * dydt

        z_out = 1.0 / (1.0 + np.exp(-(y + g.bias)))
        motor_diff = z_out[:, 4] - z_out[:, 5]
        turn = np.tanh(2.0 * motor_diff)

        heading = heading + turn * MAX_TURN_RATE
        pos = pos + SPEED * dt * np.stack([np.cos(heading), np.sin(heading)], axis=1)

    final_dist = np.linalg.norm(pos - field.source, axis=1)
    return final_dist
