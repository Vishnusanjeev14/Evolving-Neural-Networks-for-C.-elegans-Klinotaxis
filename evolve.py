"""
Evolves the CTRNN klinotaxis circuit: spawn a population of random worm
brains, evaluate them by how well they climb the chemical gradient across
several trials (different starting positions/headings), cull the weak
ones, and clone+mutate the survivors. Classic (mu, lambda)-ish GA.
"""

import numpy as np
from ctrnn import Genome
from environment import GradientField, run_trials_batch

POP_SIZE = 60
N_GENERATIONS = 80
ELITE_FRAC = 0.15
N_TRIALS_PER_GENOME = 10
N_STEPS = 350


def make_trial_conditions(rng, n=N_TRIALS_PER_GENOME):
    """Random starting positions (away from source) and headings."""
    conditions = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(60, 90)
        start_pos = (radius * np.cos(angle), radius * np.sin(angle))
        start_heading = rng.uniform(0, 2 * np.pi)
        conditions.append((start_pos, start_heading))
    return conditions


def fitness(genome, field, conditions):
    """
    Fitness = average fractional reduction in distance to source
    (1.0 = ended right on top of the source, 0.0 = no progress,
    negative = moved away). Runs all trials for this genome in one
    batched (vectorized) simulation.
    """
    starts = [c[0] for c in conditions]
    headings = [c[1] for c in conditions]
    start_dists = np.linalg.norm(np.array(starts) - field.source, axis=1)
    final_dists = run_trials_batch(genome, field, starts, headings, n_steps=N_STEPS)
    scores = (start_dists - final_dists) / start_dists
    return float(np.mean(scores))


def evolve(seed=None, verbose=True):
    rng = np.random.default_rng(seed)
    field = GradientField(source=(0, 0), peak=1.0, spread=40.0)

    population = [Genome.random(rng) for _ in range(POP_SIZE)]
    history = []
    n_elite = max(2, int(POP_SIZE * ELITE_FRAC))

    # Fixed test suite for the whole run: this keeps selection pressure
    # consistent across generations (resampling conditions every
    # generation made the reported "best" fitness mostly measurement
    # noise rather than real improvement). Refreshed occasionally to
    # avoid overfitting to one specific set of starts/headings.
    conditions = make_trial_conditions(rng, n=N_TRIALS_PER_GENOME)

    for gen in range(N_GENERATIONS):
        if gen > 0 and gen % 15 == 0:
            conditions = make_trial_conditions(rng, n=N_TRIALS_PER_GENOME)
        scores = np.array([fitness(g, field, conditions) for g in population])

        order = np.argsort(-scores)
        population = [population[i] for i in order]
        scores = scores[order]

        history.append((scores.max(), scores.mean()))
        if verbose and (gen % 5 == 0 or gen == N_GENERATIONS - 1):
            print(f"gen {gen:3d}  best={scores.max():.3f}  mean={scores.mean():.3f}")

        elites = population[:n_elite]
        new_pop = list(elites)  # elitism: survivors carry over unmutated
        while len(new_pop) < POP_SIZE:
            parent = elites[rng.integers(0, n_elite)]
            child = parent.mutate(rng, sigma=0.15, rate=0.10)
            new_pop.append(child)
        population = new_pop

    return population[0], history, field


if __name__ == "__main__":
    best, history, field = evolve()
    np.save("best_genome.npy", best.vector)
    np.save("fitness_history.npy", np.array(history))
    print("Saved best genome and fitness history.")
