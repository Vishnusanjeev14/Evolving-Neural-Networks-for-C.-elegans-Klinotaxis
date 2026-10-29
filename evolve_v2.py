"""
Review-3 version of the klinotaxis evolution loop.

Fixes four problems found in the pre-Review-3 evolve.py:

  PROBLEM 1 - OVERFITTING / FITNESS COLLAPSE.
      The old code held trial conditions fixed for 15-generation blocks.
      Best fitness climbed then collapsed every time conditions refreshed.
      FIX: conditions are resampled EVERY generation.

  PROBLEM 2 - NO VALIDATION.
      Nothing measured generalisation, so the headline number was
      meaningless.
      FIX: a fixed held-out set of 40 start conditions, never used for
      selection, evaluated every 5 generations. This is the number we
      report.

  PROBLEM 3 - NOISY FITNESS ESTIMATE.
      10 trials/genome made per-generation fitness mostly noise.
      FIX: 24 trials/genome.

  PROBLEM 4 - WORMS COULD NOT PHYSICALLY REACH THE SOURCE.
      350 steps at speed 1.0 = 35 units of travel, but start distance was
      60-90 units. Arrival was impossible regardless of circuit quality.
      FIX: 1200 steps, start distance 30-60.

Also adds uniform crossover (old code used mutation only).

Depends on the ORIGINAL ctrnn.py and environment.py - unchanged.
"""

import numpy as np
from ctrnn import Genome
from environment import GradientField, run_trials_batch

POP_SIZE = 60
N_GENERATIONS = 120
ELITE_FRAC = 0.15
N_TRIALS_PER_GENOME = 24      # was 10
N_STEPS = 1200                 # was 350
CROSSOVER_RATE = 0.5
VALIDATE_EVERY = 5
N_HELDOUT = 40
HELDOUT_SEED = 12345            # fixed forever, never used for selection


def make_trial_conditions(rng, n=N_TRIALS_PER_GENOME):
    """Random starting positions (reachable range) and headings."""
    conditions = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(30, 60)     # was 60-90
        start_pos = (radius * np.cos(angle), radius * np.sin(angle))
        start_heading = rng.uniform(0, 2 * np.pi)
        conditions.append((start_pos, start_heading))
    return conditions


def fitness(genome, field, conditions, n_steps=N_STEPS):
    """
    Fitness = mean fractional reduction in distance to the source.
        1.0  = ended exactly on the source
        0.0  = no net progress
       <0.0  = moved away
    """
    starts = [c[0] for c in conditions]
    headings = [c[1] for c in conditions]
    start_dists = np.linalg.norm(np.array(starts) - field.source, axis=1)
    final_dists = run_trials_batch(genome, field, starts, headings, n_steps=n_steps)
    scores = (start_dists - final_dists) / start_dists
    return float(np.mean(scores))


def crossover(parent_a, parent_b, rng):
    """Uniform crossover on the flat genome vector."""
    m = rng.random(len(parent_a.vector)) < 0.5
    child_vec = np.where(m, parent_a.vector, parent_b.vector)
    return Genome(vector=child_vec)


def run_chunk(state_path="v2_state.npz", chunk=40, seed=0, verbose=True):
    """Chunked/resumable driver - run this repeatedly, it saves progress
    after every chunk so you never lose a run."""
    import os
    rng = np.random.default_rng(seed)
    field = GradientField(source=(0, 0), peak=1.0, spread=40.0)
    heldout = make_trial_conditions(np.random.default_rng(HELDOUT_SEED), n=N_HELDOUT)

    if os.path.exists(state_path):
        st = np.load(state_path, allow_pickle=True)
        rng = np.random.default_rng()
        rng.bit_generator.state = st["rng_state"].item()
        population = [Genome(vector=v) for v in st["pop"]]
        history = list(map(tuple, st["history"]))
        val_history = list(map(tuple, st["val_history"])) if len(st["val_history"]) else []
        start_gen = int(st["gen"]) + 1
        if verbose:
            print(f"resuming from gen {start_gen}")
    else:
        population = [Genome.random(rng) for _ in range(POP_SIZE)]
        history, val_history, start_gen = [], [], 0

    n_elite = max(2, int(POP_SIZE * ELITE_FRAC))
    end_gen = min(start_gen + chunk, N_GENERATIONS)

    for gen in range(start_gen, end_gen):
        conditions = make_trial_conditions(rng, n=N_TRIALS_PER_GENOME)   # FIX 1: fresh every gen
        scores = np.array([fitness(g, field, conditions) for g in population])
        order = np.argsort(-scores)
        population = [population[i] for i in order]
        scores = scores[order]
        history.append((scores.max(), scores.mean()))

        if gen % VALIDATE_EVERY == 0 or gen == N_GENERATIONS - 1:
            val = fitness(population[0], field, heldout)               # FIX 2
            val_history.append((gen, val))
            if verbose:
                print(f"gen {gen:3d}  train_best={scores.max():.3f}  "
                      f"train_mean={scores.mean():.3f}  HELD-OUT={val:.3f}")

        elites = population[:n_elite]
        new_pop = list(elites)
        while len(new_pop) < POP_SIZE:
            pa = elites[rng.integers(0, n_elite)]
            if rng.random() < CROSSOVER_RATE:
                pb = elites[rng.integers(0, n_elite)]
                child = crossover(pa, pb, rng).mutate(rng, sigma=0.15, rate=0.10)
            else:
                child = pa.mutate(rng, sigma=0.15, rate=0.10)
            new_pop.append(child)
        population = new_pop

    np.savez(state_path,
             pop=np.array([g.vector for g in population]),
             rng_state=np.array(rng.bit_generator.state, dtype=object),
             history=np.array(history),
             val_history=np.array(val_history) if val_history else np.zeros((0, 2)),
             gen=end_gen - 1)

    done = end_gen >= N_GENERATIONS
    if done:
        np.save("best_genome_v2.npy", population[0].vector)
        np.save("fitness_history_v2.npy", np.array(history))
        np.save("val_history_v2.npy", np.array(val_history))
        print("DONE - saved best_genome_v2.npy, fitness_history_v2.npy, val_history_v2.npy")
    return done


if __name__ == "__main__":
    import sys
    c = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    run_chunk(chunk=c)