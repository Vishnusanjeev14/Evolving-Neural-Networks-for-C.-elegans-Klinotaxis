# Evolving Neural Networks for *C. elegans* Klinotaxis

An evolutionary neural-network simulation exploring how a continuous-time recurrent neural network (CTRNN) can learn chemotactic navigation inspired by the nematode *Caenorhabditis elegans*.

It combines computational neuroscience, evolutionary algorithms, and simulation-based evaluation to investigate how simple neural circuits can produce goal-directed behavior.

## Overview

How can a small neural circuit learn to navigate toward a chemical source without being explicitly programmed with a navigation strategy?

This project investigates that question by evolving the parameters of a six-neuron CTRNN. The simulated worm senses changes in chemical concentration over time and adjusts its movement accordingly. A genetic algorithm evaluates candidate neural circuits across multiple starting positions and headings, selecting and evolving networks that reduce their distance to the source.

Version 2 extends the original implementation with improved evolutionary training, held-out validation, resumable training, benchmarking, and robustness analysis.

## Key Features

- **Neural circuit simulation:** A six-neuron continuous-time recurrent neural network inspired by the sensory, interneuron, and motor organization of *C. elegans*.
- **Evolutionary optimization:** Population-based selection, elitism, mutation, and uniform crossover.
- **Chemotaxis environment:** A two-dimensional Gaussian chemical concentration field.
- **Generalization testing:** Evaluation on a fixed set of 40 held-out starting conditions that are not used for evolutionary selection.
- **Resumable experiments:** Checkpointed training state for continuing interrupted evolutionary runs.
- **Baseline benchmarking:** Comparison against randomly initialized neural networks and a hand-coded klinokinesis controller.
- **Robustness evaluation:** Tests under sensory noise, different chemical gradient widths, and unseen starting distances.
- **Research visualizations:** Trajectory plots, learning curves, benchmark comparisons, and robustness figures.

## How It Works

The simulation follows four main stages:

1. **Environment initialization:** A chemical source is placed at the center of a two-dimensional Gaussian concentration field.
2. **Neural processing:** The worm's sensory neurons respond to rising or falling chemical concentration. Interneurons and motor neurons transform these signals into turning behavior.
3. **Evolution:** Candidate neural networks are evaluated using their average fractional reduction in distance to the source. Better-performing networks are retained and used to generate the next population.
4. **Evaluation:** The evolved network is tested on held-out conditions and compared with baseline controllers.

### Neural Network Architecture

The CTRNN contains six neurons:

| Neuron | Functional role |
|---|---|
| ASEL | Sensory response to increasing concentration |
| ASER | Sensory response to decreasing concentration |
| AIY | Interneuron |
| AIZ | Interneuron |
| SMB_D | Motor neuron associated with dorsal turning |
| SMB_V | Motor neuron associated with ventral turning |

The evolutionary algorithm optimizes the network parameters rather than directly programming a path toward the source.

### Fitness Function

The fitness score measures the fractional reduction in distance to the chemical source:

\[
F = \frac{d_{\mathrm{start}}-d_{\mathrm{final}}}{d_{\mathrm{start}}}
\]

- \(F=1\): The worm finishes at the source.
- \(F=0\): There is no net reduction in distance.
- \(F<0\): The worm finishes farther away than it started.

The fitness used during evolution is averaged over multiple trials to reduce dependence on any single starting condition.

## Version 2: Experimental Improvements

| Parameter | Value |
|---|---:|
| Population size | 60 |
| Evolutionary generations | 120 |
| Trials per genome | 24 |
| Simulation steps per trial | 1,200 |
| Elite fraction | 15% |
| Crossover probability | 50% |
| Held-out validation trials | 40 |
| Validation interval | Every 5 generations |
| Training starting distances | 30–60 units |

Training conditions are resampled each generation, while the held-out validation conditions remain fixed and are excluded from selection.

## Reported Results

The included results report the following performance on 40 held-out starting conditions:

| Metric | Reported result |
|---|---:|
| Evolved CTRNN fitness | 0.741 |
| Hand-coded baseline fitness | 0.339 |
| Random CTRNN fitness | 0.002 |
| Trials that reduced their distance to the source | 40/40 (100%) |
| Trials finishing within 20 units of the source | 36/40 (90%) |
| Mean starting distance | 44.8 units |
| Mean final distance | 12.0 units |

The reported evolved CTRNN fitness exceeds the hand-coded baseline by 0.402 when calculated from the displayed rounded scores.

These values are from the included experiment results and are not a guarantee that a fresh training run will reproduce the same outcome. The results describe simulated behavior rather than a physical experiment with living worms.

## Visualizations

The project includes four research figures:

| Figure | Description |
|---|---|
| `fig1_trajectories.png` | Simulated worm trajectories toward the chemical source |
| `fig2_learning.png` | Training and held-out validation performance across generations |
| `fig3_benchmark.png` | Comparison with random and hand-coded controllers |
| `fig4_robustness.png` | Performance under sensory noise and altered environmental conditions |

An additional visualization, `chemotaxis_results.png`, illustrates worm trajectories and evolutionary fitness.

## Project Structure

```text
worm/
├── ctrnn.py
├── environment.py
├── evolve.py
├── evolve_v2.py
├── analyze.py
├── visualize.py
├── best_genome.npy
├── best_genome_v2.npy
├── fitness_history.npy
├── fitness_history_v2.npy
├── val_history_v2.npy
├── v2_state.npz
├── results_table.txt
├── chemotaxis_results.png
├── fig1_trajectories.png
├── fig2_learning.png
├── fig3_benchmark.png
└── fig4_robustness.png
```

**Core modules**

- `ctrnn.py` — Implements the neural network, genome representation, and worm nervous-system dynamics.
- `environment.py` — Implements the chemical gradient, movement, and trial simulation.
- `evolve.py` — Contains the original evolutionary algorithm.
- `evolve_v2.py` — Implements the extended evolutionary algorithm with crossover, validation, and checkpointing.
- `analyze.py` — Produces performance summaries, benchmarks, and robustness figures.
- `visualize.py` — Generates a combined trajectory and learning-curve visualization.

The `.npy` and `.npz` files store trained parameters, fitness histories, validation data, and checkpoint state.

## Installation

**Requirements:** Python 3.10 or later, NumPy, and Matplotlib.


Install dependencies:

```bash
python -m pip install numpy matplotlib
```

## Running the Project

### 1. Visualize the saved V2 model

The archive includes a trained genome and its fitness history.

```bash
python visualize.py
```

This generates `chemotaxis_results.png`.

### 2. Run the analysis

```bash
python analyze.py
```

This evaluates the saved model and produces the four research figures and `results_table.txt`. Run it from the directory containing the required V2 files.

### 3. Train the V2 model

```bash
python evolve_v2.py 40
```

The argument specifies the number of generations to run in one chunk. The default is 40, and the checkpoint allows subsequent runs to resume the evolutionary process.

Run the command again to continue training until all 120 generations have completed. Once training finishes, the script saves the evolved genome and the training and validation histories.

**Note:** Running training can take substantially longer than visualization or analysis.

## Scientific Scope and Limitations

- The neural circuit is inspired by *C. elegans* biology but is a simplified computational model.
- Movement occurs in a simulated two-dimensional environment with a predefined chemical field.
- Successful navigation in simulation does not establish that the evolved circuit reproduces the complete biological mechanisms of real worms.
- The held-out evaluation measures generalization to a fixed collection of starting conditions. Broader claims would benefit from additional independent seeds, repeated evolutionary runs, and statistical uncertainty estimates.
- Comparisons between the original and V2 implementations should use identical evaluation conditions and metrics.

## Future Improvements

- Evaluate performance across multiple independent evolutionary seeds.
- Add confidence intervals and statistical comparisons.
- Test more complex chemical environments and multiple chemical sources.
- Investigate alternative evolutionary strategies and network architectures.
- Compare evolved controllers with additional biologically motivated models.
- Explore transfer to embodied or physics-based simulations.

## Applications

Provides a compact experimental framework for exploring:

- Neuroevolution and artificial life
- Computational neuroscience
- Emergent navigation behavior
- Evolutionary robotics
- Robustness and generalization in neural controllers

