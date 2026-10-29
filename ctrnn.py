"""
Minimal CTRNN (Continuous-Time Recurrent Neural Network) implementation
for a 6-neuron C. elegans klinotaxis circuit.

Neurons (indices):
    0: ASEL  (sensory - detects rising concentration)
    1: ASER  (sensory - detects falling concentration)
    2: AIY   (interneuron)
    3: AIZ   (interneuron)
    4: SMB_D (motor - dorsal turning bias)
    5: SMB_V (motor - ventral turning bias)

Standard CTRNN dynamics (Beer 1995 / used by Izquierdo & Lockery 2010
for evolved C. elegans chemotaxis circuits):

    tau_i * dy_i/dt = -y_i + sum_j( w_ji * sigma(y_j + bias_j) ) + I_i(t)

where sigma is the logistic function. Only sensory neurons receive
external input I_i(t).
"""

import numpy as np

N_NEURONS = 6
SENSORY = [0, 1]      # ASEL, ASER
MOTOR = [4, 5]         # SMB_D, SMB_V


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Genome:
    """
    Encodes all tunable parameters of the circuit as a flat vector,
    so a genetic algorithm can evolve it directly.

    Parameters:
        W    : (6,6) weight matrix (fully recurrent, self-connections allowed)
        bias : (6,)  neuron biases
        tau  : (6,)  time constants (bounded to a sane biological range)
        gain : (2,)  sensory gain for ASEL/ASER
    """
    W_SHAPE = (N_NEURONS, N_NEURONS)
    N_PARAMS = N_NEURONS * N_NEURONS + N_NEURONS + N_NEURONS + 2

    def __init__(self, vector=None, rng=None):
        rng = rng or np.random.default_rng()
        if vector is None:
            vector = rng.uniform(-1, 1, self.N_PARAMS)
        self.vector = np.asarray(vector, dtype=np.float64)
        self._unpack()

    def _unpack(self):
        v = self.vector
        i = 0
        self.W = v[i:i + 36].reshape(6, 6) * 4.0          # weights in [-4,4]
        i += 36
        self.bias = v[i:i + 6] * 3.0                       # biases in [-3,3]
        i += 6
        tau_raw = v[i:i + 6]                                 # in [-1,1]
        i += 6
        self.tau = 0.1 + (tau_raw + 1) / 2 * 2.4             # tau in [0.1, 2.5]
        gain_raw = v[i:i + 2]
        self.gain = 1.0 + (gain_raw + 1) / 2 * 4.0           # gain in [1, 5]

    @classmethod
    def random(cls, rng=None):
        return cls(vector=None, rng=rng)

    def mutate(self, rng, sigma=0.2, rate=0.15):
        child_vec = self.vector.copy()
        mask = rng.random(len(child_vec)) < rate
        child_vec[mask] += rng.normal(0, sigma, mask.sum())
        child_vec = np.clip(child_vec, -1, 1)
        return Genome(vector=child_vec)


class Worm:
    """Simulates one worm's nervous system + body for one trial."""

    def __init__(self, genome: Genome, dt=0.1):
        self.genome = genome
        self.dt = dt
        self.y = np.zeros(N_NEURONS)  # neuron activation states

    def reset(self):
        self.y[:] = 0.0

    def step(self, d_concentration):
        """
        d_concentration: scalar, rate of change of chemical concentration
        sensed at the worm's nose over the last timestep (real C. elegans
        klinotaxis is driven by this temporal derivative, not a spatial
        gradient across the body).
        Returns: turning signal in [-1, 1] (positive = turn one way).
        """
        g = self.genome
        I = np.zeros(N_NEURONS)
        # ASEL fires more when concentration is rising, ASER when falling
        I[0] = g.gain[0] * max(d_concentration, 0.0)
        I[1] = g.gain[1] * max(-d_concentration, 0.0)

        z = sigmoid(self.y + g.bias)
        dydt = (-self.y + g.W.T @ z + I) / g.tau
        self.y = self.y + self.dt * dydt

        z_out = sigmoid(self.y + g.bias)
        motor_diff = z_out[MOTOR[0]] - z_out[MOTOR[1]]  # dorsal - ventral
        turn = np.tanh(2.0 * motor_diff)  # squashed to [-1,1]
        return turn
