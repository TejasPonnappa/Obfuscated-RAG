"""
Laplace noise on stored embeddings, with the noise level calibrated for retrieval.

What this is: light obfuscation of the vectors in the local vector store. What it is not: a meaningful differential-privacy
guarantee. At the calibrated epsilon the noise is about a third of the embedding's size, which is what keeps retrieval
working (eval/privacy_utility.py); formal DP would need far more noise and would make retrieval no better than guessing.

Embedding vectors from all-MiniLM-L6-v2 have length 1 and 384 dimensions. Laplace noise with scale b = sensitivity / epsilon
has expected length about b * sqrt(2 * dimensions), so the noise-to-signal ratio is roughly 27.7 / epsilon.
"""
import math

import numpy as np

EMBEDDING_DIM = 384
DEFAULT_EPSILON = 100.0         # noise is ~0.28x the vector's length; keeps ~95% of the no-noise retrieval quality
MIN_USABLE_EPSILON = 50.0       # below this, retrieval degrades quickly

# Measured on a synthetic HR corpus (120 chunks, 4 corpora): recall@5 of the right chunk with noise on both the stored
# vectors and the question, as a share of recall@5 with no noise. Source: eval/privacy_utility.py.
RETENTION_BY_EPSILON = {2: 0.15, 10: 0.20, 20: 0.35, 50: 0.75, 100: 0.95, 200: 1.0, 500: 1.0}
EPSILON_CHOICES = sorted(RETENTION_BY_EPSILON)


class DifferentialPrivacyAgent:
    def __init__(self, epsilon: float = DEFAULT_EPSILON, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = self.sensitivity / self.epsilon

    def inject_laplace_noise(self, embeddings: np.ndarray) -> np.ndarray:
        noise = np.random.laplace(loc=0.0, scale=self.scale, size=np.shape(embeddings))
        return np.asarray(embeddings) + noise


def noise_to_signal_ratio(epsilon: float, dim: int = EMBEDDING_DIM, sensitivity: float = 1.0) -> float:
    """Expected length of the noise vector divided by the length (1) of a normalised embedding."""
    return (sensitivity / epsilon) * math.sqrt(2 * dim)


def retention_estimate(epsilon: float) -> float:
    """Approximate share of the no-noise retrieval quality that survives at this epsilon (interpolated in log epsilon)."""
    xs = [math.log(e) for e in EPSILON_CHOICES]
    ys = [RETENTION_BY_EPSILON[e] for e in EPSILON_CHOICES]
    return float(np.interp(math.log(max(epsilon, 1e-9)), xs, ys))
