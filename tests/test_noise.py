import numpy as np

from noise import DifferentialPrivacyAgent, FastDiscriminator, optimize_privacy_budget


def test_laplace_scale_is_sensitivity_over_epsilon():
    agent = DifferentialPrivacyAgent(epsilon=4.0, sensitivity=2.0)
    assert agent.scale == 0.5
    noise = agent.inject_laplace_noise(np.zeros((20000, 1))).ravel()
    assert abs(np.abs(noise).mean() - 0.5) < 0.03                         # mean |Laplace(b)| equals b


def test_smaller_epsilon_means_more_noise():
    x = np.zeros((5000, 4))
    small = np.abs(DifferentialPrivacyAgent(epsilon=0.5).inject_laplace_noise(x)).mean()
    large = np.abs(DifferentialPrivacyAgent(epsilon=5.0).inject_laplace_noise(x)).mean()
    assert small > large


def test_discriminator_declines_to_judge_a_tiny_sample():
    clean = np.random.default_rng(0).normal(size=(3, 8))
    assert FastDiscriminator().evaluate_noise(clean, clean + 1.0) == 0.5


def test_optimizer_returns_vectors_of_the_same_shape():
    clean = np.random.default_rng(1).normal(size=(40, 16)) * 0.05
    epsilon, noisy = optimize_privacy_budget(clean)
    assert noisy.shape == clean.shape and epsilon > 0
