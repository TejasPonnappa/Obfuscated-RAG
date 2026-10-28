import numpy as np

from noise import (DEFAULT_EPSILON, DifferentialPrivacyAgent, MIN_USABLE_EPSILON, noise_to_signal_ratio,
                   retention_estimate)


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


def test_noise_to_signal_ratio_matches_simulation():
    np.random.seed(0)
    unit = np.zeros((300, 384))
    measured = np.linalg.norm(DifferentialPrivacyAgent(epsilon=DEFAULT_EPSILON).inject_laplace_noise(unit), axis=1).mean()
    assert abs(measured - noise_to_signal_ratio(DEFAULT_EPSILON)) < 0.03


def test_default_epsilon_keeps_noise_below_a_third_of_the_signal_and_above_usable_floor():
    assert noise_to_signal_ratio(DEFAULT_EPSILON) < 0.33
    assert DEFAULT_EPSILON >= MIN_USABLE_EPSILON


def test_retention_falls_as_epsilon_falls():
    assert retention_estimate(2) < retention_estimate(20) < retention_estimate(50) < retention_estimate(100) <= 1.0
