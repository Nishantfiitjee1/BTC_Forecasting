import numpy as np
import pytest

from src.compare_mean_models import (
    ComparisonError,
    dm_statistic,
    moving_block_bootstrap_ci,
)


def test_dm_detects_positive_mean_difference():
    rng = np.random.default_rng(123)

    x = 0.5 + rng.normal(0, 0.05, 100)

    stat, p = dm_statistic(x, bandwidth=5)

    assert stat > 0
    assert p < 1e-6


def test_dm_rejects_too_short():
    with pytest.raises(ComparisonError):
        dm_statistic(np.ones(5), bandwidth=2)


def test_bootstrap_is_reproducible():
    x = np.linspace(-1, 1, 100)

    a = moving_block_bootstrap_ci(
        x,
        block_length=10,
        n_boot=100,
        seed=7,
    )

    b = moving_block_bootstrap_ci(
        x,
        block_length=10,
        n_boot=100,
        seed=7,
    )

    assert a == b