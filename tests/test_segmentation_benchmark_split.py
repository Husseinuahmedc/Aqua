"""Offline tests for seeded, image-ID-only benchmark partitions."""

import numpy as np
import pytest

from core.water.regions import validate_binary_mask
from tools.segmentation_benchmark import _make_experiments, deterministic_split


def test_split_is_deterministic_disjoint_and_independent_of_input_order() -> None:
    sample_ids = [f"sample_{index:06d}" for index in range(1, 701)]

    first = deterministic_split(sample_ids)
    second = deterministic_split(list(reversed(sample_ids)))

    assert first == second
    development = {row["sample_id"] for row in first if row["subset"] == "development"}
    holdout = {row["sample_id"] for row in first if row["subset"] == "holdout"}
    assert len(development) == 560
    assert len(holdout) == 140
    assert development.isdisjoint(holdout)
    assert development | holdout == set(sample_ids)


@pytest.mark.parametrize(
    "sample_ids,fraction",
    [([], 0.8), (["one"], 0.8), (["a", "b"], 0.0), (["a", "b"], 1.0)],
)
def test_split_rejects_invalid_partition_sizes(
    sample_ids: list[str],
    fraction: float,
) -> None:
    with pytest.raises(ValueError):
        deterministic_split(sample_ids, development_fraction=fraction)


def test_each_registered_experiment_returns_a_binary_same_size_mask() -> None:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :32] = (255, 0, 0)
    image[:, 32:] = (0, 0, 255)

    experiments = _make_experiments()
    experiment_ids = [experiment.experiment_id for experiment in experiments]

    assert len(experiment_ids) == len(set(experiment_ids))
    assert len(experiments) >= 15
    for experiment in experiments:
        mask = experiment.predict(image.copy())
        validate_binary_mask(mask)
        assert mask.shape == image.shape[:2]
