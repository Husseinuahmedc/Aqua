"""Tests for stable sample selection in the candidate review-sheet utility."""

import pytest

from tools.inspect_candidate_sanity import _select_evenly_spaced


def test_candidate_sanity_selection_spans_sorted_samples_deterministically() -> None:
    selected = _select_evenly_spaced(700, 12)

    assert len(selected) == 12
    assert selected == sorted(set(selected))
    assert selected[0] == 0
    assert selected[-1] == 699
    assert selected == _select_evenly_spaced(700, 12)


@pytest.mark.parametrize("total,count", [(0, 1), (4, 0), (4, 5)])
def test_candidate_sanity_selection_rejects_invalid_counts(
    total: int,
    count: int,
) -> None:
    with pytest.raises(ValueError):
        _select_evenly_spaced(total, count)
