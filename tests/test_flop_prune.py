from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flop_prune import EVIDENCE_STATE, prune_scores


def test_threshold_pruning_is_deterministic_and_non_cuda() -> None:
    result = prune_scores([0.85, 0.02, 0.44, 0.01], 0.1)
    assert result["scores"] == [0.85, 0.0, 0.44, 0.0]
    assert result["pruned_count"] == 2
    assert result["pruned_fraction"] == 0.5
    assert result["evidence_state"] == EVIDENCE_STATE
    assert result["cuda_executed"] is False
    assert result["operational_authority"] is False


def test_empty_input_is_valid_and_stable() -> None:
    result = prune_scores([], 0.1)
    assert result["scores"] == []
    assert result["pruned_fraction"] == 0.0


@pytest.mark.parametrize(
    "scores,threshold",
    [
        ([math.nan], 0.1),
        ([math.inf], 0.1),
        ([0.5], math.nan),
        ([0.5], math.inf),
    ],
)
def test_non_finite_inputs_fail_closed(scores, threshold) -> None:
    with pytest.raises(ValueError):
        prune_scores(scores, threshold)
