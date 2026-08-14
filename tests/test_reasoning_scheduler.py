from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reasoning_scheduler import EVIDENCE_STATE, Step, schedule


def test_schedule_respects_token_and_flop_caps() -> None:
    steps = [
        Step("a", 5000, 2.0, 1.0),
        Step("b", 5000, 2.0, 0.5),
    ]
    result = schedule(steps, max_flops=12000, max_tokens=6000)
    assert result["used_tokens"] <= 6000
    assert result["used_flops"] <= 12000
    assert result["evidence_state"] == EVIDENCE_STATE
    assert result["operational_authority"] is False
    assert [item["status"] for item in result["plan"]] == ["FULL", "PARTIAL"]


def test_equal_priority_preserves_input_order() -> None:
    steps = [Step("first", 1, 1.0, 0.5), Step("second", 1, 1.0, 0.5)]
    result = schedule(steps, max_flops=2.0, max_tokens=2)
    assert [item["step"] for item in result["plan"]] == ["first", "second"]


@pytest.mark.parametrize(
    "steps,max_flops,max_tokens",
    [
        ([Step("", 1, 1.0, 0.5)], 1.0, 1),
        ([Step("bad", -1, 1.0, 0.5)], 1.0, 1),
        ([Step("bad", 1, 0.0, 0.5)], 1.0, 1),
        ([Step("bad", 1, 1.0, math.nan)], 1.0, 1),
        ([], math.nan, 1),
        ([], -1.0, 1),
        ([], 1.0, -1),
        ([], 1.0, True),
    ],
)
def test_invalid_budget_inputs_fail_closed(steps, max_flops, max_tokens) -> None:
    with pytest.raises(ValueError):
        schedule(steps, max_flops=max_flops, max_tokens=max_tokens)
