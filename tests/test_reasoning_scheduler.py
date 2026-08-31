from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reasoning_scheduler import PLAN_SCHEMA, Step, schedule


def test_schedule_respects_token_cap_and_reports_exact_accounting() -> None:
    steps = [Step("a", 5000, 1.0, 1.0), Step("b", 5000, 1.0, 0.5)]
    result = schedule(steps, max_flops=1e12, max_tokens=6000)
    assert result["schema"] == PLAN_SCHEMA
    assert result["used_tokens"] == 6000
    assert result["remaining_tokens"] == 0
    assert result["counts"] == {"full": 1, "partial": 1, "deferred": 0, "total": 2}
    assert result["plan"][0]["status"] == "FULL"
    assert result["plan"][1]["status"] == "PARTIAL"
    assert result["plan"][1]["admitted_tokens"] == 1000
    assert result["operational_authority"] is False
    assert "confidence" not in result


def test_priority_order_is_stable_for_equal_priorities() -> None:
    result = schedule(
        [
            Step("first", 10, 1.0, 0.8),
            Step("second", 10, 1.0, 0.8),
            Step("highest", 10, 1.0, 1.0),
        ],
        max_flops=100.0,
        max_tokens=100,
    )
    assert [row["step"] for row in result["plan"]] == ["highest", "first", "second"]


def test_flop_budget_can_partially_admit_work_and_is_fully_accounted() -> None:
    result = schedule(
        [Step("reason", 10, 2.0, 1.0)],
        max_flops=11.0,
        max_tokens=100,
    )
    row = result["plan"][0]
    assert row["status"] == "PARTIAL"
    assert row["admitted_tokens"] == 5
    assert row["admitted_flops"] == 10.0
    assert result["used_flops"] == 10.0
    assert result["remaining_flops"] == 1.0




@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "tokens": 1, "flops_per_token": 1.0, "priority": 1.0},
        {"name": "a", "tokens": 0, "flops_per_token": 1.0, "priority": 1.0},
        {"name": "a", "tokens": True, "flops_per_token": 1.0, "priority": 1.0},
        {"name": "a", "tokens": 1, "flops_per_token": 0.0, "priority": 1.0},
        {"name": "a", "tokens": 1, "flops_per_token": math.nan, "priority": 1.0},
        {"name": "a", "tokens": 1, "flops_per_token": 1.0, "priority": math.inf},
        {"name": "a", "tokens": 1, "flops_per_token": 1.0, "priority": 1.1},
    ],
)
def test_step_validation_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Step(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("max_flops", "max_tokens"),
    [
        (0.0, 10),
        (math.nan, 10),
        (100.0, 0),
        (100.0, True),
    ],
)
def test_invalid_budgets_fail_closed(max_flops: object, max_tokens: object) -> None:
    with pytest.raises(ValueError):
        schedule(
            [Step("a", 1, 1.0, 1.0)],
            max_flops=max_flops,  # type: ignore[arg-type]
            max_tokens=max_tokens,  # type: ignore[arg-type]
        )


def test_duplicate_step_names_and_wrong_collection_fail_closed() -> None:
    with pytest.raises(ValueError, match="unique"):
        schedule(
            [Step("same", 1, 1.0, 1.0), Step("same", 1, 1.0, 0.5)],
            max_flops=100.0,
            max_tokens=10,
        )
    with pytest.raises(ValueError, match="list of Step"):
        schedule(("not", "steps"), max_flops=100.0, max_tokens=10)  # type: ignore[arg-type]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
