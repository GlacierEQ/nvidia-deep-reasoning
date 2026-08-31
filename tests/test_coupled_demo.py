from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coupled_demo import budget_from_health


def test_nominal_health_maps_directly_without_artificial_floor() -> None:
    budget = budget_from_health(
        {"health_index": 0.2, "status": "NOMINAL"},
        base_flops=1000.0,
        base_tokens=1000,
    )
    assert budget == {
        "max_flops": 200.0,
        "max_tokens": 200,
        "scale": 0.2,
        "gate_state": "NOMINAL",
        "operational_authority": False,
    }


def test_zero_nominal_health_yields_zero_reasoning_budget() -> None:
    budget = budget_from_health(
        {"health_index": 0.0, "status": "NOMINAL"},
        base_flops=1000.0,
        base_tokens=1000,
    )
    assert budget["scale"] == 0.0
    assert budget["max_flops"] == 0.0
    assert budget["max_tokens"] == 0
    assert budget["gate_state"] == "REFUSED"


def test_critical_health_refuses_even_when_health_index_is_high() -> None:
    budget = budget_from_health(
        {"health_index": 0.99, "status": "CRITICAL"},
        base_flops=1000.0,
        base_tokens=1000,
    )
    assert budget["scale"] == 0.0
    assert budget["max_tokens"] == 0
    assert budget["gate_state"] == "REFUSED"


def test_warning_health_is_capped_without_upward_flooring() -> None:
    low = budget_from_health(
        {"health_index": 0.1, "status": "WARNING"},
        base_flops=1000.0,
        base_tokens=1000,
    )
    high = budget_from_health(
        {"health_index": 0.9, "status": "WARNING"},
        base_flops=1000.0,
        base_tokens=1000,
    )
    assert low["scale"] == 0.1
    assert high["scale"] == 0.5


@pytest.mark.parametrize(
    "health",
    [
        {},
        {"health_index": math.nan, "status": "NOMINAL"},
        {"health_index": -0.1, "status": "NOMINAL"},
        {"health_index": 1.1, "status": "NOMINAL"},
    ],
)
def test_malformed_health_fails_closed(health: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        budget_from_health(health)


def test_unknown_health_status_refuses_instead_of_guessing() -> None:
    budget = budget_from_health(
        {"health_index": 0.9, "status": "MYSTERY"},
        base_flops=1000.0,
        base_tokens=1000,
    )
    assert budget["gate_state"] == "REFUSED"
    assert budget["scale"] == 0.0
