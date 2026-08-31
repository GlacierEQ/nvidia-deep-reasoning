#!/usr/bin/env python3
"""Deterministic local reasoning-budget scheduler.

The scheduler accounts for caller-supplied token and modeled-work budgets. It
does not estimate model quality, issue hardware commands, or manufacture a
confidence score from budget utilization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

PLAN_SCHEMA = "glaciereq.nvidia-deep-reasoning.budget-plan.v1"


def _finite_real(value: object, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    if positive and numeric <= 0:
        raise ValueError(f"{field} must be positive")
    return numeric


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class Step:
    name: str
    tokens: int
    flops_per_token: float
    priority: float  # 0..1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("step name must be non-empty text")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "tokens", _positive_int(self.tokens, field="tokens"))
        object.__setattr__(
            self,
            "flops_per_token",
            _finite_real(self.flops_per_token, field="flops_per_token", positive=True),
        )
        priority = _finite_real(self.priority, field="priority")
        if not 0.0 <= priority <= 1.0:
            raise ValueError("priority must be in 0..1")
        object.__setattr__(self, "priority", priority)


def schedule(steps: list[Step], max_flops: float, max_tokens: int) -> dict[str, object]:
    """Build a stable, fully accounted local reasoning plan.

    Steps are ordered by descending priority and original declaration order.
    Work may be fully admitted, partially admitted, or deferred. Every admitted
    token is accounted against both budgets.
    """

    if not isinstance(steps, list) or not all(isinstance(step, Step) for step in steps):
        raise ValueError("steps must be a list of Step instances")
    names = [step.name for step in steps]
    if len(names) != len(set(names)):
        raise ValueError("step names must be unique")

    flops_budget = _finite_real(max_flops, field="max_flops", positive=True)
    token_budget = _positive_int(max_tokens, field="max_tokens")
    ordered = [
        step
        for _, step in sorted(
            enumerate(steps),
            key=lambda pair: (-pair[1].priority, pair[0]),
        )
    ]

    used_flops = 0.0
    used_tokens = 0
    plan: list[dict[str, object]] = []
    counts = {"FULL": 0, "PARTIAL": 0, "DEFERRED": 0}

    for step in ordered:
        requested_flops = step.tokens * step.flops_per_token
        remaining_tokens = token_budget - used_tokens
        remaining_flops = flops_budget - used_flops

        if step.tokens <= remaining_tokens and requested_flops <= remaining_flops:
            admitted_tokens = step.tokens
            status = "FULL"
        else:
            token_room = max(0, remaining_tokens)
            flop_room_tokens = max(0, int(remaining_flops / step.flops_per_token))
            admitted_tokens = min(step.tokens, token_room, flop_room_tokens)
            status = "PARTIAL" if admitted_tokens > 0 else "DEFERRED"

        admitted_flops = admitted_tokens * step.flops_per_token
        used_tokens += admitted_tokens
        used_flops += admitted_flops
        counts[status] += 1
        plan.append(
            {
                "step": step.name,
                "requested_tokens": step.tokens,
                "admitted_tokens": admitted_tokens,
                "requested_flops": round(requested_flops, 6),
                "admitted_flops": round(admitted_flops, 6),
                "priority": step.priority,
                "status": status,
            }
        )

    return {
        "schema": PLAN_SCHEMA,
        "plan": plan,
        "counts": {
            "full": counts["FULL"],
            "partial": counts["PARTIAL"],
            "deferred": counts["DEFERRED"],
            "total": len(plan),
        },
        "max_flops": flops_budget,
        "max_tokens": token_budget,
        "used_flops": round(used_flops, 6),
        "used_tokens": used_tokens,
        "remaining_flops": round(max(0.0, flops_budget - used_flops), 6),
        "remaining_tokens": max(0, token_budget - used_tokens),
        "utilization": round(used_flops / flops_budget, 6),
        "token_utilization": round(used_tokens / token_budget, 6),
        "operational_authority": False,
    }


if __name__ == "__main__":
    demo_steps = [
        Step("retrieve", 2000, 1e6, 0.9),
        Step("reason", 4000, 2e6, 1.0),
        Step("verify", 1000, 1.5e6, 0.7),
        Step("polish", 500, 0.5e6, 0.3),
    ]
    print(schedule(demo_steps, max_flops=1e10, max_tokens=6000))
