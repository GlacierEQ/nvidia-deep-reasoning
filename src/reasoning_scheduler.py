#!/usr/bin/env python3
"""Deterministic local reasoning-budget scheduler.

This module allocates modeled token/FLOP budgets. It does not invoke an LLM,
measure GPU FLOPs, control NVIDIA hardware, or establish inference authority.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EVIDENCE_STATE = "LOCAL_REASONING_BUDGET_MODEL_NOT_NVIDIA_OR_LLM_RUNTIME_AUTHORITY"


@dataclass(frozen=True)
class Step:
    name: str
    tokens: int
    flops_per_token: float
    priority: float


def _validate_step(step: Step) -> None:
    if not step.name:
        raise ValueError("step name is required")
    if isinstance(step.tokens, bool) or not isinstance(step.tokens, int) or step.tokens < 0:
        raise ValueError("step tokens must be a non-negative integer")
    if not math.isfinite(step.flops_per_token) or step.flops_per_token <= 0:
        raise ValueError("flops_per_token must be finite and positive")
    if not math.isfinite(step.priority) or not 0.0 <= step.priority <= 1.0:
        raise ValueError("priority must be finite and in 0..1")


def schedule(steps: list[Step], max_flops: float, max_tokens: int) -> dict:
    if not math.isfinite(max_flops) or max_flops < 0:
        raise ValueError("max_flops must be finite and non-negative")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
        raise ValueError("max_tokens must be a non-negative integer")
    for step in steps:
        _validate_step(step)

    ordered = sorted(enumerate(steps), key=lambda pair: (-pair[1].priority, pair[0]))
    used_flops = 0.0
    used_tokens = 0
    plan: list[dict] = []
    for _, step in ordered:
        need_flops = step.tokens * step.flops_per_token
        room_tokens = max_tokens - used_tokens
        room_flops = max_flops - used_flops
        admit_tokens = min(
            step.tokens,
            max(0, room_tokens),
            max(0, int(room_flops / step.flops_per_token)),
        )
        if admit_tokens == step.tokens:
            status = "FULL"
        elif admit_tokens > 0:
            status = "PARTIAL"
        else:
            status = "DEFERRED"
        used_tokens += admit_tokens
        used_flops += admit_tokens * step.flops_per_token
        plan.append(
            {
                "step": step.name,
                "requested_tokens": step.tokens,
                "admitted_tokens": admit_tokens,
                "modeled_flops": round(admit_tokens * step.flops_per_token, 2),
                "status": status,
            }
        )

    utilization = used_flops / max_flops if max_flops > 0 else 0.0
    return {
        "plan": plan,
        "used_flops": round(used_flops, 2),
        "used_tokens": used_tokens,
        "flop_budget_utilization": round(utilization, 4),
        "evidence_state": EVIDENCE_STATE,
        "operational_authority": False,
    }


if __name__ == "__main__":
    example = [
        Step("retrieve", 2000, 1e6, 0.9),
        Step("reason", 4000, 2e6, 1.0),
        Step("verify", 1000, 1.5e6, 0.7),
        Step("polish", 500, 0.5e6, 0.3),
    ]
    print(schedule(example, max_flops=1e10, max_tokens=6000))
