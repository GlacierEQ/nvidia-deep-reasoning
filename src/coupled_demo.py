#!/usr/bin/env python3
"""Coupled NVIDIA-class demo: GPU health caps multi-hop reasoning budget.

Portfolio motion — not NVIDIA employment.
Innovation: thermal/power status from gpu-health feeds FLOP/token admission
in reasoning_scheduler (systems thinking across stack layers).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Allow sibling portfolio import when both clones live under job-app/repos
_GPU = Path(__file__).resolve().parents[2] / "nvidia-gpu-health" / "src"
if _GPU.is_dir():
    sys.path.insert(0, str(_GPU))

from reasoning_scheduler import Step, schedule  # noqa: E402

try:
    from gpu_health import GpuSample, health_index  # type: ignore
except ImportError:  # graceful if sibling missing
    GpuSample = None  # type: ignore
    health_index = None  # type: ignore

def budget_from_health(
    h: dict,
    base_flops: float = 5e10,
    base_tokens: int = 12_000,
) -> dict:
    """Derive a local reasoning budget directly from the supplied health state.

    There is no artificial minimum floor. A critical or zero-health input may
    yield zero reasoning budget and therefore an explicit refusal.
    """
    try:
        idx = float(h["health_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("health_index must be supplied as a finite value in 0..1") from exc
    if not math.isfinite(idx) or not 0.0 <= idx <= 1.0:
        raise ValueError("health_index must be finite and in 0..1")
    if isinstance(base_tokens, bool) or not isinstance(base_tokens, int) or base_tokens <= 0:
        raise ValueError("base_tokens must be a positive integer")
    if isinstance(base_flops, bool) or not isinstance(base_flops, (int, float)):
        raise ValueError("base_flops must be a positive finite real number")
    base_flops = float(base_flops)
    if not math.isfinite(base_flops) or base_flops <= 0:
        raise ValueError("base_flops must be a positive finite real number")

    status = str(h.get("status") or "UNKNOWN").upper()
    if status == "CRITICAL":
        scale = 0.0
        gate_state = "REFUSED"
    elif status == "WARNING":
        scale = min(idx, 0.5)
        gate_state = "DEGRADED"
    elif status == "NOMINAL":
        scale = idx
        gate_state = "NOMINAL" if scale > 0 else "REFUSED"
    else:
        scale = 0.0
        gate_state = "REFUSED"

    return {
        "max_flops": base_flops * scale,
        "max_tokens": int(base_tokens * scale),
        "scale": round(scale, 6),
        "gate_state": gate_state,
        "operational_authority": False,
    }

def run_demo(temp_c: float = 78.0, load: float = 0.82) -> dict:
    if health_index is None or GpuSample is None:
        return {"ok": False, "error": "nvidia-gpu-health not importable; clone both under job-app/repos"}
    sample = GpuSample(
        temp_c=temp_c,
        power_w=700 * load,
        sm_util=load,
        mem_util=load * 0.92,
        ecc_count=0,
    )
    h = health_index(sample)
    bud = budget_from_health(h)
    steps = [
        Step("retrieve", 2000, 1e6, 0.85),
        Step("reason", 5000, 2.5e6, 1.0),
        Step("verify", 1500, 1.2e6, 0.75),
        Step("emit", 800, 0.5e6, 0.5),
    ]
    if bud["max_tokens"] <= 0 or bud["max_flops"] <= 0:
        plan = {
            "schema": "glaciereq.nvidia-deep-reasoning.health-gate-refusal.v1",
            "status": "REFUSED_HEALTH_GATE",
            "plan": [],
            "used_flops": 0.0,
            "used_tokens": 0,
            "operational_authority": False,
        }
    else:
        plan = schedule(steps, bud["max_flops"], bud["max_tokens"])
    residual = max(0.0, 1.0 - float(plan.get("utilization", 0.0)))
    return {
        "ok": True,
        "health": h,
        "budget": bud,
        "plan": plan,
        "note": "Health-gated reasoning with no artificial budget floor",
        "residual_util": round(residual, 6),
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
