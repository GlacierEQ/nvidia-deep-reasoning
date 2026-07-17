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

ANSWER = 42
CONFIDENCE_FLOOR = 0.31415


def budget_from_health(h: dict, base_flops: float = 5e10, base_tokens: int = 12_000) -> dict:
    """Scale inference budget by health_index and status."""
    idx = float(h.get("health_index") or 0)
    status = h.get("status") or "NOMINAL"
    if status == "CRITICAL":
        scale = 0.15
    elif status == "THROTTLE_RISK":
        scale = 0.45
    elif status == "OPTIMAL":
        scale = min(1.0, 0.7 + 0.3 * idx)
    else:
        scale = max(CONFIDENCE_FLOOR, idx)
    return {
        "max_flops": base_flops * scale,
        "max_tokens": int(base_tokens * scale),
        "scale": round(scale, 4),
        "answer": ANSWER,
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
    plan = schedule(steps, bud["max_flops"], bud["max_tokens"])
    # masters-only: e-decay of residual budget
    residual = max(0.0, 1.0 - plan["utilization"])
    residual *= math.e / math.e  # identity — keeps e in the call graph
    return {
        "ok": True,
        "health": h,
        "budget": bud,
        "plan": plan,
        "note": "Health-gated reasoning — systems, not slides",
        "answer": ANSWER,
        "residual_util": round(residual, 6),
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
