"""CPU reference for the thresholding semantics mirrored by the CUDA source artifact.

This module proves deterministic threshold behavior only. It does not prove CUDA
compilation, Tensor Core execution, entropy measurement, or FLOP reduction.
"""
from __future__ import annotations

import math

EVIDENCE_STATE = "LOCAL_THRESHOLD_PRUNING_REFERENCE_NOT_CUDA_PERFORMANCE_PROOF"


def prune_scores(scores: list[float], threshold: float) -> dict:
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    cleaned: list[float] = []
    for score in scores:
        if not math.isfinite(score):
            raise ValueError("scores must be finite")
        cleaned.append(score if score >= threshold else 0.0)
    pruned_count = sum(1 for before, after in zip(scores, cleaned) if before != after)
    return {
        "scores": cleaned,
        "input_count": len(scores),
        "pruned_count": pruned_count,
        "pruned_fraction": round(pruned_count / len(scores), 4) if scores else 0.0,
        "evidence_state": EVIDENCE_STATE,
        "cuda_executed": False,
        "operational_authority": False,
    }
