"""Deterministic local attention-score thresholding.

This module is a CPU reference implementation for portfolio verification. It does
not execute CUDA, control hardware, measure FLOPs, or establish NVIDIA telemetry
or performance authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Iterable


@dataclass(frozen=True)
class PruningResult:
    """Immutable result for one thresholding pass."""

    threshold: float
    input_count: int
    kept_count: int
    pruned_count: int
    masked_fraction: float
    output_scores: tuple[float, ...]
    operational_authority: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "input_count": self.input_count,
            "kept_count": self.kept_count,
            "pruned_count": self.pruned_count,
            "masked_fraction": self.masked_fraction,
            "output_scores": list(self.output_scores),
            "operational_authority": self.operational_authority,
        }


def _finite_real(value: Real, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real number")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def prune_attention_scores(
    scores: Iterable[Real],
    threshold: Real,
) -> PruningResult:
    """Mask scores strictly below ``threshold`` with zero.

    The function is deterministic, validates every numeric input, does not mutate
    the caller's iterable, and deliberately reports only mask statistics. A
    masked fraction is not a measured FLOP, latency, energy, or quality saving.
    """

    cutoff = _finite_real(threshold, field="threshold")
    values = tuple(_finite_real(value, field=f"scores[{index}]") for index, value in enumerate(scores))
    output = tuple(value if value >= cutoff else 0.0 for value in values)
    pruned = sum(value < cutoff for value in values)
    total = len(values)
    kept = total - pruned
    fraction = (pruned / total) if total else 0.0
    return PruningResult(
        threshold=cutoff,
        input_count=total,
        kept_count=kept,
        pruned_count=pruned,
        masked_fraction=fraction,
        output_scores=output,
    )
