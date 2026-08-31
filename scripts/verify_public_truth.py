#!/usr/bin/env python3
"""Resolve public-scope evidence into a continuation-oriented receipt.

The resolver keeps public statements, machine capabilities, and repository state
visible together. A mismatch is represented as remediation work; it never
silently upgrades a claim or terminates the inspection process.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "GlacierEQ/nvidia-deep-reasoning"

_REQUIRED_README = (
    "independent portfolio project",
    "not affiliated with NVIDIA",
    "does not measure FLOP reduction",
    "CUDA source is not compiled or benchmarked by hosted CI",
    "no live MCP, APEX, Mastermind, provider, or hardware integration",
    "no artificial minimum floor",
)
_UNSUPPORTED_PUBLIC_CLAIMS = (
    "up to 60%",
    "Fully connected to APEX Highway mesh",
    "returns attention pruning efficiency metrics",
    "on NVIDIA Tensor Cores",
)
_EXPECTED_CAPABILITIES = (
    "deterministic-local-attention-score-thresholding",
    "deterministic-local-reasoning-budget-scheduling",
    "fail-closed-health-gated-reasoning-budget",
)


def resolve_public_scope(
    readme: str,
    capabilities: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, object]:
    """Return a source-preserving evidence receipt for every scope observation."""
    work: list[str] = []
    for phrase in _REQUIRED_README:
        if phrase not in readme:
            work.append("document_scope_boundary:" + phrase)
    for phrase in _UNSUPPORTED_PUBLIC_CLAIMS:
        if phrase in readme:
            work.append("supply_evidence_or_rephrase_public_claim:" + phrase)
    observed_capabilities = capabilities.get("capabilities")
    if observed_capabilities != list(_EXPECTED_CAPABILITIES):
        work.append("reconcile_capability_inventory")
    if capabilities.get("operational_authority") is not False:
        work.append("clarify_capability_operational_authority")
    if state.get("principal_state") != "FUNCTIONAL_CANDIDATE":
        work.append("reconcile_principal_state_evidence")
    if state.get("operational_authority") is not False:
        work.append("clarify_state_operational_authority")
    return {
        "repository": REPOSITORY,
        "schema": "glaciereq.public-scope-resolution.v1",
        "continuation": "enabled",
        "status": "observed" if not work else "resolution_required",
        "verified_scope": list(_EXPECTED_CAPABILITIES),
        "public_statements_observed": True,
        "resolution_work": sorted(set(work)),
    }


def resolve_repository_scope(root: Path = ROOT) -> dict[str, object]:
    """Read repository evidence without making a missing source a terminal process error."""
    work: list[str] = []
    try:
        readme = (root / "README.md").read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "repository": REPOSITORY,
            "schema": "glaciereq.public-scope-resolution.v1",
            "continuation": "enabled",
            "status": "resolution_required",
            "resolution_work": ["restore_readme_evidence:" + type(exc).__name__],
        }
    for name in ("capabilities", "excellence-state"):
        path = root / "machine" / f"{name}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("root_not_object")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            work.append("restore_machine_evidence:" + name + ":" + type(exc).__name__)
            payload = {}
        if name == "capabilities":
            capabilities = payload
        else:
            state = payload
    receipt = resolve_public_scope(readme, capabilities, state)
    receipt["resolution_work"] = sorted(set([*receipt["resolution_work"], *work]))
    if receipt["resolution_work"]:
        receipt["status"] = "resolution_required"
    return receipt


def main() -> int:
    print(json.dumps(resolve_repository_scope(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
