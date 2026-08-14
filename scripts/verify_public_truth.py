#!/usr/bin/env python3
"""Fail closed when public claims exceed the repository's verified local scope."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
CAPABILITIES = json.loads((ROOT / "machine" / "capabilities.json").read_text(encoding="utf-8"))
STATE = json.loads((ROOT / "machine" / "excellence-state.json").read_text(encoding="utf-8"))

required_readme = (
    "independent portfolio project",
    "not affiliated with NVIDIA",
    "does not measure FLOP reduction",
    "CUDA source is not compiled or benchmarked by hosted CI",
    "no live MCP, APEX, Mastermind, provider, or hardware integration",
)
for phrase in required_readme:
    if phrase not in README:
        raise SystemExit(f"README truth boundary missing: {phrase}")

for forbidden in (
    "up to 60%",
    "Fully connected to APEX Highway mesh",
    "returns attention pruning efficiency metrics",
    "on NVIDIA Tensor Cores",
):
    if forbidden in README:
        raise SystemExit(f"unsupported public claim present: {forbidden}")

expected_capabilities = [
    "deterministic-local-attention-score-thresholding",
    "deterministic-local-reasoning-budget-scheduling",
]
if CAPABILITIES.get("capabilities") != expected_capabilities:
    raise SystemExit("machine/capabilities.json exceeds or diverges from verified scope")
if CAPABILITIES.get("operational_authority") is not False:
    raise SystemExit("operational_authority must be false")
if STATE.get("principal_state") != "FUNCTIONAL_CANDIDATE":
    raise SystemExit("repository must remain FUNCTIONAL_CANDIDATE until canonical proof")
if STATE.get("operational_authority") is not False:
    raise SystemExit("machine state must preserve operational_authority=false")

print("PUBLIC_TRUTH_GATE=PASS")
