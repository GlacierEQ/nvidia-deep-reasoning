from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_TOKEN = "LOCAL_REASONING_BUDGET_MODEL_NOT_NVIDIA_OR_LLM_RUNTIME_AUTHORITY"
PRUNE_TOKEN = "LOCAL_THRESHOLD_PRUNING_REFERENCE_NOT_CUDA_PERFORMANCE_PROOF"
EXPECTED_CAPABILITIES = [
    "deterministic-modeled-reasoning-budget-scheduling",
    "validated-partial-admission-under-token-and-flop-caps",
    "deterministic-cpu-threshold-pruning-reference",
    "cuda-threshold-kernel-source-reference",
]


def test_readme_preserves_independence_and_cuda_nonclaim() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Not affiliated with, endorsed by, or connected to NVIDIA" in readme
    assert SCHEDULER_TOKEN in readme
    assert PRUNE_TOKEN in readme
    assert "does not compile or execute CUDA" in readme
    for forbidden in (
        "FLOP count reduction** by up to 60%",
        "returns attention pruning efficiency metrics",
        "Fully connected to APEX Highway mesh",
    ):
        assert forbidden not in readme


def test_machine_surface_is_exact_and_non_operational() -> None:
    capabilities = json.loads((ROOT / "machine/capabilities.json").read_text())
    state = json.loads((ROOT / "machine/excellence-state.json").read_text())
    contract = json.loads((ROOT / "machine/target-contract.json").read_text())
    assert capabilities["capabilities"] == EXPECTED_CAPABILITIES
    assert capabilities["cuda_executed"] is False
    assert capabilities["operational_authority"] is False
    assert state["principal_state"] == "FUNCTIONAL_CANDIDATE"
    assert state["evidence_state"] == "IMPLEMENTED_CURRENT_HEAD_NATIVE_PROOF_REQUIRED"
    assert contract["current"]["cuda_executed"] is False
    assert contract["current"]["operational_authority"] is False


def test_cuda_source_is_not_misrepresented_as_entropy_or_flop_measurement() -> None:
    cuda = (ROOT / "src/flop_prune_kernel.cu").read_text(encoding="utf-8")
    assert "entropy(" not in cuda
    assert "softmax" not in cuda.lower()
    assert "flop" not in cuda.lower().replace("flop_prune", "") or "FLOP reduction" not in cuda
