from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("public_truth", ROOT / "scripts" / "verify_public_truth.py")
assert SPEC and SPEC.loader
public_truth = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(public_truth)


def clean_readme() -> str:
    return "\n".join(public_truth._REQUIRED_README)


def clean_capabilities() -> dict[str, object]:
    return {"capabilities": list(public_truth._EXPECTED_CAPABILITIES), "operational_authority": False}


def clean_state() -> dict[str, object]:
    return {"principal_state": "FUNCTIONAL_CANDIDATE", "operational_authority": False}


def test_clean_scope_is_observed_without_resolution_work():
    receipt = public_truth.resolve_public_scope(clean_readme(), clean_capabilities(), clean_state())

    assert receipt["continuation"] == "enabled"
    assert receipt["status"] == "observed"
    assert receipt["resolution_work"] == []


def test_missing_scope_and_unsupported_claim_become_resolution_work():
    receipt = public_truth.resolve_public_scope(
        "independent portfolio project\nup to 60%",
        {"capabilities": ["other"], "operational_authority": True},
        {"principal_state": "PRODUCTION", "operational_authority": True},
    )

    assert receipt["continuation"] == "enabled"
    assert receipt["status"] == "resolution_required"
    assert "supply_evidence_or_rephrase_public_claim:up to 60%" in receipt["resolution_work"]
    assert "reconcile_capability_inventory" in receipt["resolution_work"]
    assert "reconcile_principal_state_evidence" in receipt["resolution_work"]
    assert any(item.startswith("document_scope_boundary:") for item in receipt["resolution_work"])


def test_missing_repository_evidence_returns_actionable_receipt(tmp_path):
    receipt = public_truth.resolve_repository_scope(tmp_path)

    assert receipt["continuation"] == "enabled"
    assert receipt["status"] == "resolution_required"
    assert receipt["resolution_work"] == ["restore_readme_evidence:FileNotFoundError"]
