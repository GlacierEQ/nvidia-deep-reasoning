# NVIDIA Deep Reasoning — Local Reasoning-Budget & Threshold-Policy Exhibit

> **Independent GlacierEQ portfolio work. Not affiliated with, endorsed by, or connected to NVIDIA.**

This repository demonstrates two bounded local mechanisms:

1. a deterministic **reasoning-budget scheduler** that allocates modeled token/FLOP budgets across prioritized steps; and
2. a deterministic **CPU threshold-pruning reference model** aligned with the simple score-threshold semantics present in `src/flop_prune_kernel.cu`.

## Verified local scope

- `src/reasoning_scheduler.py` validates modeled step costs and allocates FULL / PARTIAL / DEFERRED token budgets without exceeding caller-supplied token or FLOP ceilings.
- `src/flop_prune.py` applies deterministic finite-value threshold pruning and reports the observed fraction of local inputs zeroed.
- Both Python surfaces return explicit evidence state and `operational_authority: false`.
- `src/flop_prune_kernel.cu` is retained as a **CUDA source reference artifact**. Current repository CI does not compile or execute CUDA and therefore does not claim CUDA runtime proof.

## Evidence boundaries

`LOCAL_REASONING_BUDGET_MODEL_NOT_NVIDIA_OR_LLM_RUNTIME_AUTHORITY`

`LOCAL_THRESHOLD_PRUNING_REFERENCE_NOT_CUDA_PERFORMANCE_PROOF`

Current proof does **not** establish:

- NVIDIA affiliation, employment, endorsement, or proprietary access;
- CUDA compilation, GPU execution, Tensor Core use, or an optimized attention implementation;
- entropy computation inside the CUDA kernel;
- any measured FLOP reduction, latency improvement, accuracy preservation, or reasoning-quality benefit;
- PyTorch parity or comparison with an executed CUDA kernel;
- an o1/o3-class model implementation or access to proprietary reasoning traces;
- a live MCP tool, Mastermind connection, APEX Highway connection, or provider integration;
- production inference scheduling, hardware control, or operational authority.

## Reproduce the verified scope

```bash
python -m pip install pytest
PYTHONPATH=src python -m pytest -q
python src/reasoning_scheduler.py
```

The repository-owned CI verifies the Python scheduler, threshold reference model, malformed-input behavior, public truth boundary, and source-level CUDA nonclaim on Python 3.11, 3.12, and 3.13.
