# NVIDIA Deep Reasoning

**Deterministic local attention-score pruning and reasoning-budget scheduling.**

This is an **independent portfolio project**. It is **not affiliated with NVIDIA**, does not use proprietary NVIDIA data, and does not establish NVIDIA employment, endorsement, telemetry access, or hardware authority.

## What is implemented

The repository contains two bounded, executable mechanisms:

1. **Attention-score thresholding** in `src/attention_pruning.py`
   - validates finite numeric inputs;
   - masks scores strictly below a caller-supplied threshold;
   - reports kept/pruned counts and masked fraction;
   - returns `operational_authority=false`.
2. **Reasoning-budget scheduling** in `src/reasoning_scheduler.py`
   - orders local reasoning steps by priority;
   - admits full, partial, or deferred token work under explicit token and modeled-work budgets;
   - never emits hardware commands.

`src/flop_prune_kernel.cu` is a small CUDA threshold-mask source artifact corresponding to the local thresholding idea. **CUDA source is not compiled or benchmarked by hosted CI.** The verified portfolio capability is the deterministic local Python behavior, not GPU execution.

## What the proof does not establish

- The project **does not measure FLOP reduction**, latency, throughput, energy savings, model quality, or production performance.
- A masked-score fraction is not a measured hardware-work reduction.
- No Tensor Core execution, CUDA runtime deployment, production inference serving, or NVIDIA telemetry is claimed.
- There is **no live MCP, APEX, Mastermind, provider, or hardware integration**.
- Historical local receipt files are not substitutes for exact-current-head hosted CI.
- No operational or production authority is granted by repository metadata or sidecar files.

## Verification

Hosted CI runs on Python 3.11, 3.12, and 3.13 and performs:

```bash
python -m compileall -q src tests scripts
python -m pytest -q
python scripts/verify_public_truth.py
```

For a local run:

```bash
python -m pip install pytest
python -m pytest -q
python scripts/verify_public_truth.py
```

## Example

```python
from src.attention_pruning import prune_attention_scores

result = prune_attention_scores([0.85, 0.02, 0.10, 0.44], threshold=0.10)
print(result.as_dict())
```

Scores equal to the threshold are retained. NaN, infinity, boolean pseudo-numerics, and other malformed numeric inputs fail closed.

## Repository truth boundary

The admissible public capability is intentionally narrow: **deterministic local attention-score thresholding plus deterministic local reasoning-budget scheduling**. Any stronger claim requires new implementation and exact-head proof before it belongs here.
