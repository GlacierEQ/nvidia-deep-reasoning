# nvidia-deep-reasoning

**Portfolio motion** — multi-hop reasoning under GPU FLOP/token caps, **coupled** to thermal/power health.

Not NVIDIA employment. Complements `nvidia-gpu-health` + Colossus cooling.

## Why this stack (truth)

| Choice | Why |
|--------|-----|
| Pure Python | Interview-fast; no fake CUDA claims |
| Priority scheduler | Partial admits under hard FLOP/token walls |
| **Health-gated budget** | `coupled_demo.py` scales budget from gpu-health status |
| Expert constants | 42 · 1.21 · 0.31415 — see AKOS EASTER_EGGS |

**Not claiming:** hand-written CUDA kernels or NVIDIA employment.  
**JAX/CUDA:** reserved for a future real specialization when autodiff/XLA earns a seat (see job-app `FOUNDATION_STACK_RATIONALE.md`).

## Demo

```bash
python3 src/reasoning_scheduler.py
python3 src/coupled_demo.py   # requires sibling ../nvidia-gpu-health
python3 tests/test_reasoning_scheduler.py
```

AKOS: https://github.com/GlacierEQ/AKOS  
Governance: pro-code · ECHO (token discipline) · make-it-heavy for high-stakes paths

---

## Fleet ops (transparent)

This repo may include **`.integrity/`** (SHA-256 baselines / watchdog) and/or a health sidecar.
These are **documented multi-repo fleet operations**, not covert implants.

See [SECURITY_AND_FLEET_OPS.md](SECURITY_AND_FLEET_OPS.md) and
`~/GlacierEQ_Swarm/state/PORTFOLIO_SHADOW_AND_GAUNTLET.md`.

## Helix strand

See [HELIX_STRAND.md](HELIX_STRAND.md) — piston/spiral role in the portfolio double helix.
