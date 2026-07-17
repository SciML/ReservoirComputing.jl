# Continuous AR predict warmup investigation

**Branch:** `investigate/continuous-predict-warmup`  
**Context:** Deferred from [#456](https://github.com/SciML/ReservoirComputing.jl/pull/456)  
(Francesco: *“add the warmup as part of a separate PR… investigate that a bit further”*).

This harness **does not modify package `src/` or root `Project.toml`**.  
It only measures cold vs warm reservoir state at the start of continuous
autoregressive `predict`, and probes candidate API shapes **locally**.

## Problem

| Path | Initial reservoir state for AR `predict` |
|------|------------------------------------------|
| Discrete `ESN` | Carries through `StatefulLayer` / sequential `apply` |
| `SciMLProblemReservoir` | `res.prob.u0` (often zeros) |
| `ContinuousESN` / `ContinuousESNCell` | Always `zeros(out_dims)` |

On Lorenz, PR #456 measured roughly **NRMSE ~1.5 (cold)** vs **~0.11 (warmed)**.

## Experiments

| ID | Question |
|----|----------|
| E1 | Reproduce cold vs terminal-train warm on `ContinuousESN` (Lorenz) |
| E2 | Same for hand-rolled `SciMLProblemReservoir` (eq. 5) |
| E3 | Warmup **length** sweep (last `K` teacher-forced inputs before AR) |
| E4 | Seed variants: zeros / random / train terminal / test-prefix warm / wrong state |
| E5 | Horizon curve: NRMSE vs Lyapunov time (cold vs warm) |
| E6 | Does `st` after `train!` / `collectstates` hold usable continuous state? |
| E7 | Discrete `ESN` control (same data split) |
| E8 | Washout interaction (train with washout, warm from post-washout tail) |

## Setup

```bash
cd benchmarks/continuous_warmup
julia --project=. -e 'using Pkg; Pkg.develop(path="../.."); Pkg.instantiate()'
```

## Run

Smoke (plumbing only — small N, short series, ~20s):

```bash
julia --project=. run.jl --smoke
```

**Full matrix** (forecast analysis — matches #456 Lorenz scale):

| knob | full default |
|------|----------------|
| `n_res` | 300 |
| train / predict | 5000 / 1250 |
| Wr spectral radius | 0.9 |
| Win scale / bias scale | 0.1 / 0.05 |
| solver | Tsit5 (`reltol=1e-6`) |

```bash
julia --project=. run.jl            # full
julia --project=. run.jl --n-res=300
julia --project=. run.jl --only=E1,E2,E5,E7
```

Expect minutes–tens of minutes for continuous `collectstates` at N=300 / T=5000 (wall times are recorded per experiment).

Outputs land in `results/`:

- `results/summary_full.json` / `summary_smoke.json` — machine-readable  
- `results/summary_full.md` / `summary_smoke.md` — tables  
- `results/summary.json` — latest run copy  
- `results/FINDINGS.md` — interpretation

## Design options under test (no API change yet)

Local experimental helpers in `src/predict_variants.jl`:

1. **`initial_state`** — pass reservoir `u0` into a mirrored AR loop  
2. **`warmup_data`** — run teacher-forced `collectstates` on a prefix, take last column as `u0`, then AR  
3. **`remake(prob; u0=…)`** — only for `SciMLProblemReservoir` public path  
4. **Read `st` after train** — check whether continuous `st` already carries terminal state (expected: **no**)

Package `predict` is left untouched so results stay comparable to merged `master`.

## Relation to other work

- Not PR4 perf ([#467](https://github.com/SciML/ReservoirComputing.jl/issues/467))  
- Not a user-facing API PR until API choice is agreed  
- Feeds a future tracking issue + design PR
