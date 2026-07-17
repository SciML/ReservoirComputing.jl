# Findings

Branch: `investigate/continuous-predict-warmup`  
Related: #456 (merged), #397  

## Reproduce

```bash
cd benchmarks/continuous_warmup
julia --project=. -e 'using Pkg; Pkg.develop(path="../.."); Pkg.instantiate()'
julia --project=. run.jl --smoke    # ~20s, n_res=80
julia --project=. run.jl            # full: n_res=300, train=5000
julia --project=. run.jl --only=E1,E2,E6
```

## Smoke run (2026-07-17, n_res=80, train=800, predict=200)

### Structural (solid)

| Check | Result |
|-------|--------|
| E1: seeded `u0=0` vs package `predict` | **max abs diff = 0** (experimental AR loop matches master) |
| E2: `remake(prob; u0=…)` vs seeded warm | **identical NRMSE** |
| E6: continuous `st` after `train!` / `collectstates` | **no carry** (`has_carry=false`) |
| E6: discrete `st` after `train!` | **has carry** (`has_carry=true`) |
| E7: discrete cold vs warm-from-`st` | cold NRMSE 1.64 → warm 1.36 (rewarm via `collectstates` same as train `st`) |

### Forecasting quality (smoke — take carefully)

| Exp | Observation |
|-----|-------------|
| E2 SciML eq.5 | cold 17.1 → warm 13.0 (warm **helps**, both still bad under smoke HPs) |
| E1 ContinuousESN | cold 4.36 → full-train-terminal warm 7.52 (warm **hurts** under smoke HPs) |
| E3/E4 | non-zero seeds cluster ~7.5; zeros 4.36; need better ContinuousESN HPs before ranking seeds |
| E7 discrete | warm helps modestly |

**Interpretation so far**

1. The **API/plumbing** problem is real: continuous paths never store terminal ODE state in `st`; package AR always cold-starts (`ContinuousESN` → zeros; `SciMLProblemReservoir` → `prob.u0`).
2. Experimental `predict_ar_seeded` is a faithful probe (bit-match on cold).
3. **Whether warm helps forecasting** depends on model + hyperparams. Smoke ContinuousESN is not yet in the #456 Lorenz regime (N=300, tuned Wr/Win). Prefer E2 + full `run.jl` before locking API preference.
4. Warmup **time grid** must use unit windows (`Δt=1`, `tspan=(0,K)`), not the train model’s long `tspan` stretched over K samples (fixed in harness).

## Checklist

- [x] E1: seeded zeros matches package cold (`match_ok`)
- [x] E2: `remake(prob; u0=…)` ≡ seeded warm
- [x] E6: continuous `st` has **no** carry
- [x] E6: discrete `st` **has** carry
- [ ] E1 full: ContinuousESN cold ≫ warm with N=300 / tuned HPs
- [ ] E3: warmup length K saturates around K ≈ ___
- [ ] E4: ranking of seeds under good HPs
- [ ] E5: horizon where warm beats cold most (t_λ ≈ ___)
- [ ] E8: washout tail vs full-train terminal under good HPs

## API recommendation (draft — pending full runs)

| Option | Pros | Cons | Prefer? |
|--------|------|------|---------|
| `predict(...; initial_state=u0)` | explicit; matches E2 remake semantics | user must compute `u0` | strong candidate |
| `predict(...; warmup_data=W)` | ergonomic | must fix unit-window collect; length/tspan docs | good sugar on top of initial_state |
| Persist terminal in `st` | discrete-like | continuous `st` redesign; train/predict model split | investigate with Francesco |
| Docs-only | no API | footgun remains | insufficient alone |

Likely shape: **`initial_state` as primitive** + optional **`warmup_data`** that runs unit-window teacher-force and seeds `initial_state`.

## Next experiments on this branch

1. `julia --project=. run.jl` (full matrix, N=300).
2. Add ContinuousESN HP sweep closer to #456 (input scale, radius, bias, NLAT2 on/off).
3. Compare warm from **last train input window** vs **test teacher-force prefix** (oracle) under good HPs.
4. Draft tracking issue for SciML once full numbers land.

## Notes for Francesco

- Confirmed continuous `st` does not carry ODE state after `train!` / `collectstates`.
- Discrete does; rewarming discrete via `collectstates` ≡ using post-train `st`.
- For continuous, public workaround today: `SciMLProblemReservoir` + `remake(prob; u0=terminal)` (validated E2). `ContinuousESN` has **no** public `u0` hook without a custom AR loop.
