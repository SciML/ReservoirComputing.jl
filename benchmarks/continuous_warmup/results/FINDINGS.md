# Findings — continuous AR predict warmup

Branch: `investigate/continuous-predict-warmup`  
PR: https://github.com/SciML/ReservoirComputing.jl/pull/476  
Related: #456 (merged), #397  

## Reproduce

```bash
cd benchmarks/continuous_warmup
julia --project=. -e 'using Pkg; Pkg.develop(path="../.."); Pkg.instantiate()'

# plumbing only (~20s)
julia --project=. run.jl --smoke

# forecast analysis — #456-scale (~5 min on M-series laptop)
julia --project=. run.jl
# → results/summary_full.{json,md}
```

### Full-run config

| knob | value |
|------|-------|
| mode | **full** (not smoke) |
| n_res | 300 |
| train / predict | 5000 / 1250 samples |
| dt / λ_max | 0.02 / 0.9056 → predict span ≈ **22.6 t_λ** |
| Wr spectral radius | 0.9 |
| Win scale / bias scale | 0.1 / 0.05 |
| state modifiers | `NLAT2()` |
| solver | Tsit5, `reltol=1e-6`, `abstol=1e-8` |
| ridge | 1e-6 |
| seed | 17 |
| suite wall | **~275 s** (this machine) |

---

## 1. Structural (independent of HPs)

| Check | Result |
|-------|--------|
| Seeded `u0=0` vs package `predict` | **max abs diff = 0** |
| `remake(prob; u0=…)` vs seeded warm | **identical NRMSE** |
| Continuous `st` after `train!` / `collectstates` | **no carry** |
| Discrete `st` after `train!` | **has carry** |
| Discrete rewarm via `collectstates` | ≡ post-train `st` |

**Conclusion:** continuous AR always cold-starts today. There is nowhere in `st` to put a terminal ODE state. Public warm workaround for generic SciML reservoirs: `remake(prob; u0=terminal)`. `ContinuousESN` has no public `u0` hook without a custom AR loop.

---

## 2. Forecast quality — full matrix (the real analysis)

### Headline: short-horizon NRMSE (ContinuousESN, E5 / E1 horizons)

| Horizon | steps | t_λ | cold NRMSE | warm (train-terminal) NRMSE |
|---------|------:|----:|-----------:|----------------------------:|
| short | 28 | 0.5 | 0.75 | **0.21** |
| 1 t_λ | 55 | 1.0 | 0.69 | **0.15** |
| 2 t_λ | 110 | 2.0 | 1.34 | **0.085** |
| 3 t_λ | 166 | 3.0 | 1.51 | **0.093** |
| 4 t_λ | 221 | 4.0 | 1.66 | **0.14** |
| 6 t_λ | 331 | 6.0 | 1.51 | **0.67** |

Warm recovers the #456-class short-horizon gap: at 2–3 Lyapunov times, warm is ~**0.09** vs cold ~**1.3–1.5**.

Full-horizon NRMSE over all 1250 steps (≈22 t_λ) is high for both (cold **1.48**, warm **1.23**) — expected once chaos has diverged; the interesting regime is the first few t_λ.

### Valid prediction time (E3, threshold 0.5)

| Warmup K | VPT (t_λ) | full-horizon NRMSE |
|---------:|----------:|-------------------:|
| 0 (cold) | **0.22** | 1.48 |
| 10 | **4.87** | 1.14 |
| 50–2000 | ~4.13 | ~1.18–1.28 |

Even a short teacher-forced warmup (K=10) lifts VPT from ~0.2 → ~5 t_λ.

### Seed ranking (E4, full-horizon NRMSE)

| seed | NRMSE |
|------|------:|
| train_terminal | **1.23** |
| zeros (cold) | 1.48 |
| oracle test prefix (K=100) | 1.45 |
| randn / shuffled terminal | ~13.5 (catastrophic) |

Wrong non-zero seeds are much worse than cold. Warm is not “any u0” — it has to be a dynamically consistent terminal state.

### SciMLProblemReservoir eq.5 (E2)

| variant | full-horizon NRMSE |
|---------|-------------------:|
| cold | 1.47 |
| warm train-terminal | **1.31** |
| `remake(prob; u0=…)` | **1.31** (matches seeded) |

### Discrete control (E7)

| variant | full-horizon NRMSE |
|---------|-------------------:|
| cold (fresh `st`) | 1.55 |
| warm (post-train `st`) | **1.14** |
| rewarm via `collectstates` | **1.14** |

### Washout (E8, washout=200)

| variant | NRMSE |
|---------|------:|
| cold | 1.48 |
| warm full train | 1.18 |
| warm post-washout tail | **1.13** |

Post-washout tail is slightly better than full-train terminal.

---

## 3. Wall times (full, not a perf PR — order-of-magnitude only)

| stage | wall |
|-------|------|
| ContinuousESN train (`collectstates` + ridge), N=300, T=5000 | ~18 s |
| AR cold predict, 1250 steps | ~2 s |
| AR warm predict, 1250 steps | ~1 s |
| Warmup collect (full train pass for terminal u0) | ~train-scale |
| Full E1–E8 suite | ~275 s |

Warm vs cold AR cost is comparable; the extra cost of warm is mainly the one-time teacher-forced collect for `u0`. Not a throughput regression story — a correctness/usability one. Perf work stays on #467.

---

## 4. Smoke vs full (why smoke lied)

| | smoke (n=80, T=800) | full (n=300, T=5000) |
|--|---------------------|----------------------|
| ContinuousESN warm vs cold | warm **hurt** | warm **helps** (esp. short horizon) |
| SciML warm vs cold | helps a little | helps |
| Structural checks | same | same |

Smoke is fine for plumbing (`match_ok`, carry inspection). **Do not** use smoke for API justification.

---

## 5. API recommendation (updated after full run)

| Option | Verdict after full data |
|--------|-------------------------|
| `predict(...; initial_state=u0)` | **Yes — primitive.** Matches remake; E4 shows wrong seeds are dangerous so it should be explicit. |
| `predict(...; warmup_data=W)` | **Yes — sugar.** K≥10 already unlocks ~5 t_λ VPT; must use unit windows (`Δt=1`). |
| Persist terminal in `st` | Still attractive for discrete-like ergonomics; needs continuous `st` design — separate decision |
| Docs-only | Insufficient: cold looks like a broken model at 1–4 t_λ |

**Proposed first code PR:**  
`initial_state` on continuous AR `predict` + unit tests that seeded zeros ≡ today’s cold path and that a known `u0` is actually used. Optional `warmup_data` in the same PR or immediately after. Lorenz short-horizon eye-test optional but strong.

---

## 6. Questions for @MartinuzziFrancesco

1. OK to treat short-horizon (1–4 t_λ) NRMSE / VPT as the acceptance metric for warmup, not full-horizon NRMSE?
2. Ship `initial_state` first, or `warmup_data` first, or both?
3. Should `train!` / `collectstates` start writing a continuous terminal into `st` (discrete-like), or keep state external via kwargs?
4. Keep this harness under `benchmarks/` long-term?

---

## File map

| Path | Role |
|------|------|
| `run.jl` | `--smoke` / full / `--only=` / `--n-res=` |
| `src/predict_variants.jl` | experimental seeded AR (no package API) |
| `src/experiments.jl` | E1–E8 |
| `results/summary_full.md` | latest full table |
| `results/FINDINGS.md` | this document |
