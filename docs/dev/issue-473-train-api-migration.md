# Migration contract: train API v1.0 + LinearSolve default

**Status:** phases 1–6 implemented on branch; phase 7 (remove `train!`) open  
**Branch:** `wip/473-linearsolve-default-readout`  
**Package version target:** `0.12.28` (breaking: default ridge solver + LinearSolve hard dep)  
**Issues:** [#473](https://github.com/SciML/ReservoirComputing.jl/issues/473), [#367](https://github.com/SciML/ReservoirComputing.jl/issues/367)  
**Related:** closed [#294](https://github.com/SciML/ReservoirComputing.jl/issues/294) (LinearSolve as extension)

Working notes for Option B (full train redesign). Not a public user guide.
Phases 1–6 are landed (`train!` depwarn + docs on `train`). Phase 7 removes
`train!` in a future breaking release.

---

## 1. End state (Option B)

After v1.0 removal of shims:

1. Primary user entry point is non-mutating **`train`** (no bang).
2. Ridge training separates **objective** (`StandardRidge`) from **solver**
   (LinearSolve algorithm, or legacy `QRSolver` while retained).
3. **LinearSolve.jl is a hard dependency**; ridge default solver is a LinearSolve
   algorithm (explicitly pinned — see §3).
4. Package extension `RCLinearSolveExt` is gone (or empty / removed).
5. MLJLinearModels and LIBSVM remain optional extensions with clear dispatch
   rules; they are not forced hard deps.

`LinearReadout` itself does not change. It only receives trained weights.

---

## 2. Target public API

### 2.1 Model-level training (end-to-end)

```julia
ps, st = train(rc, train_data, target_data, ps, st;
    objective = StandardRidge(0.0),
    solver = nothing,          # → package default for this objective
    washout = 0,
    return_states = false,
    kwargs...,                 # forwarded to backend (LS init, MLJ fit, …)
)
```

**Returns** (unchanged semantics):

- `(ps, st)` normally
- `((ps, st), states)` if `return_states=true` (states after washout)

**Positional arguments stay fixed:**

`rc, train_data, target_data, ps, st`

No positional `train_method`. That becomes keyword `objective` to match the
#367 sketch and to free the call site for objective/solver clarity.

### 2.2 Feature-level training (lower hook)

Keep a two-level design (do not collapse):

```julia
W = train(objective, states, target_data;
    solver = nothing,
    kwargs...,
)
```

- `states :: AbstractMatrix` shape `(n_features, T)`
- `target_data :: AbstractMatrix` shape `(n_outputs, T)`
- Ridge return: `AbstractMatrix` shape `(n_outputs, n_features)` for
  `LinearReadout` weights

Washout is **not** applied here; callers (or model-level `train`) handle it.

### 2.3 Compatibility shim (migration only)

```julia
function train!(rc, train_data, target_data, ps, st,
        train_method = StandardRidge(0.0);
        washout = 0, return_states = false, kwargs...)
    # Phase 2–5: silent or soft forward to train(...)
    # Phase 6: @deprecate / depwarn
    # Phase 7: removed
    return train(rc, train_data, target_data, ps, st;
        objective = train_method,
        washout = washout,
        return_states = return_states,
        kwargs...)
end
```

Positional `train_method` on `train!` maps to `objective`.  
`solver` remains a keyword on both during dual-API life.

### 2.4 Objective vs solver (dispatch rules)

| `objective` | Role of `solver` | Notes |
|-------------|------------------|--------|
| `StandardRidge` | LinearSolve `SciMLLinearSolveAlgorithm`, or `QRSolver` while retained | Default when `solver === nothing`: §3 |
| MLJ regressor (ext) | MLJ solver **only if** that package’s API expects it; must not be a LinearSolve alg | Prefer MLJ’s own solver field / kwargs; if both conflict → `ArgumentError` |
| LIBSVM SVR types (ext) | `solver` ignored / rejected | `solver !== nothing` → clear `ArgumentError` |

**Rule:** one keyword name `solver`, but **meaning is backend-specific**.
Invalid combinations error loudly; they never silently ignore.

### 2.5 `StandardRidge`

- Remains a pure objective: **regularization coefficient only** (`reg`).
- Does **not** store a solver.
- Docstring equation is the **semantic target** (normal equations / ridge
  closed form). Numerical method is the solver’s job.

```math
W = Y X^\top (X X^\top + \lambda I)^{-1}
```

with package layout `X = states` `(n×T)`, `Y = targets` `(m×T)`,
`W` `(m×n)` so that `Y ≈ W * X`.

(Equivalent row-wise form for each output channel.)

---

## 3. Defaults (pinned, reproducible)

| Knob | v1.0 default | Rationale |
|------|----------------|-----------|
| `objective` | `StandardRidge(0.0)` | Same as today |
| Ridge `solver` when `nothing` | **`LinearSolve.QRFactorization()`** | Explicit algorithm — **not** “whatever LinearSolve picks this month” |
| Legacy in-house | `QRSolver()` still callable via `solver=QRSolver()` until Phase 7 | Escape hatch + characterization baseline |
| Hard dependency | LinearSolve (Phase 5+) | Required for default path without `using LinearSolve` at call site |

**Compat while dual-API (Phases 2–3):**  
New `train(...; objective=, solver=nothing)` may keep **`QRSolver()`** as
default until Phase 4, so introducing the API is non-breaking.  
**Phase 4** is the intentional numerics break: `nothing` → `QRFactorization()`.

**Re-export policy (Phase 5+):**  
Either re-export `QRFactorization` from ReservoirComputing or document
`using LinearSolve` for non-default algs. Prefer: default works with only
`using ReservoirComputing`; advanced algs need `using LinearSolve`.

---

## 4. Numerical formulation

### 4.1 LinearSolve / default path (authoritative for `StandardRidge`)

Build the Gram system once:

```text
A = X * X' + λ I          # (n, n)
B = X * Y'                # (n, m)   multi-RHS
solve A \ B  (batched)    # (n, m)
W = B'                    # (m, n)
```

Requirements:

- **One factorization, multi-column RHS** (no per-output `solve!` loop in the
  default path). Uses LinearSolve batch RHS support.
- Element type follows `states` / `targets` (Float32 stays Float32 unless the
  chosen LS alg forces otherwise — document if so).
- `λ` converted to `eltype(states)`.

### 4.2 Legacy `QRSolver`

Current in-house QR-augmented implementation remains available as
`solver=QRSolver()` until removal. Characterization tests record its behavior;
it is **not** required to bit-match LinearSolve.

**Known risk (must be characterized in Phase 1):**  
Current QR assembly uses array concatenations that only make sense for certain
dimension patterns; docs examples often use `res_dims == T`. Phase 1 tests must
either prove general correctness or file/fix bugs before relying on it as
oracle.

### 4.3 Agreement policy (tests)

| Regime | Expectation |
|--------|-------------|
| Well-conditioned synthetic features, `λ > 0`, Float64 | `W` agrees across QRSolver (if valid), `QRFactorization`, `SVDFactorization` within tight `rtol` (e.g. `1e-8`–`1e-10`) |
| Float32, same regime | Looser `rtol` (e.g. `1e-4`–`1e-5`) |
| `λ = 0`, rank-deficient / underdetermined | **No** universal agreement; assert finite weights + defined error policy only |
| Closed-form problems (orthogonal / scaled identity features) | Match analytical `W` |

Do **not** use random ESN weight bit-identity as the primary oracle.

---

## 5. Dependency and module layout

| Phase | LinearSolve | `RCLinearSolveExt` | `QRSolver` |
|-------|-------------|--------------------|------------|
| 0–1 | weakdep (today) | present | default |
| 2–3 | weakdep or test-only hard in test env | present | default for `solver=nothing` |
| 4 | still weakdep **or** hard (prefer hard if default needs it without user `using`) | dispatch may live in core | non-default |
| 5 | **hard dep** | **removed** | optional legacy |
| 7 | hard dep | gone | **removed** (or undocumented internal) |

**Compat bounds:** multi-RHS Gram solves require LinearSolve **4.2+**
(`LinearSolve = "4.2"`). Bump to 5.x separately if/when Dependabot #472 lands.

**Precompile:** Phase 4+ workload must call default ridge `train` so TTFX
reflects the real default path. Record rough numbers in the PR body.

---

## 6. Phased plan (exit criteria)

### Phase 0 — Contract (this document)

- [x] Draft on branch  
- [ ] Maintainer / self review of open questions §9  
- **Exit:** no unresolved “blocker” questions

### Phase 1 — Characterization tests (no API break)

Add `test/test_train_ridge.jl` (name flexible) + put **LinearSolve in test
targets**.

**Tier 1 — pure ridge (no ESN):**

- Shape `(m, n)` for various `(n, T, m)`
- Closed-form cases
- Washout via model-level or `_apply_washout` unit tests
- Float32 / Float64
- Multi-output

**Tier 2 — solver matrix (extension loaded in tests):**

- `QRSolver`, `QRFactorization`, `SVDFactorization` under agreement policy §4.3
- Document disagreements with `@test_broken` or explicit “no agreement” cases

**Tier 3 — thin integration:**

- One `ESN` + ridge `train!` smoke (current API)
- One MLJ ridge path smoke (existing ext)
- One SVR path smoke (existing) — ensure still green, not redesigned

**Status (branch work):**

- [x] `LinearSolve` added to `[extras]` and `[targets].test` in `Project.toml`
- [x] `test/test_train_ridge.jl` — Tier 1–2 ridge characterization + ESN smoke
- [x] Local run: **67 passed** (`include("test/test_train_ridge.jl")`)
- [x] Runic check on the new test file
- [ ] Full `Pkg.test()` Core group green on CI / local full suite
- [ ] MLJ / SVR smokes remain covered by existing test files (not duplicated here)

**Characterization findings locked by tests:**

- Default ridge solver is `QRSolver()`
- Output weights shape is always `(n_outputs, n_features)`
- Well-conditioned / regularized problems: `QRSolver`, `QRFactorization`,
  `SVDFactorization` agree with the normal-equations reference within tight rtol
- Ill-conditioned `λ = 0`, `n ≈ T`, Float32: only finiteness + shape asserted
- LinearSolve path preserves Float32 eltype; QRSolver may promote when `reg`
  is Float64
- `train!(...; solver = QRFactorization())` works end-to-end on `ESN`

**Exit:** CI green on master-compatible API; LinearSolve path is first-class in CI.
### Phase 2 — Dual API (additive, non-breaking)

- Implement model-level `train(rc, data, y, ps, st; objective, solver, ...)`
- Implement/adjust feature-level `train(objective, X, Y; solver, ...)`
- `train!` forwards to `train` **without** depwarn yet
- New tests call **new** API; old tests keep `train!`

**Status (branch work):**

- [x] Model-level `train(rc, …; objective, solver, washout, return_states)`
- [x] Feature-level ridge `solver=nothing` → `_default_ridge_solver()` = `QRSolver()`
- [x] `train!` silent forwarder (positional method → `objective`)
- [x] `_fit_readout` avoids leaking `solver=nothing` into MLJ/LIBSVM kwargs
- [x] `test/test_train_api.jl` (17 tests)
- [x] Phase 1 suite still green (67 tests)
- [x] API docs: `docs/src/api/train.md` lists `train`, `train!`, `QRSolver`
- [x] Precompile hits model-level `train` and `train!`
- [x] Runic clean on touched files
- [ ] Full Core `Pkg.test()` on CI

**Exit:** both APIs green; no default numerics change; no hard dep yet
(`solver=nothing` still `QRSolver`).

### Phase 3 — Backend quality

- Multi-RHS LinearSolve implementation (core or ext)
- Parity tests Tier 2 green under §4.3
- Clear errors for invalid objective/solver pairs
- Optional: move LS code into core early if it simplifies testing

**Status (branch work):**

- [x] `RCLinearSolveExt`: one Gram factorization + multi-column RHS via
  `LinearProblem(gram, rhs)` / `solve` (no per-output loop)
- [x] Descriptive names in ext (`n_features`, `n_samples`, `n_outputs`)
- [x] `QRSolver` path checks sample-count `DimensionMismatch`
- [x] Fallback `_train_ridge` → clear `ArgumentError` for unknown solvers
- [x] Tests: unsupported solver, QR + LS sample mismatch, multi-RHS vs reference
- [ ] Still **not** default (`solver=nothing` → `QRSolver`)
- [ ] Still weakdep (hard dep is Phase 5)

**Exit:** default path *implementation* is production-quality; still not default.

### Phase 4 — Default solver flip (breaking numerics)

- `solver === nothing` → `QRFactorization()` for `StandardRidge`
- NEWS / release note: results may change
- Update goldens only where defaults are asserted
- Precompile uses new default

**Status (branch work, shipped with Phase 5):**

- [x] `_default_ridge_solver() = QRFactorization()`
- [x] Tests assert default ≡ explicit `QRFactorization()`
- [x] Training tutorial documents new default
- [x] Breaking: default ridge numerics may differ from pre-#473 `QRSolver`

**Exit:** tests define and pass new default; versioning policy decided (minor vs 1.0-rc)

### Phase 5 — Hard dependency

- LinearSolve → `[deps]`
- Remove `RCLinearSolveExt`
- Export / document default algorithm story
- Measure load/precompile vs Phase 1 baseline

**Status (branch work, combined with Phase 4):**

- [x] LinearSolve in `[deps]`; removed from weakdeps / extensions
- [x] Multi-RHS `_train_ridge(::SciMLLinearSolveAlgorithm, …)` in `src/train.jl`
- [x] Deleted `ext/RCLinearSolveExt.jl`
- [x] Export `QRFactorization` for the pinned default without `using LinearSolve`
- [x] Advanced algs (e.g. `SVDFactorization`) still need `using LinearSolve`

**Exit:** `using ReservoirComputing` alone can `train` with default ridge solver

### Phase 6 — Deprecate old surface

- `train!` → `Base.depwarn` once per session / standard Julia deprecation
- Docs tutorials switched to `train`
- Optional: deprecate `QRSolver` public export

**Status (branch work):**

- [x] `Base.depwarn` on `train!`
- [x] Tutorials / README / examples prefer `train`
- [x] `@test_deprecated` coverage for `train!` wrapper
- [ ] `QRSolver` remains public (not deprecated yet)

**Exit:** docs + examples use new API; depwarn tests exist

### Phase 7 — Remove (v1.0)

- Remove `train!` (or keep forever as alias — **decision in §9**)
- Remove `QRSolver` if deprecated
- Labels `breaking` + `v1.0` satisfied

**Exit:** only Option B surface remains

**Hard rule:** never merge Phase 1 with Phase 4 in one PR.  
Prefer separate PRs per phase (2+3 may combine; 4+5 may combine only if review is tight).

---

## 7. Test ownership (first-class)

| File (proposed) | Contents |
|-----------------|----------|
| `test/test_train_ridge.jl` | Tier 1–2 ridge/solvers |
| `test/test_train_api.jl` | Dual API, kwargs, errors, deprecations (Phases 2+) |
| Existing model tests | Keep smoke; do **not** multiply per-model solver matrices |

`Project.toml` `[targets].test` must include `LinearSolve` from Phase 1.

SciMLTesting / `test_groups.toml`: ridge tests run in **Core** group (not QA-only).

---

## 8. Non-goals (explicit)

- Redesigning `LinearReadout`, `Collect`, or `collectstates`
- GPU / sparse ridge performance work (may work via LinearSolve incidentally)
- Changing MLJ / LIBSVM mathematics
- Making MLJ or LIBSVM hard dependencies
- Bit-identical weights across all solvers
- Migrating every model test file to exhaustive solver grids
- Folding unrelated v1.0 cleanups into this branch
- Jumping LinearSolve major versions as part of the default flip (track #472 separately)

---

## 9. Open questions (resolve before Phase 2 code)

Blockers marked **(blocker)**:

1. **(blocker)** After Phase 7, is `train!` **deleted** or kept as a permanent
   non-warning alias?  
   *Recommendation:* delete at 1.0 after one deprecation cycle; matches #367.

2. **(blocker)** Should Phase 4 (default flip) wait for a **1.0-rc** tag, or is a
   **0.x breaking minor** acceptable under COLPRAC/SemVer for this package?  
   *Recommendation:* implement on branch behind dual API; ship default flip when
   maintainers cut 1.0-rc. Avoid surprising 0.12 users mid-stream.

3. Default algorithm: confirm **`QRFactorization()`** vs `CholeskyFactorization()`
   / other for SPD Gram matrix.  
   *Recommendation:* QR is a safe general default; optional later specialized
   default for SPD.

4. Re-export `QRFactorization` from ReservoirComputing?  
   *Recommendation:* yes for the default symbol only, or document
   `using LinearSolve` — pick one in Phase 5 PR.

5. Fix vs keep current `QRSolver` dimensional behavior if Phase 1 finds bugs?  
   *Recommendation:* fix if broken for general `(n,T,m)`; do not preserve bugs
   as “characterization.”

Non-blockers can default to recommendations above.

---

## 10. Immediate next actions (after contract approval)

1. Resolve §9 blockers (short discussion).
2. **Phase 1 only:** add LinearSolve to test targets + `test_train_ridge.jl`
   characterization suite against **current** `train!` / `QRSolver` / ext path.
3. No dual API, no default flip, no Project.toml hard dep until Phase 1 is green.

---

## 11. Success criteria (project-level)

- Users can write the #367-style API and get SciML-default ridge solves without
  loading LinearSolve manually (Phase 5+).
- Migration is possible without a single mega-PR.
- Numerics changes are intentional, documented, and test-gated.
- MLJ and LIBSVM keep working with explicit error boundaries.
- Docs and precompile match the real default path.

---

## 12. Changelog intent (for eventual NEWS)

```markdown
### Breaking
- `train!` deprecated then removed in favor of `train` (#367)
- Ridge training takes `objective` + `solver` kwargs (#473)
- Default ridge solver is LinearSolve `QRFactorization` (was in-house `QRSolver`)
- LinearSolve is a required dependency

### Deprecated
- `train!`, `QRSolver` (during dual-API window)
```

(Exact wording when shipping.)
