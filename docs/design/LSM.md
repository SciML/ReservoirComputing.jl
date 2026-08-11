# Liquid State Machine (LSM) — Design Document

**Status:** design + research only (no `src/` implementation yet)  
**Tracking:** [SciML/ReservoirComputing.jl#494](https://github.com/SciML/ReservoirComputing.jl/issues/494)  
**Fellowship:** [SciML/ReservoirComputing.jl#397](https://github.com/SciML/ReservoirComputing.jl/issues/397)  
**Branch:** `Saswatsusmoy/add-lsm-model` @ `eaf0db80` (master)  
**Closest analog:** `ContinuousESN` / `ContinuousESNCell` + `RCODEReservoirExt`  
**Author session:** design handoff (implementation deferred)

> **Naming trap:** existing `LIFESN` / `LocalInformationFlow` is *Local Information Flow* (Liu 2025), **not** Leaky Integrate-and-Fire. LSM’s LIF neuron must be a new type (`LIFCell` / `LIFNeuron`). Do not overload the LIFESN name.

---

## 1. Goals (from #494)

Ship a **solid foundation** for continuous-time spiking reservoirs on top of the existing SciML continuous path (`AbstractSciMLProblemReservoir`, `RCODEReservoirExt`), modelled on how `ContinuousESN` landed:

- Pluggable neuron model (`AbstractSpikingNeuron`); ship LIF first.
- First-class input encoding (`AbstractInputEncoder`); ship current-injection + Poisson rate.
- First-class spike-side readout (`AbstractSpikeReadout`); ship spike-count, exponential-filter, filtered-voltage.
- Dale-compliant connectivity initializer(s).
- Correct callback semantics for threshold / reset / refractory — once, correctly.
- Extension seams for Izhikevich / AdEx / HH, more encoders, STDP (deferred).

Non-goals for v1: STDP, multi-compartment, GPU, sparse-spike perf pass, product AR warmup API (still deferred for continuous models generally; see #476 notes).

---

## 2. Codebase map (current state)

### 2.1 Continuous stack (use as template)

| Piece | Path | Role |
| --- | --- | --- |
| Supertype | `src/layers/sciml_reservoir.jl` | `AbstractSciMLProblemReservoir`, `SciMLProblemReservoir`, `TerminalStateSampling` |
| Cell | `src/layers/continuous_esn_cell.jl` | `ContinuousESNCell <: AbstractSciMLProblemReservoir` — holds dims, inits, `equations`, `tspan`, solve `args`/`kwargs` |
| Model | `src/models/continuous_esn.jl` | 3-field `ContinuousESN` + **stub ctor** that errors without extension |
| Extension | `ext/RCODEReservoirExt.jl` | Real ctor; ZOH input; `_collectstates` / `_predict` for cell + generic `SciMLProblemReservoir` |
| Container | `src/reservoircomputer.jl` | Two-level `_collectstates(rc.reservoir, rc, …)` dispatch |

### 2.2 How `ContinuousESN` wires into SciML

**Construction (extension):**

```text
ContinuousESN(in, res, out, tspan, solver_args...; kwargs...)
  → ContinuousESNCell(...)          # AbstractSciMLProblemReservoir
  → state_modifiers (wrapped)
  → LinearReadout(res => out)
  → ContinuousESN(cell, mods, ro)
```

**`collectstates` (extension, `_collectstates(::ContinuousESNCell, …)`):**

1. Build sample grid from `tspan` and `size(data, 2)` (same layout as `SciMLProblemReservoir`).
2. Build ZOH input `ZeroOrderHoldInterp(data, input_ts)`.
3. Merge `ps.reservoir` + `input` into solve `p` via `_build_solve_params`.
4. Allocate `u0 = zeros(eltype(Win), out_dims)`.
5. Build a **fresh** `ODEProblem` each call (does **not** remake a stored user problem).
6. `solve(...; saveat=sample_ts, save_everystep=false, dense=false, cell.kwargs...)`.
7. Sample with `TerminalStateSampling` → `(res_dims, T)` matrix.
8. Apply `state_modifiers` column-wise.
9. Return `(states, newst)` with **`st.reservoir` passed through unchanged** (no carry today).

**AR `predict`:** `init` once + `reinit!` per window; cold `u0 = zeros(...)` (warmup deferred).

**Generic `SciMLProblemReservoir` path:** `remake(res.prob; tspan, p=solve_p)` then solve — used when the user supplies their own problem.

### 2.3 Discrete cousin (do not confuse)

- `LIFESN` / `LocalInformationFlow` — discrete lookback wrapper around `ESNCell`.
- No spiking, no callbacks, no continuous time.

### 2.4 Params / states doctrine (Francesco)

- Trainable / fixed weights in `ps`.
- Mutable solver caches / RNG / discrete clocks in `st`.
- Thin 3-field models: `(:reservoir, :state_modifiers, :readout)`.
- Continuous path lives behind `RCODEReservoirExt` (weakdep: `DataInterpolations`).

---

## 3. Research result: `remake` × callbacks

Probed against **SciMLBase v3.41** + **OrdinaryDiffEqTsit5** (2026-08 design session). Source of truth for `remake(::ODEProblem)` (SciMLBase `remake.jl`):

```julia
# when kwargs === missing (the usual path):
ODEProblem{iip}(f, newu0, tspan, newp, prob.problem_type;
    (values(prob.kwargs)::NamedTuple)..., (values(_kwargs)::NamedTuple)...)
```

### 3.1 Findings

| Question | Result |
| --- | --- |
| Does `remake(prob; tspan, p, u0)` preserve `callback` stored on the problem? | **Yes.** Callback remains in `prob.kwargs`; **same object identity** (`===`). |
| Does `CallbackSet` survive remake? | **Yes**, same identity. |
| Does `VectorContinuousCallback` survive remake? | **Yes**, same identity. |
| Does `remake(prob; kwargs=(;))` preserve callbacks? | **No** — replaces `kwargs` entirely and drops them. |
| Problem-level + solve-level `callback=`? | **Both fire** (DiffEq merges them). |
| Does `reinit!` re-run callback `initialize`? | **Yes** (observed 1 call per `reinit!`+`solve!` window). |
| Does remake/reinit reset **mutable state closed over by callbacks**? | **No.** Shared `Ref` / arrays persist across solves. Leftover refractory clocks can zero out all subsequent spikes. |
| Condition evaluation count vs tol | Loose `1e-3` ≈ 248 evals; tight `1e-10` ≈ 1912 evals over same interval. **Never draw RNG inside continuous `condition`.** |

### 3.2 Implication for LSM (biggest design decision)

**Two viable placement strategies:**

**A. Solve-time callbacks (recommended for LSMCell, mirrors ContinuousESN freshness)**

- Build `CallbackSet` (threshold + refractory bookkeeping + encoder events) **inside** `_collectstates` / `_predict` each call.
- Pass as `solve(...; callback=cbset, …)` or attach to a **fresh** `ODEProblem(...; callback=cbset)`.
- Close over buffers owned by this call (or views into `st.reservoir` that we reset first).
- Remake interaction is **irrelevant** for the ContinuousESN-style path because we do not rely on a long-lived problem’s kwargs.

**B. Problem-kwargs callbacks (only if using user-supplied `SciMLProblemReservoir`)**

- `remake` **does** preserve callbacks cleanly — the “unknown” from #494 is resolved: **no SciMLBase bug**.
- Hazard is **not** remake dropping callbacks; it is **shared mutable callback state** across solves if the same callback object is reused without `initialize` / explicit reset.
- Mitigation: always implement `initialize` that zeroes refractory clocks / spike counters; never put irreversible RNG draws in `condition`.

**Recommendation:** LSM’s built-in cell follows **A** (like `ContinuousESNCell`). Document that user-built `SciMLProblemReservoir` problems that attach spiking callbacks themselves must provide a correct `initialize` (B). No upstream SciMLBase issue needed unless we later hit a specialization/identity edge case.

### 3.3 Ordering notes for spiking

- Threshold detection: `VectorContinuousCallback` with `out[i] = V[i] - V_th` (upcrossing).
- `affect!`: reset `V[i] → V_reset`, set `ref_until[i] = t + τ_ref`, record spike for readout / last-spike-time.
- During refractory: either clamp RHS `dV[i]=0` when `t < ref_until[i]`, or force condition negative so no re-fire; **both** need the refractory clock.
- `save_positions=(false,false)` recommended so spike events do not inflate `sol.u` beyond `saveat` (feature sampling stays grid-aligned with ContinuousESN).
- Simultaneous multi-neuron events: implement `affect!` robustly for single-index calls; verify DiffEq simultaneous-event path in unit tests (implementation detail — see open questions).

---

## 4. Proposed architecture

### 4.1 Layer cake

```text
                    ┌─────────────────────────────────────┐
 train / predict    │  LSM  (3-field AbstractESN model)   │
                    │  reservoir | state_modifiers | readout│
                    └───────────────┬─────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────┐
 collectstates      │  LSMCell <: AbstractSciMLProblem…   │
 (RCODEReservoirExt)│  neuron + encoder + spike_readout   │
                    │  + W_in, W_res inits + tspan/solver │
                    └───────────────┬─────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
     AbstractSpikingNeuron  AbstractInputEncoder  AbstractSpikeReadout
         LIFCell()           CurrentInjection()    ExponentialFilter…
                             PoissonRateEncoder()  SpikeCount…
                                                   FilteredVoltage…
              │                     │                     │
              └──────── CallbackSet + ODE RHS ────────────┘
                                    │
                          solve → features (F × T)
                                    │
                          LinearReadout (ridge) → y
```

**Important split:**

- **Spike-side readout** (`AbstractSpikeReadout`) maps continuous trajectory + spike events → **feature matrix** `(feature_dims, T)` consumed by `train` / outer `LinearReadout`.
- **Outer readout** remains the package-standard `LinearReadout` (ridge via `train`). This matches ContinuousESN and keeps `addreadout!` / train API unchanged.

`SpikeCountReadout` features are discrete counts → ridge still works; AR predict is teacher-forced only (no continuous inverse of counts). Document that.

### 4.2 File plan (from #494, lightly adjusted)

```text
src/layers/abstract_spiking.jl      # AbstractSpikingNeuron + contracts
src/layers/lif_neuron.jl            # LIFCell (name: avoid collision with lif_wrapper.jl)
src/encoders/abstract_encoder.jl
src/encoders/current_injection.jl
src/encoders/poisson_rate.jl
src/readouts/spike_readout.jl       # AbstractSpikeReadout + three concrete types
src/inits/inits_lsm.jl              # dale_sparse [, distance_dependent_sparse]
src/models/lsm.jl                   # LSM model + stub ctor
ext/RCODEReservoirExt.jl            # real LSM ctor + _collectstates/_predict
test/Extensions/lsm_tests.jl
test/Layers/lif_neuron_tests.jl
docs/src/tutorials/lsm.md
docs/src/api/...                    # models, layers, encoders
```

> Prefer `lif_neuron.jl` over `lif_cell.jl` next to existing `lif_wrapper.jl` (Local Information Flow) to reduce human confusion. Final public name still `LIFCell` / `LIFNeuron` — **open naming question for Francesco**.

---

## 5. API sketches

All signatures are **design-level**; types follow existing `@concrete` / LuxCore patterns. Method bodies deferred.

### 5.1 `AbstractSpikingNeuron`

```julia
"""
Developer interface for a continuous-time spiking neuron population.

## Required interface

- `neuron_rhs!(du, u, p, t, neuron, st_neuron)` — population ODE for membrane
  (and any continuous auxiliaries). Must be allocation-free in the hot path.
- `spike_condition!(out, u, t, integrator, neuron, st_neuron)` — fill
  `out[i] = g_i(u,t)` with zero-crossing ⇒ spike of unit `i` (upcrossing).
- `reset!(integrator, idx, neuron, st_neuron)` — atomic reset of unit(s)
  that spiked; update refractory clock / last-spike-time in `st_neuron`.
- `state_dim(neuron, n_units)` — length of continuous state vector `u`.
- `init_neuron_state(rng, neuron, n_units)` — initial `u0`.
- `init_neuron_st(rng, neuron, n_units)` — discrete callback state
  (refractory clocks, last spike times, …) living in `st.reservoir.neuron`.

Optional:

- `build_spike_callback(neuron, n_units, st_neuron)` — default builds a
  `VectorContinuousCallback` from `spike_condition!` + `reset!`.
"""
abstract type AbstractSpikingNeuron end

function neuron_rhs! end
function spike_condition! end
function reset! end
function state_dim end
function init_neuron_state end
function init_neuron_st end
```

#### Ship: `LIFCell` (population LIF)

Canonical single-neuron LIF (current-based):

```math
\tau_m \dot V = -(V - V_{\mathrm{rest}}) + R_m\, I(t)
\quad\text{if } t \ge t_{\mathrm{ref,end}},
\quad V \leftarrow V_{\mathrm{reset}}\text{ when }V\ge V_{\mathrm{th}}
```

```julia
@concrete struct LIFCell <: AbstractSpikingNeuron
    τ_m          # membrane time constant
    V_rest
    V_reset
    V_th
    τ_ref        # refractory period
    R_m
end

LIFCell(; τ_m=0.01, V_rest=0.0, V_reset=0.0, V_th=1.0, τ_ref=0.002, R_m=1.0) =
    LIFCell(τ_m, V_rest, V_reset, V_th, τ_ref, R_m)

# Continuous state: u :: (n_units,) membrane voltages
state_dim(::LIFCell, n_units) = n_units

function init_neuron_st(::AbstractRNG, ::LIFCell, n_units::Integer)
    (;
        ref_until = fill(typemin(Float64), n_units),  # or zeros; see §6
        last_spike = fill(typemin(Float64), n_units),
    )
end
```

Network RHS (sketch; synaptic current from recurrent spikes may be instantaneous jumps in `reset!` or filtered synapses in `u` — **open question §8**):

```julia
function neuron_rhs!(du, u, p, t, neuron::LIFCell, st_neuron)
    # p holds: input_matrix, reservoir_matrix, input(t), maybe bias,
    #          and encoder-injected current I_enc(t) or synaptic state
    I = p.I_syn  # or computed from p.input(t), Win, W, …
    @inbounds for i in eachindex(u)
        if t < st_neuron.ref_until[i]
            du[i] = 0
        else
            du[i] = (-(u[i] - neuron.V_rest) + neuron.R_m * I[i]) / neuron.τ_m
        end
    end
    return nothing
end
```

Analytical ISI check (validation gate from #494): constant suprathreshold current \(I\), no refractory:

```math
T = \tau_m \ln\frac{R_m I - V_{\mathrm{rest}}}{R_m I - V_{\mathrm{th}}}
```

### 5.2 `AbstractInputEncoder`

```julia
"""
Maps external input u(t) into a drive the neuron population understands.

## Required interface

- `encode_current!(I, u_t, t, encoder, ps_enc, st_enc)` — write injected
  current (or conductance drive) for this time. Pure for CurrentInjection;
  may advance discrete RNG only through documented hooks for Poisson.
- `encoder_callbacks(encoder, tspan, data_or_rate_fn, st_enc)` — return
  `nothing` or a `DECallback` / tuple of callbacks (e.g. preset spike times).
- `init_encoder_st(rng, encoder, in_dims, res_dims)` — state including RNG.
- `reset_encoder_st!(st_enc, encoder)` — prepare for a new solve window.

## Determinism contract

- No global RNG.
- No RNG draws inside continuous `condition` functions.
- Poisson: prefer **precomputed event times** (or fixed-grid Bernoulli) so
  adaptive solver retries cannot re-consume RNG streams.
"""
abstract type AbstractInputEncoder end

function encode_current! end
function encoder_callbacks end
function init_encoder_st end
function reset_encoder_st! end
```

#### Ship: `CurrentInjection`

```julia
struct CurrentInjection <: AbstractInputEncoder end

# I = Win * u(t)  (and optionally bias); no callbacks, no RNG
function encode_current!(I, u_t, t, ::CurrentInjection, ps, st)
    mul!(I, ps.input_matrix, u_t)   # or fused into neuron_rhs!
    return nothing
end

encoder_callbacks(::CurrentInjection, args...) = nothing
init_encoder_st(::AbstractRNG, ::CurrentInjection, args...) = NamedTuple()
```

#### Ship: `PoissonRateEncoder`

```julia
@concrete struct PoissonRateEncoder <: AbstractInputEncoder
    scale           # maps |u| → firing rate (Hz)
    weight          # synaptic weight of each input spike onto postsynaptic I
end

function init_encoder_st(rng::AbstractRNG, ::PoissonRateEncoder, in_dims, res_dims)
    (;
        rng = copy(rng),           # or Random.seed! clone pattern used by Lux
        # filled at solve start:
        spike_times = Vector{Float64}[],   # per input channel
        next_idx = Int[],                  # queue pointers
    )
end
```

**Poisson determinism recipe (recommended):**

1. At the start of `_collectstates`, given ZOH windows and rate function \(\lambda_k(t) = \mathrm{softplus}(\mathrm{scale}\cdot u_k)\) (exact link TBD):
2. Draw inter-arrival times **once** from `st.encoder.rng` for the full `tspan` (or per window then concatenate).
3. Build `DiscreteCallback` / `PresetTimeCallback` over the union of event times; each event adds a pulse into a synaptic current buffer (or jumps `u`).
4. Advance / store the RNG in returned `st.encoder` so a second `collectstates` with a **fresh** st seed is independent, and the same st continues the stream if desired.
5. Never call `rand` from `condition`.

### 5.3 `AbstractSpikeReadout`

These are **feature extractors**, not the ridge `LinearReadout`.

```julia
"""
Maps continuous solution + spike events to a feature matrix (F × T).

## Required interface

- `feature_dim(ro, n_units)` — rows of the feature matrix.
- `init_spike_readout_st(rng, ro, n_units, n_samples)` — accumulators.
- `on_spike!(st_ro, unit, t, ro)` — called from threshold `affect!`.
- `on_sample!(out_col, u, t, sample_idx, ro, st_ro)` — write column at
  TerminalStateSampling times (end of each input window).
- `finalize_features(st_ro, ro)` → AbstractMatrix  (optional if columns
  written in-place during sampling).

## AR / train contracts

- SpikeCountReadout: teacher-forced predict only; document no AR.
- ExponentialFilterReadout / FilteredVoltageReadout: AR OK (features are
  continuous-valued and Markovian enough to roll forward).
"""
abstract type AbstractSpikeReadout end

function feature_dim end
function init_spike_readout_st end
function on_spike! end
function on_sample! end
```

#### Concrete sketches

```julia
@concrete struct SpikeCountReadout <: AbstractSpikeReadout
    # counts spikes in each sample window [t_{k-1}, t_k]
end
# feature_dim = n_units; on_spike! increments counter for current window;
# on_sample! writes counts and zeroes for next window.

@concrete struct ExponentialFilterReadout <: AbstractSpikeReadout
    τ   # filter time constant
end
# maintain ṡ = -s/τ + Σ_spikes δ(t-t_f); sample s(t_k) as features.
# Can be continuous state in u (preferred for AR) or callback-updated buffer.

@concrete struct FilteredVoltageReadout <: AbstractSpikeReadout
    # simply sample membrane u(t_k) — ContinuousESN-parity / diagnostics
end
# feature_dim = n_units; on_sample! copies u; on_spike! no-op.
```

**Default for LSM:** `ExponentialFilterReadout` (issue #494) — smooth, AR-capable.

### 5.4 Connectivity: `dale_sparse`

```julia
function dale_sparse(
    rng::AbstractRNG, ::Type{T}, n::Integer, n::Integer;
    excitatory_fraction = T(0.8),      # Maass-style 4:1
    sparsity = T(0.1),
    radius = T(1.0),                   # spectral scaling — exact target open
    ei_weight_ratio = T(1.0),          # |w_I| vs |w_E| balance knob
    return_sparse = false,
) where {T}
```

Semantics:

- Partition neurons into E / I once (stored or re-derived from a mask in `ps`?).
- Columns (or rows — **pick one convention and document**) of E neurons are ≥ 0; I neurons ≤ 0 (Dale).
- Scale to a usable dynamical regime (mean rate 5–30 Hz under random drive — validation harness).
- Spectral radius vs “balanced amplification” scaling is an **open question** (continuous ESN uses spectral radius; spiking literature often uses synaptic gain + E/I balance).

Optional later: `distance_dependent_sparse`.

### 5.5 `LSMCell` and `LSM` constructor

```julia
@concrete struct LSMCell <: AbstractSciMLProblemReservoir
    neuron <: AbstractSpikingNeuron
    encoder <: AbstractInputEncoder
    spike_readout <: AbstractSpikeReadout
    in_dims <: IntegerType
    out_dims <: IntegerType          # n_units (reservoir size)
    init_reservoir
    init_input
    init_bias
    init_state
    use_bias <: StaticBool
    tspan
    args                             # solve positional (solver first)
    kwargs                           # solve kwargs (dtmax, reltol, …)
end

function LSMCell(
    (in_dims, out_dims)::Pair;
    tspan,
    neuron = LIFCell(),
    encoder = CurrentInjection(),
    spike_readout = ExponentialFilterReadout(0.03),
    init_reservoir = dale_sparse,
    init_input = scaled_rand,
    init_bias = zeros32,
    init_state = zeros32,            # voltages
    use_bias = false,
    args = (),
    kwargs...,
)
```

```julia
@concrete struct LSM <:
    AbstractEchoStateNetwork{(:reservoir, :state_modifiers, :readout)}
    reservoir    # LSMCell
    state_modifiers
    readout      # LinearReadout(feature_dim => out_dims)
end

function LSM(
    in_dims::Integer, res_dims::Integer, out_dims::Integer,
    tspan, args...;
    neuron = LIFCell(),
    encoder = CurrentInjection(),
    spike_readout = ExponentialFilterReadout(0.03),
    state_modifiers = (),
    readout_activation = identity,
    init_reservoir = dale_sparse,
    # … neuron / encoder / init / solve kwargs …
    kwargs...,
)
    # stub in src/models/lsm.jl without extension;
    # real body in RCODEReservoirExt (ContinuousESN pattern)
end
```

**Usage sketch:**

```julia
using ReservoirComputing, OrdinaryDiffEqTsit5, SciMLBase, DataInterpolations

lsm = LSM(3, 200, 3, (0.0, 1000.0), Tsit5();
    neuron = LIFCell(; τ_m=0.02, τ_ref=0.002, V_th=1.0),
    encoder = CurrentInjection(),
    spike_readout = ExponentialFilterReadout(0.03),
    init_reservoir = dale_sparse,
    dtmax = 0.0005,          # ≤ τ_ref/4 recommended
    reltol = 1e-6, abstol = 1e-8,
)

ps, st = setup(rng, lsm)
ps, st = train(lsm, train_u, train_y, ps, st; washout=100)
yhat, st = predict(lsm, test_u, ps, st)
```

### 5.6 Parameters / states shape

```julia
# ps
(
  reservoir = (
    input_matrix = …,          # (res_dims × in_dims)
    reservoir_matrix = …,      # (res_dims × res_dims), Dale-compliant
    bias = …,                  # optional
    # encoder-specific params if any (e.g. Poisson scale as param?)
  ),
  state_modifiers = (…,),
  readout = (weight = …, [bias = …]),
)

# st
(
  reservoir = (
    neuron = (
      ref_until = Vector{T},     # length res_dims
      last_spike = Vector{T},
    ),
    encoder = (
      rng = AbstractRNG,         # Poisson only; empty NT otherwise
      spike_times = …,           # precomputed queue (ephemeral ok)
      next_idx = …,
    ),
    spike_readout = (
      filter_state = Vector{T},  # for ExponentialFilter
      window_counts = Vector{Int}, # for SpikeCount
      # …
    ),
    # optional: last continuous u for diagnostics / future warmup
  ),
  state_modifiers = (…,),
  readout = (…,),
)
```

`initialparameters` / `initialstates` on `LSMCell` populate the above; discrete clocks start “not refractory”.

---

## 6. Callback state threading (`st.reservoir`)

### 6.1 Ownership

| State | Lives in | Why |
| --- | --- | --- |
| Membrane `V` (and optional synaptic filters) | ODE `u` | Continuous, differentiated by solver |
| `ref_until`, `last_spike` | `st.reservoir.neuron` (arrays) | Discrete; updated only in `affect!` / `initialize` |
| Encoder RNG + event queue | `st.reservoir.encoder` | Determinism; not in global RNG |
| Spike-readout accumulators | `st.reservoir.spike_readout` | Features without bloating `sol` |
| `W_in`, `W`, bias | `ps.reservoir` | Trainable / fixed weights |

### 6.2 Lifecycle per `collectstates`

```text
1. reset_neuron_st!(st.reservoir.neuron)     # ref_until ← -Inf, etc.
2. reset_spike_readout_st!(…)
3. prepare_encoder!(st.reservoir.encoder, data, tspan, ps)
     - Poisson: draw all event times from st.encoder.rng ONCE
4. build CallbackSet(
     spike_cb,                    # VectorContinuousCallback
     encoder_cbs…,                # DiscreteCallback / PresetTime
     sample_hook?                 # optional DiscreteCallback at sample_ts
   )
5. ODEProblem(rhs!, u0, tspan, p; callback=cbset)   # fresh problem
6. solve(...; saveat=sample_ts, save_everystep=false, dense=false, kwargs...)
7. assemble feature matrix from spike_readout st (+ sol if FilteredVoltage)
8. apply state_modifiers
9. return features, updated st  (encoder.rng advanced; clocks left as-is or reset)
```

### 6.3 AR predict windows

Mirror ContinuousESN: `init` + `reinit!` per window.

- Callback `initialize` **does** re-fire on `reinit!` (probed) → use it to clear `ref_until` / window counters **if** per-window independence is desired.
- For true continuous rollout across windows, **do not** clear refractory clocks or filter state between windows; only reset when starting a new trajectory.
- SpikeCount AR: **unsupported** (document); force teacher-forced `predict(rc, data, ps, st)`.

### 6.4 Passing `st` into callbacks

Callbacks are plain functions; they cannot see Lux `st` unless closed over.

**Pattern:**

```julia
neuron_st = st.reservoir.neuron          # NamedTuple of arrays (mutable contents)
spike_st  = st.reservoir.spike_readout

affect! = function (integrator, idx)
    reset!(integrator, idx, neuron, neuron_st)
    on_spike!(spike_st, idx, integrator.t, spike_readout)
end
```

Arrays inside the NamedTuple are mutated in place; the NamedTuple shell in `st` stays identity-stable. This matches “mutable caches in `st`”.

**Avoid:** storing refractory state only in a `Ref` that is **not** hung off `st` (leaks across calls, breaks determinism tests that reuse st incorrectly).

---

## 7. Poisson encoder RNG and solver retries

### 7.1 Failure mode

Adaptive steppers re-evaluate continuous conditions many times (probe: 248 vs 1912 evals for different tols). Drawing `rand` in `condition` or even in RHS without care yields:

- non-reproducible spike trains at different tolerances;
- path-dependent noise if steps are rejected and retried (depending on where draws live).

### 7.2 Rules

1. **Precompute** Poisson event times (or fixed-grid Bernoulli decisions) up front from `st.encoder.rng`.
2. Drive the solver with **`DiscreteCallback` / `PresetTimeCallback`** on that finite set.
3. RHS and continuous conditions stay **deterministic** given `(u, p, t, st_discrete)`.
4. Same seed → identical event list → identical solve path (up to solver tol); document that changing `reltol` can still change threshold-crossing detection for the LIF callback (inherent to continuous events), so correctness tests use tight tols / `dtmax`.

### 7.3 `dtmax` default guidance

Issue #494: default `dtmax = τ_ref/4` to reduce missed spikes. Implementation options:

- LSM ctor injects `dtmax` into `kwargs` if user did not pass one (derived from `neuron.τ_ref`).
- Or document-only recommendation without changing solver defaults silently.

**Open:** Francesco prefers not to change continuous solver defaults without discussion (`MEMORY.md` doctrine). Prefer **document + test with explicit `dtmax`**, optionally soft-default only on `LSM` kwargs when absent.

---

## 8. Extension wiring (`RCODEReservoirExt`)

### 8.1 What changes in the extension

| Addition | Notes |
| --- | --- |
| `LSM(...)` real constructor | Same pattern as `ContinuousESN` |
| `_collectstates(::LSMCell, …)` | Fresh ODEProblem + CallbackSet + feature assembly |
| `_predict(::LSMCell, …)` teacher-forced | Via collectstates + LinearReadout |
| `_predict(::LSMCell, steps; initialdata)` AR | Windowed reinit; reject or special-case SpikeCount |
| Shared helpers | `_build_spike_callback`, `_prepare_poisson_events`, reuse ZOH + `_build_solve_params` |

### 8.2 Remake

For `LSMCell`, **do not depend on remake preserving callbacks** — build fresh problems (ContinuousESN style).

For users who put spiking dynamics in bare `SciMLProblemReservoir(prob, …)`:

- Remake **preserves** problem callbacks (verified).
- Document `initialize` requirements for discrete clocks.

### 8.3 Protected solve kwargs

Existing: `saveat`, `save_everystep`, `dense`.  
LSM should keep the same rejections. `callback` should **not** be a user-facing solve kwarg on `LSM` (owned by the cell); if passed, error with a clear message to avoid double CallbackSets.

### 8.4 Feature dimension vs `LinearReadout`

`LinearReadout(feature_dim(spike_readout, res_dims) => out_dims)` — must use spike readout’s `feature_dim`, not raw `res_dims`, when they differ (v1 they match for all three shipped readouts).

---

## 9. Validation plan (from #494, test-group aligned)

Groups: `All`, `Core`, `nopre` (+ existing `Extensions` / `Layers` / `Models` layout).

| Test | Group | Assertion |
| --- | --- | --- |
| Single LIF ISI | Extensions / Layers | analytical \(T\) within 1% |
| Refractory exactness | Extensions | no spike in \((t_f, t_f+τ_{\mathrm{ref}})\) within callback tol |
| Atomic reset | Extensions | post-event \(V = V_{\mathrm{reset}}\) |
| `dale_sparse` E/I ratio | Core (no DiffEq) | fraction + sign pattern |
| Determinism same seed | Extensions | identical spike counts / features |
| Rich vs quiet regime | Extensions (slow ok) | mean rate band under scaled \(W\) |
| ESP-like | Extensions | van Rossum or feature L2 convergence from two \(V_0\) |
| Shape parity | Extensions | `collectstates` → `(F, T)` |
| SpikeCount AR rejection | Extensions | clear error |
| Lorenz / MG smoke | Extensions / docs | honest NRMSE vs ContinuousESN (~2–3× worse expected) |

Julia **1.10 LTS + 1.12** both green. Formatter: **Runic**.

---

## 10. Docs plan

- `docs/src/api/models.md` — `LSM`
- `docs/src/api/layers.md` — `LIFCell`, `AbstractSpikingNeuron`, `LSMCell`
- New `docs/src/api/encoders.md` — encoders + spike readouts
- `docs/src/tutorials/lsm.md` — Maass framing, LIF math, one regression + one classification, capacity caveat
- `docs/src/refs.bib` — Maass 2002; Legenstein & Maass 2007; Vreeken 2003; Gerstner & Kistler LIF chapter
- Docstring bar: human-facing only; one citation; no dispatch narrative (ContinuousESN PR #456 standard)

---

## 11. Implementation phases (suggested, not started)

| Phase | Scope | Depends |
| --- | --- | --- |
| 0 | This design doc + Francesco API go/no-go on open questions | — |
| 1 | `AbstractSpikingNeuron` + `LIFCell` + unit ISI/refractory tests (can use raw DiffEq without full LSM) | design sign-off |
| 2 | Encoders + `dale_sparse` + spike readouts (unit) | phase 1 |
| 3 | `LSMCell` + extension `_collectstates` + shape/determinism tests | phase 2 |
| 4 | `LSM` model + train/predict + SpikeCount AR guard | phase 3 |
| 5 | Tutorial + API docs + Lorenz/classification harness | phase 4 |
| 6 (later) | Perf (sparse callbacks, GPU), more neurons, STDP | after land |

Estimate from #494: ~3 weeks solid, ~1500–2500 LOC code+tests+docs; single PR or 2 stacked (infra + model/demos).

---

## 12. Open questions (need decisions before coding)

### Q1 — Neuron type name

`LIFCell` collides conceptually with existing `LIFESN` (Local Information Flow). Options:

- **(a)** `LIFCell` / `LIFNeuron` anyway (disambiguate in docs) — *issue #494 default*
- **(b)** `LeakyIFCell` / `SpikingLIFCell` to reduce confusion
- **(c)** Rename nothing; accept doc emphasis

**Recommendation:** (a) with loud docs, unless Francesco prefers (b).

### Q2 — Recurrent spikes: instantaneous jumps vs synaptic ODE

When unit \(j\) spikes, how does unit \(i\) feel it?

- **(a)** Instantaneous `u[i] += W[i,j]` in `affect!` (delta synapses; simple, discontinuous)
- **(b)** Current-based exponential synapse: extra continuous state \(I_{\mathrm{syn}}\), jump \(I\) on spike, \(\dot I = -I/τ_{\mathrm{syn}}\)
- **(c)** Conductance-based (more bio, more state)

**Recommendation for v1:** (b) with small \(τ_{\mathrm{syn}}\) default, or (a) if we want minimal state — need Francesco preference. (a) is less state but harder on adaptive solvers; (b) is smoother and AR-friendlier.

### Q3 — Where does synaptic state live?

If (b): append to `u` (`state_dim = 2n`) vs separate `p` buffers updated only in callbacks. Appending to `u` is cleaner for `reinit!` carry.

### Q4 — Default encoder / readout / connectivity

Issue defaults: `CurrentInjection`, `ExponentialFilterReadout`, `dale_sparse`. Confirm before coding.

### Q5 — `dtmax` soft default

Inject `dtmax=τ_ref/4` when absent vs document-only? Doctrine says ask before default solver changes — **recommend document + LSM docstring example**, not silent global default.

### Q6 — Spike feature path vs `TerminalStateSampling`

Is `AbstractSpikeReadout` a new sampler subtype (`AbstractSampler`) or a separate seam on `LSMCell` only?

**Recommendation:** keep `AbstractSampler` for trajectory→matrix policies used by generic `SciMLProblemReservoir`; put spike readouts **on `LSMCell` only** so generic continuous reservoirs stay non-spiking. Avoid forcing spike semantics into `SciMLProblemReservoir`.

### Q7 — E/I mask persistence

Store excitatory mask in `ps.reservoir` (reproducible, serializable) vs re-derive from sign pattern of `W` after `dale_sparse`?

**Recommendation:** signs of `W` are the source of truth post-init; no separate mask unless we need STDP later.

### Q8 — Continuous AR warmup

Same open design as ContinuousESN (#476): cold `u0` vs terminal carry vs `warmup_data`. **Do not block LSM landing** on warmup; default cold zeros + document.

### Q9 — DiffEq simultaneous multi-event `affect!` API

Probes hit edge cases when multiple units cross threshold together. Need a small harness that freezes the intended `affect!(integrator, idx)` contract (scalar vs vector `idx`) before locking population reset code.

### Q10 — Package layout / exports

New submodules vs flat `ReservoirComputing` exports? Repo is currently flat includes. **Recommend flat** includes + exports, matching ContinuousESN; no new submodule.

### Q11 — Weakdeps

Does LSM need anything beyond current `RCODEReservoirExt` weakdep (`DataInterpolations`)? Callbacks live in SciMLBase (already a hard dep). **No new weakdep expected.**

### Q12 — Francesco API polish gate

Public names (`LSM`, `LIFCell`, `dale_sparse`, encoder/readout type names) should get a quick maintainer nod before the implementation PR to avoid rename churn (doctrine: Francesco owns public naming).

---

## 13. Risks and mitigations (updated)

| Risk | Mitigation |
| --- | --- |
| Callback × adaptive stepper misses spikes | Explicit `dtmax`; tight-tol unit tests; optional fixed-step smoke |
| Poisson non-determinism | Precomputed events; RNG only in `st.encoder`; never in `condition` |
| remake drops callbacks | **False alarm** for default remake; still prefer fresh problems for LSMCell |
| Shared mutable callback state | Reset via `initialize` + `st`-owned arrays; no free-floating Refs |
| SpikeCount AR | Constructor/docs + hard error on AR predict |
| Capacity vs ContinuousESN | Tutorial states expected ~2–3× worse NRMSE honestly |
| Name clash LIFESN vs LIF | Docs + possibly `LeakyIF` naming (Q1) |
| Scope creep (STDP, HH, GPU) | Seams only; deferred list in #494 |

---

## 14. Decision log

| Date | Decision | Who |
| --- | --- | --- |
| 2026-08-04 | remake preserves problem callbacks (same identity); mutable state is the real hazard | design probe |
| 2026-08-04 | Prefer ContinuousESN-style fresh ODEProblem + solve-time CallbackSet for LSMCell | design |
| 2026-08-04 | Name `LIFCell` + clear docs vs LIFESN | Francesco |
| 2026-08-04 | Exponential synaptic current in continuous state | Francesco |
| 2026-08-04 | `dtmax` document-only (no soft default) | Francesco |
| 2026-08-04 | Defaults: CurrentInjection, ExponentialFilterReadout, dale_sparse | Francesco |
| 2026-08-05 | Implementation landed on branch (not necessarily merged) | — |

---

## 15. References (to add to `refs.bib` at implementation time)

- Maass, Natschläger, Markram (2002) — Liquid State Machine
- Legenstein & Maass (2007) — edge of chaos / separation
- Vreeken (2003) — LSM review
- Gerstner & Kistler — LIF chapter
- Lukoševičius (2012) — continuous ESN eq. (5) (already cited for ContinuousESN)

---

## 16. What was *not* done (phase boundary)

- No `src/**` implementation files added.
- No tests added.
- No commits / PR.
- No Francesco sign-off yet on Q1–Q12.

**Next step after review:** resolve open questions (especially Q1, Q2, Q4, Q5, Q12), then Phase 1 (LIF neuron + ISI tests) on this branch.
