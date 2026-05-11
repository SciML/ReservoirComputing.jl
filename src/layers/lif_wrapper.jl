@doc raw"""
    LocalInformationFlow(cell, in_dims => out_dims, lookback_horizon, args...;
        init_buffer=zeros32, kwargs...)

Wrap an echo-state cell to enforce a local-information horizon: the output at time `t`
is computed by reconstructing the recurrent state using only the most recent
`lookback_horizon` inputs [Liu2025](@cite).

The wrapper is generic over wrapped cells that support the standard
ReservoirComputing/Lux “explicit state” calling convention: `(x, states)`, so either
`(x, (h_state, ))` or `(x, (h_state, c_state))`.

## Equations

Let `F_cell(u, s)` denote one step of the wrapped cell (shared parameters), returning
`(y, s⁺)`. With horizon `H = lookback_horizon` and the input history
`u(t-H+1), …, u(t-1)`, the wrapper computes:

```math
\begin{aligned}
    s_0 &= \mathrm{InitStates}(t) \\
    (y_{t-H+1}, s_1) &= F_{\text{cell}}\!\left(u(t-H+1), s_0\right) \\
    (y_{t-H+2}, s_2) &= F_{\text{cell}}\!\left(u(t-H+2), s_1\right) \\
    &\ \vdots \\
    (y_{t-1}, s_{H-1}) &= F_{\text{cell}}\!\left(u(t-1), s_{H-2}\right) \\
    (y_t, s_H) &= F_{\text{cell}}\!\left(u(t), s_{H-1}\right)
\end{aligned}
```
## Arguments

  - `in_dims`: Input dimension (passed to `cell` constructor).
  - `out_dims`: Output dimension / state dimension (passed to `cell` constructor).
  - `cell`: A cell constructor or callable that accepts `in_dims => out_dims` plus
    `args...; kwargs...`.
  - `lookback_horizon`: Number of most recent inputs used to reconstruct the
    recurrent state at each step. Must satisfy `lookback_horizon ≥ 1`.
    - `lookback_horizon = 1` disables replay (no history); the wrapper reduces
      to a single wrapped-cell step.
  - `args...`: Positional arguments for the `cell`.

## Keyword arguments

  - `init_buffer`: Initializer for the input history buffer (used before enough
    inputs have been observed). Default: `zeros32`.
  - `kwargs...`: Keyword arguments forwarded to the wrapped `cell` constructor.

## Inputs

  - **Case 1:** `x :: AbstractArray (in_dims, batch)`
    The wrapper constructs an initial state using the wrapped cell’s default
    initialization and forwards the call to Case 2.
  - **Case 2:** `(x, states)` where `states` is a tuple of one or more state
    components as expected by the wrapped cell (e.g. `(h,)`, `(h, c)`, …).

In both cases, the forward returns `((y, states_out), st_out)` where `st_out`
contains the updated wrapped-cell states plus the updated input buffer.

## Returns

  - Output `y` (as produced by the wrapped cell at the current step).
  - State tuple `states_out` (as produced by the wrapped cell at the current step).
  - Updated layer state (NamedTuple).

## Parameters

Created by `initialparameters(rng, lif)`:

  - `cell` — parameters of the wrapped cell.

## States

Created by `initialstates(rng, lif)`:

  - `rng`: a replicated RNG used to initialize the input buffer and/or any
    wrapped-cell initialization that relies on sampling.
  - `input_buffer`: a length `lookback_horizon-1` history of past inputs; initialized
    on first call using `init_buffer`.
  - `cell`: the wrapped cell’s own internal state (NamedTuple).
"""
@concrete struct LocalInformationFlow <: AbstractEchoStateNetworkCell
    cell
    lookback_horizon <: IntegerType
    init_buffer
end

_cell_out_dims(lif::LocalInformationFlow) = _cell_out_dims(lif.cell)

function LocalInformationFlow(
        cell, (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType},
        lookback_horizon::IntegerType, args...; init_buffer = zeros32, kwargs...
    )
    built_cell = cell(in_dims => out_dims, args...; kwargs...)

    return LocalInformationFlow(built_cell, lookback_horizon, init_buffer)
end

function initialparameters(rng::AbstractRNG, lif::LocalInformationFlow)
    return (cell = initialparameters(rng, lif.cell),)
end

function initialstates(rng::AbstractRNG, lif::LocalInformationFlow)
    return (
        rng = sample_replicate(rng),
        input_buffer = nothing,
        cell = initialstates(rng, lif.cell),
    )
end


function _init_input_buffer(rng::AbstractRNG, lif::LocalInformationFlow, inp::AbstractArray)
    n = max(lif.lookback_horizon - 1, 0)
    n == 0 && return ()
    buf = ntuple(_ -> similar(inp), n)
    in_dims = size(inp, 1)
    batch = ndims(inp) == 1 ? 1 : size(inp, 2)
    for b in buf
        b .= lif.init_buffer(rng, in_dims, batch)
    end

    return buf
end

function init_hidden_states(rng::AbstractRNG, lif::LocalInformationFlow, inp::AbstractArray)
    return init_hidden_states(rng, lif.cell, inp)
end

function (lif::LocalInformationFlow)(inp::AbstractArray, ps, st::NamedTuple)
    rng = replicate(st.rng)
    states = init_hidden_states(rng, lif, inp)

    return lif((inp, states), ps, merge(st, (; rng)))
end

function (lif::LocalInformationFlow)((inp, states)::Tuple, ps, st::NamedTuple)
    buf = st.input_buffer
    if buf === nothing
        buf = _init_input_buffer(st.rng, lif, inp)
    end
    cell_ps = ps.cell
    cell_st = st.cell
    states_cur = states
    for inp_prev in buf
        (_, states_cur), cell_st = lif.cell((inp_prev, states_cur), cell_ps, cell_st)
    end
    (h_new, states_out), cell_st = lif.cell((inp, states_cur), cell_ps, cell_st)
    newbuf = isempty(buf) ? buf : (Base.tail(buf)..., inp)

    return (h_new, states_out), merge(st, (; input_buffer = newbuf, cell = cell_st))
end
