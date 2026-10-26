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
`u(t-H+1), …, u(t-1)`, the wrapper computes, at every time step `t`:

```math
\begin{aligned}
    s_0 &= \mathrm{InitStates} \\
    (y_{t-H+1}, s_1) &= F_{\text{cell}}\!\left(u(t-H+1), s_0\right) \\
    (y_{t-H+2}, s_2) &= F_{\text{cell}}\!\left(u(t-H+2), s_1\right) \\
    &\ \vdots \\
    (y_{t-1}, s_{H-1}) &= F_{\text{cell}}\!\left(u(t-1), s_{H-2}\right) \\
    (y_t, s_H) &= F_{\text{cell}}\!\left(u(t), s_{H-1}\right)
\end{aligned}
```

`InitStates` is the initial state produced by the wrapped cell's own initializer
(`init_state`). Every window starts from it: it is sampled once, on the first call, and
stored in the layer state, so it is the same at every time step. The state carried over
from the previous step is never used to start the replay, so the output at time `t`
depends only on `u(t-H+1), …, u(t)`.

## Arguments

  - `in_dims`: Input dimension (passed to `cell` constructor).
  - `out_dims`: Output dimension / state dimension (passed to `cell` constructor).
  - `cell`: A cell constructor or callable that accepts `in_dims => out_dims` plus
    `args...; kwargs...`.
  - `lookback_horizon`: Number of most recent inputs used to reconstruct the
    recurrent state at each step. Must satisfy `lookback_horizon ≥ 1`.
    - `lookback_horizon = 1` disables replay (no history); the wrapper reduces
      to a single wrapped-cell step from the initial state.
  - `args...`: Positional arguments for the `cell`.

## Keyword arguments

  - `init_buffer`: Initializer for the input history buffer (used before enough
    inputs have been observed). Default: `zeros32`.
  - `kwargs...`: Keyword arguments forwarded to the wrapped `cell` constructor.

## Inputs

  - **Case 1:** `x :: AbstractArray (in_dims, batch)`
    Same as Case 2, without a carried state.
  - **Case 2:** `(x, states)` where `states` is the carried state tuple of a previous
    call (e.g. `(h,)`, `(h, c)`, …), as passed by [`StatefulLayer`](@ref).
    `states` is ignored: it does not influence the output.

In both cases, the forward returns `((y, states_out), st_out)` where `st_out`
contains the updated wrapped-cell states plus the updated input buffer.

## Returns

  - Output `y` (as produced by the wrapped cell at the current step).
  - State tuple `states_out` (as produced by the wrapped cell at the current step).
    It is returned as the new carry, but it is never used to start a later window.
  - Updated layer state (NamedTuple).

## Parameters

Created by `initialparameters(rng, lif)`:

  - `cell` — parameters of the wrapped cell.

## States

Created by `initialstates(rng, lif)`:

  - `rng`: an owned copy of the RNG given to `initialstates`. The first call samples
    `start_states` and `input_buffer` from a copy of it and returns the advanced copy,
    so the RNG of the input state is never advanced.
  - `input_buffer`: a length `lookback_horizon-1` history of past inputs, stored as
    copies; initialized on first call using `init_buffer`.
  - `start_states`: the state tuple every window starts from; sampled on the first call
    with the wrapped cell's `init_state`. A state saved without this entry, as by earlier
    versions, is accepted: the next call samples it from a copy of `rng`, keeps the saved
    `input_buffer`, and returns the advanced `rng`.
  - `cell`: the wrapped cell’s own internal state (NamedTuple).
"""
@concrete struct LocalInformationFlow <: AbstractEchoStateNetworkCell
    cell
    lookback_horizon <: IntegerType
    init_buffer
end

__cell_out_dims(lif::LocalInformationFlow) = __cell_out_dims(lif.cell)

function LocalInformationFlow(
        cell, (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType},
        lookback_horizon::IntegerType, args...; init_buffer = zeros32, kwargs...
    )
    built_cell = cell(in_dims => out_dims, args...; kwargs...)

    return LocalInformationFlow(built_cell, lookback_horizon, init_buffer)
end

function Base.show(io::IO, lif::LocalInformationFlow)
    print(io, "LocalInformationFlow(")
    show(io, lif.cell)
    print(io, ", ", lif.lookback_horizon)
    lif.init_buffer === zeros32 || print(io, ", init_buffer=", lif.init_buffer)
    return print(io, ")")
end

function initialparameters(rng::AbstractRNG, lif::LocalInformationFlow)
    return (cell = initialparameters(rng, lif.cell),)
end

function initialstates(rng::AbstractRNG, lif::LocalInformationFlow)
    return (
        rng = __owned_sample_replicate(rng),
        input_buffer = nothing,
        start_states = nothing,
        cell = initialstates(rng, lif.cell),
    )
end


function __init_input_buffer(
        rng::AbstractRNG, lif::LocalInformationFlow, inp::AbstractArray
    )
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
    return lif((inp, nothing), ps, st)
end

function __init_window_states(lif::LocalInformationFlow, inp::AbstractArray, st::NamedTuple)
    rng = __owned_replicate(st.rng)
    start_states = get(st, :start_states, nothing)
    if start_states === nothing
        start_states = init_hidden_states(rng, lif, inp)
    end
    input_buffer = st.input_buffer
    if input_buffer === nothing
        input_buffer = __init_input_buffer(rng, lif, inp)
    end

    return merge(st, (; rng, input_buffer, start_states))
end

function (lif::LocalInformationFlow)((inp, _)::Tuple, ps, st::NamedTuple)
    if st.input_buffer === nothing || get(st, :start_states, nothing) === nothing
        st = __init_window_states(lif, inp, st)
    end
    buf = st.input_buffer
    cell_ps = ps.cell
    cell_st = st.cell
    states_cur = st.start_states
    for inp_prev in buf
        (_, states_cur), cell_st = lif.cell((inp_prev, states_cur), cell_ps, cell_st)
    end
    (h_new, states_out), cell_st = lif.cell((inp, states_cur), cell_ps, cell_st)
    newbuf = isempty(buf) ? buf : (Base.tail(buf)..., copy(inp))

    return (h_new, states_out), merge(st, (; input_buffer = newbuf, cell = cell_st))
end
