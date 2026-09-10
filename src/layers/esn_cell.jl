"""
    AbstractEchoStateNetworkCell <: AbstractReservoirRecurrentCell

Developer interface for an echo-state-network recurrent cell with the shared
ESN parameter and state initialization implementation.

## Required fields

The generic methods require these fields:

- `in_dims`: input feature dimension.
- `out_dims`: reservoir-state dimension.
- `init_input(rng, out_dims, in_dims)`: input-matrix initializer.
- `init_reservoir(rng, out_dims, out_dims)`: recurrent-matrix initializer.
- `init_state(rng, out_dims, batch_size)`: initial hidden-state initializer.
- `use_bias`: `Static.True()` or `Static.False()`. When true,
  `init_bias(rng, out_dims)` is also required.

## Extension contract

Subtypes inherit `LuxCore.initialparameters` and `LuxCore.initialstates` from
this interface. Those methods create `input_matrix` and `reservoir_matrix`, an
optional `bias`, and a replicated RNG state. Implement the recurrent call form
from [`AbstractReservoirRecurrentCell`](@ref): given `(x, (carry,))`, return
`((output, (next_carry,)), st_new)`. The generic one-input method initializes a
hidden state with `init_state` and delegates to that form.

`input_matrix` must have shape `(out_dims, in_dims)`, `reservoir_matrix` must
have shape `(out_dims, out_dims)`, and every carry must be compatible with the
chosen `out_dims` and batch dimension.

## Example

```julia
struct MyESNCell <: AbstractEchoStateNetworkCell
    in_dims
    out_dims
    init_input
    init_reservoir
    init_bias
    init_state
    use_bias
end
```
"""
abstract type AbstractEchoStateNetworkCell <: AbstractReservoirRecurrentCell end

@doc raw"""
    ESNCell(in_dims => out_dims, [activation];
        use_bias=false, init_bias=rand32,
        init_reservoir=rand_sparse, init_input=scaled_rand,
        init_state=randn32, leak_coefficient=1.0,
        use_feedback=false, feedback_dims=0, init_feedback=scaled_rand)

Echo State Network (ESN) recurrent cell with optional leaky integration
and optional output feedback [Jaeger2004](@cite).

## Equations

```math
\begin{aligned}
    \mathbf{x}(t) &= (1-\alpha)\, \mathbf{x}(t-1)
        + \alpha\, \phi\!\left(\mathbf{W}_{\text{in}}\, \mathbf{u}(t)
        + \mathbf{W}_r\, \mathbf{x}(t-1)
        + \mathbf{W}_{\mathrm{fb}}\, \mathbf{y}(t-1)
        + \mathbf{b} \right)
\end{aligned}
```

The \(\mathbf{W}_{\mathrm{fb}}\mathbf{y}(t-1)\) term is included only when
`use_feedback=true`. During training, \(\mathbf{y}\) is the teacher signal;
after training it is the model's previous output.

## Arguments

  - `in_dims`: Input dimension.
  - `out_dims`: Reservoir (hidden state) dimension.
  - `activation`: Activation function. Default: `tanh_fast`.

## Keyword arguments

  - `use_bias`: Whether to include a bias term. Default: `false`.
  - `init_bias`: Initializer for the bias. Used only if `use_bias=true`.
      Default is `rand32`.
  - `init_reservoir`: Initializer for the reservoir matrix `W_res`.
    Default is [`rand_sparse`](@ref).
  - `init_input`: Initializer for the input matrix `W_in`.
    Default is [`scaled_rand`](@ref).
  - `init_state`: Initializer for the hidden state when an external
    state is not provided. Default is `randn32`.
  - leak_coefficient: Leak rate `α ∈ (0,1]`. Can be a scalar (uniform leak)
    or a vector of size `out_dims` (heterogeneous leak rates). Default: `1.0`.
  - `use_feedback`: Whether to include output feedback `W_fb`. Default: `false`.
  - `feedback_dims`: Width of the feedback signal (readout output dimension).
    Required and must be positive when `use_feedback=true`. Default: `0`.
  - `init_feedback`: Initializer for `W_fb`. Used only if `use_feedback=true`.
    Default is [`scaled_rand`](@ref).

## Inputs

  - **Case 1:** `x :: AbstractArray (in_dims, batch)`
    A fresh state is created via `init_state`; the call is forwarded to Case 2.
  - **Case 2:** `(x, (h,))` where `h :: AbstractArray (out_dims, batch)`
    Computes the update and returns the new state.
  - **Case 3:** `((x, y), (h,))` when `use_feedback=true`, with
    `y :: AbstractArray (feedback_dims, batch)` the previous output
    (or teacher). A first call `(x, y)` with no carry is also accepted.

In all cases, the forward returns `((h_new, (h_new,)), st_out)` where `st_out`
contains any updated internal state.

## Returns

  - Output/hidden state `h_new :: out_dims` and state tuple `(h_new,)`.
  - Updated layer state (NamedTuple).

## Parameters

Created by `initialparameters(rng, esn)`:

  - `input_matrix :: (out_dims × in_dims)` — `W_in`
  - `reservoir_matrix :: (out_dims × out_dims)` — `W_res`
  - `bias :: (out_dims,)` — present only if `use_bias=true`
  - `feedback_matrix :: (out_dims × feedback_dims)` — `W_fb`,
    present only if `use_feedback=true`

## States

Created by `initialstates(rng, esn)`:

  - `rng`: a replicated RNG used to sample initial hidden states when needed.
"""
@concrete struct ESNCell <: AbstractEchoStateNetworkCell
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_reservoir
    init_input
    init_feedback
    init_state
    leak_coefficient
    feedback_dims <: IntegerType
    use_bias <: StaticBool
    use_feedback <: StaticBool
end

function ESNCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType},
        activation = tanh_fast; use_bias::BoolType = False(), init_bias = zeros32,
        init_reservoir = rand_sparse, init_input = scaled_rand,
        init_state = randn32,
        leak_coefficient::Union{AbstractFloat, AbstractVector} = 1.0,
        use_feedback::BoolType = False(),
        feedback_dims::IntegerType = 0,
        init_feedback = scaled_rand
    )

    if isa(leak_coefficient, AbstractVector)
        length(leak_coefficient) == out_dims || throw(
            DimensionMismatch(
                "leak_coefficient must have length out_dims=$out_dims, " *
                    "got $(length(leak_coefficient))."
            )
        )
    end

    use_fb = static(use_feedback)
    if known(use_fb)
        Int(feedback_dims) > 0 || throw(
            ArgumentError(
                "feedback_dims must be positive when use_feedback=true, " *
                    "got $feedback_dims"
            )
        )
    end

    return ESNCell(
        activation, in_dims, out_dims, init_bias, init_reservoir,
        init_input, init_feedback, init_state, leak_coefficient,
        feedback_dims, static(use_bias), use_fb
    )
end

function initialparameters(rng::AbstractRNG, esn::AbstractEchoStateNetworkCell)
    ps = (
        input_matrix = esn.init_input(rng, esn.out_dims, esn.in_dims),
        reservoir_matrix = esn.init_reservoir(rng, esn.out_dims, esn.out_dims),
    )
    if has_bias(esn)
        ps = merge(ps, (bias = esn.init_bias(rng, esn.out_dims),))
    end
    if has_feedback(esn)
        ps = merge(
            ps,
            (
                feedback_matrix = esn.init_feedback(
                    rng, esn.out_dims, esn.feedback_dims
                ),
            ),
        )
    end
    return ps
end

function initialstates(rng::AbstractRNG, esn::AbstractEchoStateNetworkCell)
    return (rng = sample_replicate(rng),)
end

function init_hidden_states(rng::AbstractRNG, cell::AbstractEchoStateNetworkCell, inp::AbstractArray)
    return (init_hidden_state(rng, cell, inp),)
end

function (esn::AbstractEchoStateNetworkCell)(inp::AbstractArray, ps, st::NamedTuple)
    rng = replicate(st.rng)
    hidden_state = init_hidden_state(rng, esn, inp)
    return esn((inp, (hidden_state,)), ps, merge(st, (; rng)))
end

function (esn::ESNCell)(
        inp::Tuple{<:AbstractArray, <:AbstractArray}, ps, st::NamedTuple
    )
    rng = replicate(st.rng)
    hidden_state = init_hidden_state(rng, esn, first(inp))
    return esn((inp, (hidden_state,)), ps, merge(st, (; rng)))
end

function __esncell_step(esn::ESNCell, inp, hidden_state, ps, st, feedback)
    T = eltype(inp)
    bias = safe_getproperty(ps, Val(:bias))
    preact = dense_bias(ps.input_matrix, inp, nothing) .+
        dense_bias(ps.reservoir_matrix, hidden_state, bias)
    if feedback !== nothing
        preact = preact .+ dense_bias(ps.feedback_matrix, feedback, nothing)
    end
    candidate_h = esn.activation.(preact)
    lc = __format_leak(T, esn.leak_coefficient)
    h_new = __one_minus_leak(T, lc) .* hidden_state .+ lc .* candidate_h
    return (h_new, (h_new,)), st
end

function (esn::ESNCell)((inp, (hidden_state,))::InputType, ps, st::NamedTuple)
    has_feedback(esn) && throw(
        ArgumentError(
            "ESNCell with use_feedback=true expects input (u, y_prev), got a single array"
        )
    )
    return __esncell_step(esn, inp, hidden_state, ps, st, nothing)
end

function (esn::ESNCell)(
        ((inp, feedback), (hidden_state,))::FeedbackInputType, ps, st::NamedTuple
    )
    has_feedback(esn) || throw(
        ArgumentError("ESNCell received (u, y_prev) but use_feedback=false")
    )
    return __esncell_step(esn, inp, hidden_state, ps, st, feedback)
end

function __format_leak(::Type{T}, leak::Number) where {T <: Number}
    return convert(T, leak)
end

function __format_leak(::Type{T}, leak::AbstractArray) where {T <: Number}
    return reshape(convert.(T, leak), :, 1)
end

function __one_minus_leak(::Type{T}, leak::Number) where {T <: Number}
    return one(T) - leak
end

function __one_minus_leak(::Type{T}, leak::AbstractArray) where {T <: Number}
    return one(T) .- leak
end

function Base.show(io::IO, esn::ESNCell)
    print(io, "ESNCell($(esn.in_dims) => $(esn.out_dims)")
    if !(esn.leak_coefficient isa Number && esn.leak_coefficient == 1.0)
        print(io, ", leak_coefficient=$(esn.leak_coefficient)")
    end
    has_bias(esn) || print(io, ", use_bias=false")
    if has_feedback(esn)
        print(io, ", use_feedback=true, feedback_dims=$(esn.feedback_dims)")
    end
    return print(io, ")")
end
