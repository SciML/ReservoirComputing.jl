@doc raw"""
    MemoryCell(in_dims => out_dims, [activation];
        use_bias=False(), init_bias=zeros32,
        init_reservoir=simple_cycle, init_input=scaled_rand,
        init_state=randn32)

Linear memory reservoir cell used in Reservoir Memory Networks
[Gallicchio2024b](@cite) and Residual Reservoir Memory Networks
[Ceni2025b](@cite).

The memory reservoir applies a linear transformation driven solely by the
external input. Its recurrent kernel is, by default, a cyclic shift
permutation (an orthogonal matrix) via [`simple_cycle`](@ref) with unit
weight, so the layer behaves as a deterministic delay line over the input.

## Equations

```math
\begin{aligned}
    \mathbf{m}(t) &= \phi\!\left(\mathbf{W}_{\text{in}}\, \mathbf{u}(t)
        + \mathbf{C}\, \mathbf{m}(t - 1) + \mathbf{b}\right)
\end{aligned}
```

where ``\phi`` is `identity` by default and ``\mathbf{C}`` is a cyclic
orthogonal matrix.

## Arguments

  - `in_dims`: Input dimension.
  - `out_dims`: Reservoir (hidden state) dimension.
  - `activation`: Activation function. Default: `identity`.

## Keyword arguments

  - `use_bias`: Whether to include a bias term. Default: `false`.
  - `init_bias`: Initializer for the bias. Used only if `use_bias=true`.
    Default is `zeros32`.
  - `init_reservoir`: Initializer for the recurrent memory kernel `C`.
    Default is [`simple_cycle`](@ref) with unit cycle weight.
  - `init_input`: Initializer for the input matrix `W_in`.
    Default is [`scaled_rand`](@ref).
  - `init_state`: Initializer for the hidden state when an external state
    is not provided. Default is `randn32`.

## Inputs

  - **Case 1:** `x :: AbstractArray (in_dims, batch)`
    A fresh state is created via `init_state`; the call is forwarded to Case 2.
  - **Case 2:** `(x, (m,))` where `m :: AbstractArray (out_dims, batch)`
    Computes the update and returns the new state.

## Returns

  - Output/hidden state `m_new :: out_dims` and state tuple `(m_new,)`.
  - Updated layer state (NamedTuple).

## Parameters

  - `input_matrix :: (out_dims × in_dims)` — `W_in`
  - `reservoir_matrix :: (out_dims × out_dims)` — `C`
  - `bias :: (out_dims,)` — present only if `use_bias=true`

## States

  - `rng`: a replicated RNG used to sample initial hidden states when needed.
"""
@concrete struct MemoryCell <: AbstractEchoStateNetworkCell
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_reservoir
    init_input
    init_state
    use_bias <: StaticBool
end

_default_memory_reservoir(rng::AbstractRNG, dims::Integer...) =
    simple_cycle(rng, dims...; cycle_weight = 1.0f0)

function MemoryCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType},
        activation = identity; use_bias::BoolType = False(), init_bias = zeros32,
        init_reservoir = _default_memory_reservoir, init_input = scaled_rand,
        init_state = randn32
    )
    return MemoryCell(
        activation, in_dims, out_dims, init_bias, init_reservoir,
        init_input, init_state, static(use_bias)
    )
end

function (mc::MemoryCell)((inp, (hidden_state,))::InputType, ps, st::NamedTuple)
    bias = safe_getproperty(ps, Val(:bias))
    win_inp = dense_bias(ps.input_matrix, inp, nothing)
    w_state = dense_bias(ps.reservoir_matrix, hidden_state, bias)
    m_new = mc.activation.(win_inp .+ w_state)
    return (m_new, (m_new,)), st
end

function Base.show(io::IO, mc::MemoryCell)
    print(io, "MemoryCell($(mc.in_dims) => $(mc.out_dims)")
    if mc.activation !== identity
        print(io, ", $(mc.activation)")
    end
    has_bias(mc) || print(io, ", use_bias=false")
    return print(io, ")")
end
