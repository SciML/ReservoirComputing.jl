abstract type AbstractEchoStateNetworkCell <: AbstractReservoirRecurrentCell end

@doc raw"""
    ES2NCell(in_dims => out_dims, [activation];
        use_bias=False(), init_bias=zeros32,
        init_reservoir=rand_sparse, init_input=scaled_rand,
        init_state=randn32, init_orthogonal=orthogonal,
        proximity=1.0))

Edge of Stability Echo State Network (ES2N) cell [Ceni2025](@cite).

## Equations

```math
\begin{aligned}
    \mathbf{x}(t) &= (1-\beta)\, \mathbf{O}\, \mathbf{x}(t-1) +
        \beta\, \phi\!\left(\mathbf{W}_{\text{in}} \mathbf{u}(t) +
        \mathbf{W}_r \mathbf{x}(t-1) + \mathbf{b} \right)
\end{aligned}
```
## Arguments

  - `in_dims`: Input dimension.
  - `out_dims`: Reservoir (hidden state) dimension.
  - `activation`: Activation function. Default: `tanh`.

## Keyword arguments

  - `use_bias`: Whether to include a bias term. Default: `false`.
  - `init_bias`: Initializer for the bias. Used only if `use_bias=true`.
    Default is `rand32`.
  - `init_reservoir`: Initializer for the reservoir matrix `W_res`.
    Default is [`rand_sparse`](@ref).
  - `init_orthogonal`: Initializer for the orthogonal matrix `O`.
    Default is [`orthogonal`](@ref).
  - `init_input`: Initializer for the input matrix `W_in`.
    Default is [`scaled_rand`](@ref).
  - `init_state`: Initializer for the hidden state when an external
    state is not provided. Default is `randn32`.
  - `proximity`: Proximity coefficient `α ∈ (0,1]`. Default: `1.0`.

## Inputs

  - **Case 1:** `x :: AbstractArray (in_dims, batch)`
    A fresh state is created via `init_state`; the call is forwarded to Case 2.
  - **Case 2:** `(x, (h,))` where `h :: AbstractArray (out_dims, batch)`
    Computes the update and returns the new state.

In both cases, the forward returns `((h_new, (h_new,)), st_out)` where `st_out`
contains any updated internal state.

## Returns

  - Output/hidden state `h_new :: out_dims` and state tuple `(h_new,)`.
  - Updated layer state (NamedTuple).

## Parameters

  - `input_matrix :: (out_dims × in_dims)` — `W_in`
  - `reservoir_matrix :: (out_dims × out_dims)` — `W_res`
  - `orthogonal_matrix :: (res_dims × res_dims)` — `O`
  - `bias :: (out_dims,)` — present only if `use_bias=true`

## States

Created by `initialstates(rng, esn)`:

  - `rng`: a replicated RNG used to sample initial hidden states when needed.
"""
@concrete struct ES2NCell <: AbstractEchoStateNetworkCell
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_reservoir
    init_input
    init_orthogonal
    init_state
    proximity
    use_bias <: StaticBool
end

function ES2NCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType},
        activation = tanh; use_bias::BoolType = False(), init_bias = zeros32,
        init_reservoir = rand_sparse, init_input = scaled_rand,
        init_state = randn32, init_orthogonal = orthogonal,
        proximity::AbstractFloat = 1.0)
    return ES2NCell(activation, in_dims, out_dims, init_bias, init_reservoir,
        init_input, init_orthogonal, init_state, proximity, static(use_bias))
end

function initialparameters(rng::AbstractRNG, esn::ES2NCell)
    ps = (input_matrix = esn.init_input(rng, esn.out_dims, esn.in_dims),
        reservoir_matrix = esn.init_reservoir(rng, esn.out_dims, esn.out_dims),
        orthogonal_matrix = esn.init_orthogonal(rng, esn.out_dims, esn.out_dims))
    if has_bias(esn)
        ps = merge(ps, (bias = esn.init_bias(rng, esn.out_dims),))
    end
    return ps
end

function (esn::ES2NCell)((inp, (hidden_state,))::InputType, ps, st::NamedTuple)
    T = eltype(inp)
    bias = safe_getproperty(ps, Val(:bias))
    t_prox = T(esn.proximity)
    win_inp = dense_bias(ps.input_matrix, inp, nothing)
    w_state = dense_bias(ps.reservoir_matrix, hidden_state, bias)
    candidate_h = esn.activation.(win_inp .+ w_state)
    h_new = (one(T) - t_prox) .* ps.orthogonal_matrix * hidden_state .+
            t_prox .* candidate_h
    return (h_new, (h_new,)), st
end

function Base.show(io::IO, esn::ES2NCell)
    print(io, "ES2NCell($(esn.in_dims) => $(esn.out_dims)")
    if esn.proximity != eltype(esn.proximity)(1.0)
        print(io, ", proximity=$(esn.proximity)")
    end
    has_bias(esn) || print(io, ", use_bias=false")
    print(io, ")")
end
