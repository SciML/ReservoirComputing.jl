abstract type AbstractEchoStateNetworkCell <: AbstractReservoirRecurrentCell end

@doc raw"""
    ESNCell(in_dims => out_dims, [activation];
        use_bias=false, init_bias=rand32,
        init_reservoir=rand_sparse, init_input=scaled_rand,
        init_state=randn32, leak_coefficient=1.0)

Echo State Network (ESN) recurrent cell with optional leaky integration.

## Equations

```math
\begin{aligned}
    \mathbf{x}(t) &= (1-\alpha)\, \mathbf{x}(t-1)
        + \alpha\, \phi\!\left(\mathbf{W}_{\text{in}}\, \mathbf{u}(t)
        + \mathbf{W}_r\, \mathbf{x}(t-1) + \mathbf{b} \right)
\end{aligned}
```
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
  - `leak_coefficient`: Leak rate `α ∈ (0,1]`. Default: `1.0`.

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

Created by `initialparameters(rng, esn)`:

  - `input_matrix :: (out_dims × in_dims)` — `W_in`
  - `reservoir_matrix :: (out_dims × out_dims)` — `W_res`
  - `bias :: (out_dims,)` — present only if `use_bias=true`

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
    #init_feedback::F
    init_state
    leak_coefficient
    use_bias <: StaticBool
end

function ESNCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType},
        activation = tanh_fast; use_bias::BoolType = False(), init_bias = zeros32,
        init_reservoir = rand_sparse, init_input = scaled_rand,
        init_state = randn32, leak_coefficient::AbstractFloat = 1.0
    )
    return ESNCell(
        activation, in_dims, out_dims, init_bias, init_reservoir,
        init_input, init_state, leak_coefficient, static(use_bias)
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
    return ps
end

function initialstates(rng::AbstractRNG, esn::AbstractEchoStateNetworkCell)
    return (rng = sample_replicate(rng),)
end

function (esn::AbstractEchoStateNetworkCell)(inp::AbstractArray, ps, st::NamedTuple)
    rng = replicate(st.rng)
    hidden_state = init_hidden_state(rng, esn, inp)
    return esn((inp, (hidden_state,)), ps, merge(st, (; rng)))
end

function (esn::ESNCell)((inp, (hidden_state,))::InputType, ps, st::NamedTuple)
    T = eltype(inp)
    bias = safe_getproperty(ps, Val(:bias))
    t_lc = T(esn.leak_coefficient)
    win_inp = dense_bias(ps.input_matrix, inp, nothing)
    w_state = dense_bias(ps.reservoir_matrix, hidden_state, bias)
    candidate_h = esn.activation.(win_inp .+ w_state)
    h_new = (one(T) - t_lc) .* hidden_state .+ t_lc .* candidate_h
    return (h_new, (h_new,)), st
end

function Base.show(io::IO, esn::ESNCell)
    print(io, "ESNCell($(esn.in_dims) => $(esn.out_dims)")
    if esn.leak_coefficient != eltype(esn.leak_coefficient)(1.0)
        print(io, ", leak_coefficient=$(esn.leak_coefficient)")
    end
    has_bias(esn) || print(io, ", use_bias=false")
    return print(io, ")")
end
