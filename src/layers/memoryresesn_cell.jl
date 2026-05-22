@doc raw"""
    MemoryResESNCell((in_dims, mem_dims) => out_dims, [activation];
        use_bias=False(), init_bias=zeros32,
        init_reservoir=rand_sparse, init_input=scaled_rand,
        init_memory=scaled_rand, init_state=randn32,
        init_orthogonal=orthogonal, alpha=1.0, beta=1.0)

Residual Echo State Network cell with a memory input matrix
[Ceni2025b](@cite). Extends [`ResESNCell`](@ref) with a linear map from a
memory reservoir output `m(t)`, in the same way that [`MemoryESNCell`](@ref)
extends [`ESNCell`](@ref). Intended to be plugged into an [`RMNCell`](@ref) as
the nonlinear reservoir, with an [`ESNCell`](@ref) (with `identity` activation)
playing the role of the linear memory reservoir.

## Equations

```math
\begin{aligned}
    \mathbf{x}(t) &= \alpha\, \mathbf{O}\, \mathbf{x}(t-1) +
        \beta\, \phi\!\left(\mathbf{W}_{\text{in}}\, \mathbf{u}(t) +
        \mathbf{W}_r\, \mathbf{x}(t-1) +
        \mathbf{W}_m\, \mathbf{m}(t-1) + \mathbf{b} \right)
\end{aligned}
```

## Arguments

  - `in_dims`: External input dimension.
  - `mem_dims`: Memory reservoir output dimension.
  - `out_dims`: Reservoir (hidden state) dimension.
  - `activation`: Activation function. Default: `tanh_fast`.

## Keyword arguments

  - `use_bias`: Whether to include a bias term. Default: `false`.
  - `init_bias`: Initializer for the bias. Used only if `use_bias=true`.
    Default is `zeros32`.
  - `init_reservoir`: Initializer for the reservoir matrix `W_r`.
    Default is [`rand_sparse`](@ref).
  - `init_input`: Initializer for the input matrix `W_in`.
    Default is [`scaled_rand`](@ref).
  - `init_memory`: Initializer for the memory matrix `W_m`.
    Default is [`scaled_rand`](@ref).
  - `init_orthogonal`: Initializer for the orthogonal matrix `O`.
    Default is `orthogonal`.
  - `init_state`: Initializer for the hidden state when an external
    state is not provided. Default is `randn32`.
  - `alpha`: Residual skip weight `α`. Default: `1.0`.
  - `beta`: Nonlinear transform weight `β`. Default: `1.0`.

## Inputs

  - **Case 1:** `x :: AbstractArray (in_dims, batch)`
    Fresh hidden and memory states are created via `init_state`; the call is
    forwarded to Case 2.
  - **Case 2:** `(x, (h, m))` where `h :: AbstractArray (out_dims, batch)` and
    `m :: AbstractArray (mem_dims, batch)`. Computes the update and returns
    the new hidden state.

## Returns

  - Output/hidden state `h_new :: out_dims` and state tuple `(h_new,)`.
  - Updated layer state (NamedTuple).

## Parameters

  - `input_matrix :: (out_dims × in_dims)` — `W_in`
  - `reservoir_matrix :: (out_dims × out_dims)` — `W_r`
  - `memory_matrix :: (out_dims × mem_dims)` — `W_m`
  - `orthogonal_matrix :: (out_dims × out_dims)` — `O`
  - `bias :: (out_dims,)` — present only if `use_bias=true`

## States

  - `rng`: a replicated RNG used to sample initial hidden states when needed.
"""
@concrete struct MemoryResESNCell <: AbstractEchoStateNetworkCell
    activation
    in_dims <: IntegerType
    mem_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_reservoir
    init_memory
    init_input
    init_orthogonal
    init_state
    alpha
    beta
    use_bias <: StaticBool
end

function MemoryResESNCell(
        ((in_dims, mem_dims), out_dims)::Pair{<:Tuple{<:IntegerType, <:IntegerType}, <:IntegerType},
        activation = tanh_fast; use_bias::BoolType = False(), init_bias = zeros32,
        init_reservoir = rand_sparse, init_input = scaled_rand,
        init_memory = scaled_rand, init_state = randn32,
        init_orthogonal = orthogonal,
        alpha::AbstractFloat = 1.0, beta::AbstractFloat = 1.0
    )
    return MemoryResESNCell(
        activation, in_dims, mem_dims, out_dims, init_bias, init_reservoir,
        init_memory, init_input, init_orthogonal, init_state, alpha, beta,
        static(use_bias)
    )
end

function initialparameters(rng::AbstractRNG, esn::MemoryResESNCell)
    ps = (
        input_matrix = esn.init_input(rng, esn.out_dims, esn.in_dims),
        reservoir_matrix = esn.init_reservoir(rng, esn.out_dims, esn.out_dims),
        memory_matrix = esn.init_memory(rng, esn.out_dims, esn.mem_dims),
        orthogonal_matrix = esn.init_orthogonal(rng, esn.out_dims, esn.out_dims),
    )
    if has_bias(esn)
        ps = merge(ps, (bias = esn.init_bias(rng, esn.out_dims),))
    end
    return ps
end

function (esn::MemoryResESNCell)(inp::AbstractArray, ps, st::NamedTuple)
    rng = replicate(st.rng)
    hidden_state = init_hidden_state(rng, esn, inp)
    memory_state = init_hidden_state(rng, esn, inp)
    return esn((inp, (hidden_state, memory_state)), ps, merge(st, (; rng)))
end

function (esn::MemoryResESNCell)(
        (inp, (hidden_state, memory_state))::Tuple{AbstractArray, Tuple{AbstractArray, AbstractArray}},
        ps, st::NamedTuple
    )
    T = eltype(inp)
    bias = safe_getproperty(ps, Val(:bias))
    t_alpha = T(esn.alpha)
    t_beta = T(esn.beta)
    win_inp = dense_bias(ps.input_matrix, inp, nothing)
    w_state = dense_bias(ps.reservoir_matrix, hidden_state, bias)
    w_memory = dense_bias(ps.memory_matrix, memory_state, nothing)
    candidate_h = esn.activation.(win_inp .+ w_state .+ w_memory)
    h_new = t_alpha .* (ps.orthogonal_matrix * hidden_state) .+
        t_beta .* candidate_h
    return (h_new, (h_new, w_memory)), st
end

function Base.show(io::IO, esn::MemoryResESNCell)
    print(io, "MemoryResESNCell($(esn.in_dims) => $(esn.out_dims)")
    if esn.alpha != eltype(esn.alpha)(1.0)
        print(io, ", alpha=$(esn.alpha)")
    end
    if esn.beta != eltype(esn.beta)(1.0)
        print(io, ", beta=$(esn.beta)")
    end
    has_bias(esn) || print(io, ", use_bias=false")
    return print(io, ")")
end
