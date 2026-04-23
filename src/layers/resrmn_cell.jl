@doc raw"""
    ResRMNCell((in_dims, mem_dims) => out_dims, [activation];
        use_bias=False(), init_bias=zeros32,
        init_reservoir=rand_sparse, init_input=scaled_rand,
        init_memory=scaled_rand, init_state=randn32,
        init_orthogonal=orthogonal,
        alpha=1.0, beta=1.0)

Nonlinear residual reservoir cell used in Residual Reservoir Memory Networks
[Ceni2025b](@cite). Extends [`ResESNCell`](@ref) with a second linear map from
a memory reservoir output `m(t)` (the linear memory reservoir is an
[`ESNCell`](@ref) with `identity` activation in the [`ResRMN`](@ref) model).

## Equations

```math
\begin{aligned}
    \mathbf{h}(t) &= \alpha\, \mathbf{O}\, \mathbf{h}(t - 1) +
        \beta\, \phi\!\left(\mathbf{W}_{\text{in}}\, \mathbf{u}(t) +
        \mathbf{W}_{\text{mem}}\, \mathbf{m}(t) +
        \mathbf{W}_r\, \mathbf{h}(t - 1) + \mathbf{b}\right)
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
  - `init_memory`: Initializer for the memory-to-reservoir matrix `W_mem`.
    Default is [`scaled_rand`](@ref).
  - `init_orthogonal`: Initializer for the orthogonal matrix `O`.
    Default is `orthogonal`.
  - `init_state`: Initializer for the hidden state when an external state is
    not provided. Default is `randn32`.
  - `alpha`: Residual skip weight `α`. Default: `1.0`.
  - `beta`: Nonlinear transform weight `β`. Default: `1.0`.

## Inputs

  - **Case 1:** `(u, m)` where `u :: (in_dims, batch)` and
    `m :: (mem_dims, batch)`. A fresh state is created via `init_state`; the
    call is forwarded to Case 2.
  - **Case 2:** `((u, m), (h,))` with `h :: (out_dims, batch)`.

## Returns

  - Output/hidden state `h_new :: (out_dims, batch)` and state tuple `(h_new,)`.
  - Updated layer state (NamedTuple).

## Parameters

  - `input_matrix :: (out_dims × in_dims)` — `W_in`
  - `memory_matrix :: (out_dims × mem_dims)` — `W_mem`
  - `reservoir_matrix :: (out_dims × out_dims)` — `W_r`
  - `orthogonal_matrix :: (out_dims × out_dims)` — `O`
  - `bias :: (out_dims,)` — present only if `use_bias=true`

## States

  - `rng`: a replicated RNG used to sample initial hidden states when needed.
"""
@concrete struct ResRMNCell <: AbstractEchoStateNetworkCell
    activation
    in_dims <: IntegerType
    mem_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_reservoir
    init_input
    init_memory
    init_orthogonal
    init_state
    alpha
    beta
    use_bias <: StaticBool
end

function ResRMNCell(
        ((in_dims, mem_dims), out_dims)::Pair{<:Tuple{<:IntegerType, <:IntegerType}, <:IntegerType},
        activation = tanh_fast; use_bias::BoolType = False(), init_bias = zeros32,
        init_reservoir = rand_sparse, init_input = scaled_rand,
        init_memory = scaled_rand, init_state = randn32,
        init_orthogonal = orthogonal,
        alpha::AbstractFloat = 1.0, beta::AbstractFloat = 1.0
    )
    return ResRMNCell(
        activation, in_dims, mem_dims, out_dims, init_bias, init_reservoir,
        init_input, init_memory, init_orthogonal, init_state, alpha, beta,
        static(use_bias)
    )
end

function initialparameters(rng::AbstractRNG, rc::ResRMNCell)
    ps = (
        input_matrix = rc.init_input(rng, rc.out_dims, rc.in_dims),
        memory_matrix = rc.init_memory(rng, rc.out_dims, rc.mem_dims),
        reservoir_matrix = rc.init_reservoir(rng, rc.out_dims, rc.out_dims),
        orthogonal_matrix = rc.init_orthogonal(rng, rc.out_dims, rc.out_dims),
    )
    if has_bias(rc)
        ps = merge(ps, (bias = rc.init_bias(rng, rc.out_dims),))
    end
    return ps
end

function (rc::ResRMNCell)(
        inp::Tuple{<:AbstractArray, <:AbstractArray}, ps, st::NamedTuple
    )
    rng = replicate(st.rng)
    hidden_state = init_hidden_state(rng, rc, first(inp))
    return rc((inp, (hidden_state,)), ps, merge(st, (; rng)))
end

function (rc::ResRMNCell)(
        ((u, m), (hidden_state,))::Tuple{
            <:Tuple{<:AbstractArray, <:AbstractArray},
            <:Tuple{<:AbstractArray},
        },
        ps, st::NamedTuple
    )
    T = eltype(u)
    bias = safe_getproperty(ps, Val(:bias))
    t_alpha = T(rc.alpha)
    t_beta = T(rc.beta)
    win_inp = dense_bias(ps.input_matrix, u, nothing)
    wmem_inp = dense_bias(ps.memory_matrix, m, nothing)
    w_state = dense_bias(ps.reservoir_matrix, hidden_state, bias)
    candidate_h = rc.activation.(win_inp .+ wmem_inp .+ w_state)
    # Parenthesise the matmul so we scale the O(n) result vector,
    # not the O(n²) matrix — avoids a full-matrix temporary.
    h_new = t_alpha .* (ps.orthogonal_matrix * hidden_state) .+
        t_beta .* candidate_h
    return (h_new, (h_new,)), st
end

function Base.show(io::IO, rc::ResRMNCell)
    print(io, "ResRMNCell(($(rc.in_dims), $(rc.mem_dims)) => $(rc.out_dims)")
    if rc.alpha != eltype(rc.alpha)(1.0)
        print(io, ", alpha=$(rc.alpha)")
    end
    if rc.beta != eltype(rc.beta)(1.0)
        print(io, ", beta=$(rc.beta)")
    end
    has_bias(rc) || print(io, ", use_bias=false")
    return print(io, ")")
end
