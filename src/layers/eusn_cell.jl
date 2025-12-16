@doc raw"""
    EuSNCell(in_dims => out_dims, [activation];
        use_bias=false, init_bias=rand32,
        init_reservoir=rand_sparse, init_input=scaled_rand,
        init_state=randn32, leak_coefficient=1.0, diffusion=1.0)

Euler State Network (EuSN) cell.

## Equations

```math
\begin{aligned}
    \mathbf{h}(t) = \mathbf{h}(t-1) + \varepsilon \tanh\!\left((\mathbf{W}_h
        - \gamma \mathbf{I})\mathbf{h}(t-1) + \mathbf{W}_x \mathbf{x}(t)
        + \mathbf{b}\right)
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
  - `init_input`: Initializer for the input matrix `W_in`.
    Default is [`scaled_rand`](@ref).
  - `init_state`: Initializer for the hidden state when an external
    state is not provided. Default is `randn32`.
  - `leak_coefficient`: Leak rate `α ∈ (0,1]`. Default: `1.0`.
  - `diffusion`: Diffusiona parameter `∈ (0,1]`. Default: `1.0`.

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
@concrete struct EuSNCell <: AbstractEchoStateNetworkCell
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_reservoir
    init_input
    #init_feedback::F
    init_state
    leak_coefficient
    diffusion
    use_bias <: StaticBool
end

function EuSNCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType},
        activation = tanh; use_bias::BoolType = False(), init_bias = zeros32,
        init_reservoir = rand_sparse, init_input = scaled_rand,
        init_state = randn32, leak_coefficient::AbstractFloat = 1.0,
        diffusion::AbstractFloat = 1.0)
    return ESNCell(activation, in_dims, out_dims, init_bias, init_reservoir,
        init_input, init_state, leak_coefficient, diffusion, static(use_bias))
end

function (esn::EuSNCell)((inp, (hidden_state,))::InputType, ps, st::NamedTuple)
    T = eltype(inp)
    bias = safe_getproperty(ps, Val(:bias))
    t_lc = T(esn.leak_coefficient)
    t_diff = T(esn.diffusion)
    asynm_matrix = compute_asym_recurrent(ps.reservoir_matrix, t_diff)
    win_inp = dense_bias(ps.input_matrix, inp, nothing)
    w_state = dense_bias(asynm_matrix, hidden_state, bias)
    candidate_h = esn.activation.(win_inp .+ w_state)
    h_new = (one(T) - t_lc) .* hidden_state .+ t_lc .* candidate_h
    return (h_new, (h_new,)), st
end

function Base.show(io::IO, esn::EuSNCell)
    print(io, "EuSNCell($(esn.in_dims) => $(esn.out_dims)")
    if esn.leak_coefficient != eltype(esn.leak_coefficient)(1.0)
        print(io, ", leak_coefficient=$(esn.leak_coefficient)")
    end
    if esn.diffusion != eltype(esn.diffusion)(1.0)
        print(io, ", diffusion=$(esn.diffusion)")
    end
    has_bias(esn) || print(io, ", use_bias=false")
    print(io, ")")
end

function compute_asym_recurrent(weight_hh::AbstractMatrix, gamma::AbstractFloat)
    return weight_hh .- transpose(weight_hh) .-
           gamma .* Matrix{eltype(weight_hh)}(I, size(weight_hh, 1), size(weight_hh, 1))
end
