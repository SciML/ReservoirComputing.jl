@doc raw"""
    ParameterAwareESNCell(in_dims => out_dims, [activation];
        use_bias=false, init_bias=zeros32,
        init_reservoir=rand_sparse, init_input=scaled_rand,
        init_parameter=scaled_rand, init_state=randn32,
        leak_coefficient=1.0, parameter_coupling=1.0, parameter_offset=0.0)

Parameter-Aware Echo State Network (PA-ESN) recurrent cell.

This cell extends the standard ESN by incorporating a control/bifurcation parameter
into the reservoir dynamics, enabling the network to learn dynamics across different
parameter regimes. Based on the parameter-aware reservoir computing framework
described in the literature.

## Equations

```math
\begin{aligned}
    \tilde{\mathbf{h}}(t) &= \phi\!\left(\mathbf{W}_{in}\,\mathbf{x}(t) +
        \mathbf{W}_{res}\,\mathbf{h}(t-1) + k_b\,\mathbf{W}_b\,(\varepsilon - \varepsilon_b)
        + \mathbf{b}\right) \\
    \mathbf{h}(t) &= (1-\alpha)\,\mathbf{h}(t-1) + \alpha\,\tilde{\mathbf{h}}(t)
\end{aligned}
```

where ``\varepsilon`` is the control parameter, ``k_b`` is the parameter coupling strength,
``\mathbf{W}_b`` is the parameter weight matrix, and ``\varepsilon_b`` is the parameter offset.

## Arguments

  - `in_dims`: Input dimension (not including the control parameter).
  - `out_dims`: Reservoir (hidden state) dimension.
  - `activation`: Activation function. Default: `tanh`.

## Keyword arguments

  - `use_bias`: Whether to include a bias term. Default: `false`.
  - `init_bias`: Initializer for the bias. Used only if `use_bias=true`.
      Default is `zeros32`.
  - `init_reservoir`: Initializer for the reservoir matrix `W_res`.
    Default is [`rand_sparse`](@ref).
  - `init_input`: Initializer for the input matrix `W_in`.
    Default is [`scaled_rand`](@ref).
  - `init_parameter`: Initializer for the parameter weight matrix `W_b`.
    Default is [`scaled_rand`](@ref).
  - `init_state`: Initializer for the hidden state when an external
    state is not provided. Default is `randn32`.
  - `leak_coefficient`: Leak rate `α ∈ (0,1]`. Default: `1.0`.
  - `parameter_coupling`: Parameter coupling strength `k_b`. Default: `1.0`.
  - `parameter_offset`: Parameter offset `ε_b`. Default: `0.0`.

## Inputs

  - **Case 1:** `(x, param) :: (AbstractArray (in_dims, batch), AbstractArray (1, batch))`
    A fresh state is created via `init_state`; the call is forwarded to Case 2.
  - **Case 2:** `((x, param), (h,))` where `x :: AbstractArray (in_dims, batch)`,
    `param :: AbstractArray (1, batch)`, and `h :: AbstractArray (out_dims, batch)`
    Computes the update and returns the new state.

In both cases, the forward returns `((h_new, (h_new,)), st_out)` where `st_out`
contains any updated internal state.

## Returns

  - Output/hidden state `h_new :: out_dims` and state tuple `(h_new,)`.
  - Updated layer state (NamedTuple).

## Parameters

Created by `initialparameters(rng, paesn)`:

  - `input_matrix :: (out_dims × in_dims)` — `W_in`
  - `reservoir_matrix :: (out_dims × out_dims)` — `W_res`
  - `parameter_matrix :: (out_dims × 1)` — `W_b`
  - `bias :: (out_dims,)` — present only if `use_bias=true`

## States

Created by `initialstates(rng, paesn)`:

  - `rng`: a replicated RNG used to sample initial hidden states when needed.

## References

  - Kong, L.-W., Weng, Y., Glaz, B., Haile, M., & Lai, Y.-C. (2021). Reservoir
    computing as digital twins for nonlinear dynamical systems. Chaos.
  - Patel, D., Canaday, D., Girvan, M., Pomerance, A., & Ott, E. (2021). Using
    machine learning to predict statistical properties of non-stationary
    dynamical processes. Physical Review E.
"""
@concrete struct ParameterAwareESNCell <: AbstractEchoStateNetworkCell
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_reservoir
    init_input
    init_parameter
    init_state
    leak_coefficient
    parameter_coupling
    parameter_offset
    use_bias <: StaticBool
end

function ParameterAwareESNCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType},
        activation = tanh; use_bias::BoolType = False(), init_bias = zeros32,
        init_reservoir = rand_sparse, init_input = scaled_rand,
        init_parameter = scaled_rand, init_state = randn32,
        leak_coefficient::AbstractFloat = 1.0,
        parameter_coupling::AbstractFloat = 1.0,
        parameter_offset::AbstractFloat = 0.0)
    return ParameterAwareESNCell(activation, in_dims, out_dims, init_bias, init_reservoir,
        init_input, init_parameter, init_state, leak_coefficient,
        parameter_coupling, parameter_offset, static(use_bias))
end

function initialparameters(rng::AbstractRNG, paesn::ParameterAwareESNCell)
    ps = (input_matrix = paesn.init_input(rng, paesn.out_dims, paesn.in_dims),
        reservoir_matrix = paesn.init_reservoir(rng, paesn.out_dims, paesn.out_dims),
        parameter_matrix = paesn.init_parameter(rng, paesn.out_dims, 1))
    if has_bias(paesn)
        ps = merge(ps, (bias = paesn.init_bias(rng, paesn.out_dims),))
    end
    return ps
end

# Input type for parameter-aware ESN: ((x, param), (hidden_state,))
const PAESNInputType = Tuple{Tuple{<:AbstractArray, <:AbstractVecOrMat}, Tuple{<:AbstractArray}}

# Case 1: Only (x, param) input - create initial hidden state
function (paesn::ParameterAwareESNCell)(inp::Tuple{<:AbstractArray, <:AbstractVecOrMat},
        ps, st::NamedTuple)
    x, param = inp
    rng = replicate(st.rng)
    hidden_state = init_hidden_state(rng, paesn, x)
    return paesn(((x, param), (hidden_state,)), ps, merge(st, (; rng)))
end

# Case 2: Full input with hidden state
function (paesn::ParameterAwareESNCell)(inp::PAESNInputType, ps, st::NamedTuple)
    (x, param), (hidden_state,) = inp
    T = eltype(x)
    bias = safe_getproperty(ps, Val(:bias))
    t_lc = T(paesn.leak_coefficient)
    t_pc = T(paesn.parameter_coupling)
    t_po = T(paesn.parameter_offset)

    # W_in * x
    win_inp = dense_bias(ps.input_matrix, x, nothing)
    # W_res * h
    w_state = dense_bias(ps.reservoir_matrix, hidden_state, bias)
    # k_b * W_b * (param - param_offset)
    param_shifted = param .- t_po
    w_param = t_pc .* dense_bias(ps.parameter_matrix, param_shifted, nothing)

    # Apply activation
    candidate_h = paesn.activation.(win_inp .+ w_state .+ w_param)
    # Leaky integration
    h_new = (one(T) - t_lc) .* hidden_state .+ t_lc .* candidate_h
    return (h_new, (h_new,)), st
end

function Base.show(io::IO, paesn::ParameterAwareESNCell)
    print(io, "ParameterAwareESNCell($(paesn.in_dims) => $(paesn.out_dims)")
    if paesn.leak_coefficient != eltype(paesn.leak_coefficient)(1.0)
        print(io, ", leak_coefficient=$(paesn.leak_coefficient)")
    end
    if paesn.parameter_coupling != eltype(paesn.parameter_coupling)(1.0)
        print(io, ", parameter_coupling=$(paesn.parameter_coupling)")
    end
    if paesn.parameter_offset != eltype(paesn.parameter_offset)(0.0)
        print(io, ", parameter_offset=$(paesn.parameter_offset)")
    end
    has_bias(paesn) || print(io, ", use_bias=false")
    print(io, ")")
end

@doc raw"""
    ParameterAwareStatefulLayer(cell::ParameterAwareESNCell)

A stateful wrapper for [`ParameterAwareESNCell`](@ref) that manages the hidden state
across timesteps.

This layer expects input as a tuple `(x, param)` where:
- `x`: The state input of shape `(in_dims, batch)`
- `param`: The control parameter of shape `(1, batch)` or scalar

## States

- `cell`: internal states for the wrapped cell (e.g., RNG replicas).
- `carry`: the per-sequence hidden state; initialized to `nothing`.
"""
@concrete struct ParameterAwareStatefulLayer <: AbstractLuxWrapperLayer{:cell}
    cell <: ParameterAwareESNCell
end

function initialstates(rng::AbstractRNG, sl::ParameterAwareStatefulLayer)
    return (cell = initialstates(rng, sl.cell), carry = nothing)
end

function (sl::ParameterAwareStatefulLayer)(inp::Tuple{<:AbstractArray, <:AbstractVecOrMat},
        ps, st::NamedTuple)
    x, param = inp
    if st.carry === nothing
        # No carry yet, let the cell handle initialization
        (out, carry), newst = apply(sl.cell, inp, ps, st.cell)
    else
        # Have existing carry
        (out, carry), newst = apply(sl.cell, (inp, st.carry), ps, st.cell)
    end
    return out, (; cell = newst, carry)
end

function Base.show(io::IO, sl::ParameterAwareStatefulLayer)
    print(io, "ParameterAwareStatefulLayer(")
    show(io, sl.cell)
    print(io, ")")
end
