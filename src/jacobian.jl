@doc raw"""
    jacobian(esn::ESN, state, ps, st; backend=:analytical)
    jacobian!(J, esn::ESN, state, ps, st; backend=:analytical)

Jacobian of the closed-loop reservoir map at reservoir state `state`
[Pathak2017](@cite).

Requires `in_dims == out_dims`. With readout feedback the reservoir obeys

```math
\begin{aligned}
\mathbf{z} &= \mathrm{Mods}(\mathbf{x}), \\
\mathbf{u} &= \rho(\mathbf{W}_{\mathrm{out}}\mathbf{z}+\mathbf{b}_{\mathrm{out}}), \\
\mathbf{a} &= \mathbf{W}_{\mathrm{in}}\mathbf{u}+\mathbf{W}_r\mathbf{x}+\mathbf{b}, \\
F(\mathbf{x}) &= (1-\alpha)\odot\mathbf{x}+\alpha\odot\varphi(\mathbf{a}),
\end{aligned}
```

and this returns ``J = DF/D\mathbf{x}``. The carry is not advanced.

## Arguments

  - `esn`: an [`ESN`](@ref).
  - `state`: reservoir carry, length `res_dims` (vector or single-column matrix).
  - `ps`: model parameters.
  - `st`: model states.
  - `J`: preallocated `res_dims × res_dims` buffer for `jacobian!`.

## Keyword arguments

  - `backend`: `:analytical` (default) or `:forwarddiff`.

## Returns

  - `(J, st)` with `J` of size `(res_dims, res_dims)`. `st` is unchanged.
"""
function jacobian(
        esn::ESN, state::AbstractVecOrMat, ps, st;
        backend::Symbol = :analytical
    )
    x = __jacobian_state_vector(state)
    J = Matrix{eltype(x)}(undef, length(x), length(x))
    return jacobian!(J, esn, x, ps, st; backend)
end

function jacobian!(
        J::AbstractMatrix, esn::ESN, state::AbstractVecOrMat, ps, st;
        backend::Symbol = :analytical
    )
    x = __jacobian_state_vector(state)
    __check_closedloop_jacobian_dims(esn, x, J)
    if backend === :analytical
        __analytical_closedloop_jacobian!(J, esn, x, ps)
    elseif backend === :forwarddiff
        __forwarddiff_closedloop_jacobian!(J, esn, x, ps)
    else
        throw(ArgumentError("backend must be :analytical or :forwarddiff, got $(repr(backend))"))
    end
    return J, st
end

@doc raw"""
    jacobians(esn::ESN, steps, ps, st; initialdata, backend=:analytical)

Jacobians of the closed-loop reservoir map along an autoregressive trajectory.

Uses the same feedback as [`predict`](@ref): the first step is driven by
`initialdata`, then each output is fed back as the next input. After step
``t``, `Js[:, :, t]` stores ``DF(\mathbf{x}_t)`` at the current carry.

## Arguments

  - `esn`: an [`ESN`](@ref) with `in_dims == out_dims`.
  - `steps`: number of autoregressive steps.
  - `ps`: model parameters.
  - `st`: model states.

## Keyword arguments

  - `initialdata`: column vector used as the first input.
  - `backend`: `:analytical` (default) or `:forwarddiff`.

## Returns

  - `Js`: array of size `(res_dims, res_dims, steps)`.
  - `outputs`: generated outputs of shape `(out_dims, steps)`.
  - `st`: model state after `steps` updates.
"""
function jacobians(
        esn::ESN, steps::Integer, ps, st;
        initialdata::AbstractVector, backend::Symbol = :analytical
    )
    steps ≥ 1 || throw(ArgumentError("steps must be ≥ 1, got $steps"))
    input_length = length(initialdata)
    __require_esn_closedloop_io(esn)

    current_output, st = apply(esn, initialdata, ps, st)
    __require_closed_loop_dimension(current_output, input_length, 1)

    x = __carry_state_vector(st)
    n = length(x)
    Js = Array{eltype(x)}(undef, n, n, steps)
    outputs = similar(current_output, length(current_output), steps)

    jacobian!(view(Js, :, :, 1), esn, x, ps, st; backend)
    outputs[:, 1] .= current_output

    for step in 2:steps
        current_output, st = apply(esn, current_output, ps, st)
        __require_closed_loop_dimension(current_output, input_length, step)
        x = __carry_state_vector(st)
        jacobian!(view(Js, :, :, step), esn, x, ps, st; backend)
        outputs[:, step] .= current_output
    end
    return Js, outputs, st
end

function __jacobian_state_vector(state::AbstractVector)
    return state
end

function __jacobian_state_vector(state::AbstractMatrix)
    size(state, 2) == 1 || throw(
        ArgumentError(
            "jacobian expects a vector or single-column matrix, got size $(size(state))"
        )
    )
    return vec(state)
end

function __carry_state_vector(st::NamedTuple)
    carry = get(st.reservoir, :carry, nothing)
    carry === nothing && throw(
        ArgumentError("reservoir carry is unset; run a model step or pass `state` explicitly")
    )
    return __jacobian_state_vector(first(carry))
end

function __require_esn_closedloop_io(esn::ESN)
    in_dims = Int(esn.reservoir.cell.in_dims)
    out_dims = Int(esn.readout.out_dims)
    in_dims == out_dims || throw(
        DimensionMismatch(
            "closed-loop jacobian requires in_dims == out_dims (got $in_dims and $out_dims)"
        )
    )
    return nothing
end

function __check_closedloop_jacobian_dims(esn::ESN, x::AbstractVector, J::AbstractMatrix)
    __require_esn_closedloop_io(esn)
    n = Int(esn.reservoir.cell.out_dims)
    length(x) == n || throw(DimensionMismatch("reservoir state length must be $n, got $(length(x))"))
    size(J) == (n, n) || throw(DimensionMismatch("Jacobian buffer must be ($n, $n), got $(size(J))"))
    return nothing
end

function __closed_loop_quantities(esn::ESN, x::AbstractVector, ps)
    cell = esn.reservoir.cell
    z = __apply_modifiers_pure(esn.state_modifiers, x, ps.state_modifiers)
    u = __readout_pure(esn.readout, z, ps.readout)

    input_matrix = ps.reservoir.input_matrix
    reservoir_matrix = ps.reservoir.reservoir_matrix
    bias = safe_getproperty(ps.reservoir, Val(:bias))
    preactivation = dense_bias(input_matrix, u, nothing) .+
        dense_bias(reservoir_matrix, x, bias)

    T = eltype(x)
    leak = __format_leak(T, cell.leak_coefficient)
    x_new = __one_minus_leak(T, leak) .* x .+ leak .* cell.activation.(preactivation)
    return (; x_new, u, preactivation, z, leak, input_matrix, reservoir_matrix)
end

function __closed_loop_step(esn::ESN, x::AbstractVector, ps)
    q = __closed_loop_quantities(esn, x, ps)
    return q.x_new, q.u
end

__apply_modifiers_pure(::Tuple{}, x, ::Tuple{}) = x

function __apply_modifiers_pure(modifiers::Tuple, x, ps_mods::Tuple)
    features = x
    for (modifier, ps_mod) in zip(modifiers, ps_mods)
        features, _ = __apply_state_modifier(modifier, features, x, ps_mod, NamedTuple())
    end
    return features
end

function __readout_pure(readout::LinearReadout, features, ps_readout)
    out = ps_readout.weight * features
    if has_bias(readout)
        out = out .+ ps_readout.bias
    end
    return readout.activation.(out)
end

function __analytical_closedloop_jacobian!(J::AbstractMatrix, esn::ESN, x::AbstractVector, ps)
    __ensure_supported_modifiers(esn.state_modifiers)
    q = __closed_loop_quantities(esn, x, ps)
    M = __state_modifiers_jacobian(esn.state_modifiers, x)
    Du_Dx = __readout_jacobian(esn.readout, q.z, ps.readout, M)

    J .= q.input_matrix * Du_Dx
    J .+= q.reservoir_matrix # densifies if `reservoir_matrix` is sparse

    dφ = __activation_derivative(esn.reservoir.cell.activation, q.preactivation)
    __scale_rows!(J, q.leak, dφ)
    __add_leak_identity!(J, q.leak)
    return J
end

function __scale_rows!(J::AbstractMatrix, leak::Number, dφ::AbstractVector)
    @inbounds for i in axes(J, 1)
        scale = leak * dφ[i]
        for j in axes(J, 2)
            J[i, j] *= scale
        end
    end
    return J
end

function __scale_rows!(J::AbstractMatrix, leak::AbstractArray, dφ::AbstractVector)
    leak_vec = vec(leak)
    length(leak_vec) == length(dφ) || throw(
        DimensionMismatch(
            "leak_coefficient length $(length(leak_vec)) must match reservoir size $(length(dφ))"
        )
    )
    @inbounds for i in axes(J, 1)
        scale = leak_vec[i] * dφ[i]
        for j in axes(J, 2)
            J[i, j] *= scale
        end
    end
    return J
end

function __add_leak_identity!(J::AbstractMatrix, leak::Number)
    one_minus = one(eltype(J)) - leak
    @inbounds for i in axes(J, 1)
        J[i, i] += one_minus
    end
    return J
end

function __add_leak_identity!(J::AbstractMatrix, leak::AbstractArray)
    leak_vec = vec(leak)
    @inbounds for i in axes(J, 1)
        J[i, i] += one(eltype(J)) - leak_vec[i]
    end
    return J
end

function __readout_jacobian(readout::LinearReadout, z, ps_readout, M::AbstractMatrix)
    weight_M = ps_readout.weight * M
    readout.activation === identity && return weight_M

    pre = ps_readout.weight * z
    if has_bias(readout)
        pre = pre .+ ps_readout.bias
    end
    return __activation_derivative(readout.activation, pre) .* weight_M
end

__activation_derivative(::typeof(identity), a::AbstractVector) = ones(eltype(a), length(a))

function __activation_derivative(::typeof(tanh), a::AbstractVector)
    y = tanh.(a)
    return one(eltype(a)) .- y .* y
end

function __activation_derivative(::typeof(tanh_fast), a::AbstractVector)
    y = tanh_fast.(a)
    return one(eltype(a)) .- y .* y
end

function __activation_derivative(activation, ::AbstractVector)
    throw(
        ArgumentError(
            "no analytical derivative for activation $(activation); " *
                "use `backend=:forwarddiff` after `using ForwardDiff`"
        )
    )
end

__unwrap_modifier(wf::WrappedFunction) = wf.func
__unwrap_modifier(modifier) = modifier

function __ensure_supported_modifiers(modifiers::Tuple)
    for modifier in modifiers
        unwrapped = __unwrap_modifier(modifier)
        if unwrapped isa Extend
            throw(
                ArgumentError(
                    "closed-loop jacobian does not support `Extend` " *
                        "(autonomous state is not the reservoir carry alone)"
                )
            )
        end
        if !hasmethod(__state_modifier_jacobian, Tuple{typeof(unwrapped), AbstractVector})
            throw(
                ArgumentError(
                    "no analytical Jacobian for state modifier $(unwrapped); " *
                        "use `backend=:forwarddiff` after `using ForwardDiff`"
                )
            )
        end
    end
    return nothing
end

function __state_modifiers_jacobian(::Tuple{}, x::AbstractVector)
    n = length(x)
    return Matrix{eltype(x)}(I, n, n)
end

function __state_modifiers_jacobian(modifiers::Tuple, x::AbstractVector)
    features = x
    unwrapped = __unwrap_modifier(first(modifiers))
    M = __state_modifier_jacobian(unwrapped, features)
    features = unwrapped(features)
    for modifier in Base.tail(modifiers)
        unwrapped = __unwrap_modifier(modifier)
        M = __state_modifier_jacobian(unwrapped, features) * M
        features = unwrapped(features)
    end
    return M
end

function __state_modifier_jacobian(::typeof(NLAT1), x::AbstractVector)
    T = eltype(x)
    n = length(x)
    M = Matrix{T}(I, n, n)
    @inbounds for i in eachindex(x)
        if isodd(i)
            M[i, i] = 2 * x[i]
        end
    end
    return M
end

function __state_modifier_jacobian(::typeof(NLAT2), x::AbstractVector)
    T = eltype(x)
    n = length(x)
    M = Matrix{T}(I, n, n)
    first_i = firstindex(x)
    @inbounds for i in eachindex(x)
        if i > first_i && isodd(i)
            M[i, i] = zero(T)
            M[i, i - 1] = x[i - 2]
            M[i, i - 2] = x[i - 1]
        end
    end
    return M
end

function __state_modifier_jacobian(::typeof(NLAT3), x::AbstractVector)
    T = eltype(x)
    n = length(x)
    M = Matrix{T}(I, n, n)
    first_i = firstindex(x)
    last_i = lastindex(x)
    @inbounds for i in eachindex(x)
        if first_i < i < last_i && isodd(i)
            M[i, i] = zero(T)
            M[i, i - 1] = x[i + 1]
            M[i, i + 1] = x[i - 1]
        end
    end
    return M
end

function __state_modifier_jacobian(::Pad, x::AbstractVector)
    T = eltype(x)
    n = length(x)
    M = zeros(T, n + 1, n)
    @inbounds for i in 1:n
        M[i, i] = one(T)
    end
    return M
end

function __state_modifier_jacobian(partial_square::PartialSquare, x::AbstractVector)
    T = eltype(x)
    n = length(x)
    M = Matrix{T}(I, n, n)
    threshold = floor(Int, partial_square.eta * n)
    @inbounds for i in 1:threshold
        M[i, i] = 2 * x[i]
    end
    return M
end

function __state_modifier_jacobian(::typeof(ExtendedSquare), x::AbstractVector)
    T = eltype(x)
    n = length(x)
    M = zeros(T, 2n, n)
    @inbounds for i in 1:n
        M[i, i] = one(T)
        M[n + i, i] = 2 * x[i]
    end
    return M
end

function __forwarddiff_closedloop_jacobian!(
        J::AbstractMatrix, esn::ESN, x::AbstractVector, ps
    )
    ext = Base.get_extension(@__MODULE__, :RCForwardDiffExt)
    ext === nothing && error(
        "backend=:forwarddiff requires ForwardDiff (`using ForwardDiff`)"
    )
    return ext.forwarddiff_closedloop_jacobian!(J, esn, x, ps)
end
