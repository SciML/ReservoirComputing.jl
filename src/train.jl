@doc raw"""
    StandardRidge([Type], [reg])

Ridge regression for readout training.

With feature matrix ``\mathbf{X}`` of size `(n_features, T)` and target matrix
``\mathbf{Y}`` of size `(n_outputs, T)`, the fitted weights
``\mathbf{W}`` of size `(n_outputs, n_features)` satisfy

```math
\mathbf{W}^{\top}
=
(\mathbf{X}\mathbf{X}^{\top} + \lambda \mathbf{I})^{-1}
\mathbf{X}\mathbf{Y}^{\top}
```

so that ``\mathbf{Y} \approx \mathbf{W}\mathbf{X}``.

## Arguments

 - `Type`: type of the regularization coefficient. Default is inferred internally.
 - `reg`: regularization coefficient ``\lambda``. Default `0.0` (ordinary least squares).
"""
struct StandardRidge
    reg::Number
end

function StandardRidge(::Type{T}, reg) where {T <: Number}
    return StandardRidge(T.(reg))
end

function StandardRidge()
    return StandardRidge(0.0)
end

function _apply_washout(states::AbstractMatrix, targets::AbstractMatrix, washout::Integer)
    washout ≥ 0 || throw(ArgumentError("washout must be ≥ 0, got $washout"))
    n_samples = size(states, 2)
    washout < n_samples || throw(
        ArgumentError(
            "washout=$washout is ≥ number of time steps=$n_samples"
        )
    )
    states_wo = states[:, (washout + 1):end]
    targets_wo = targets[:, (washout + 1):end]
    return states_wo, targets_wo
end

_set_readout(ps, m::ReservoirChain, W) = first(addreadout!(m, W, ps, NamedTuple()))

abstract type AbstractReservoirComputingSolver end

@doc raw"""
    QRSolver()

Legacy built-in QR solver for [`StandardRidge`](@ref) training.

The package default is LinearSolve's `QRFactorization()`. Prefer that
unless you need this implementation explicitly.
"""
struct QRSolver <: AbstractReservoirComputingSolver end

_default_ridge_solver() = QRFactorization()
_resolve_ridge_solver(::Nothing) = _default_ridge_solver()
_resolve_ridge_solver(solver) = solver

function _fit_readout(objective, states, targets, ::Nothing; kwargs...)
    return train(objective, states, targets; kwargs...)
end

function _fit_readout(objective, states, targets, solver; kwargs...)
    return train(objective, states, targets; solver = solver, kwargs...)
end

@doc raw"""
    train(objective, states, target_data; solver=nothing, kwargs...)

Fit a readout from precomputed reservoir features and targets.

## Arguments

- `objective`: training objective (for example [`StandardRidge`](@ref), or a
  method from an extension such as MLJLinearModels / LIBSVM).
- `states`: feature matrix from [`collectstates`](@ref), shape `(n_features, T)`.
- `target_data`: targets aligned with `states`, shape `(n_outputs, T)`.

## Keyword arguments

- `solver`: for [`StandardRidge`](@ref), a LinearSolve algorithm such as
  `QRFactorization()` (default) or the legacy [`QRSolver`](@ref).
  Default `nothing` selects the package default for the objective.
- `kwargs...`: forwarded to the objective backend.

## Returns

Readout weights or backend-specific fit result (for ridge, a matrix usable by
[`LinearReadout`](@ref)).

Washout and state collection are handled by the model-level [`train`](@ref)
method, not here.
"""
function train(
        sr::StandardRidge, states::AbstractMatrix, target_data::AbstractMatrix;
        solver = nothing, kwargs...
    )
    ridge_solver = _resolve_ridge_solver(solver)
    return _train_ridge(ridge_solver, sr, states, target_data; kwargs...)
end

function _train_ridge(
        ::QRSolver, sr::StandardRidge,
        states::AbstractMatrix, target_data::AbstractMatrix; kwargs...
    )
    n_samples = size(states, 2)
    n_target_samples = size(target_data, 2)
    n_samples == n_target_samples || throw(
        DimensionMismatch(
            "states has $n_samples samples, targets has $n_target_samples"
        )
    )

    n_states = size(states, 1)
    A = [states'; sqrt(sr.reg) * I(n_states)]
    b = [target_data'; zeros(eltype(target_data), n_states, size(target_data, 1))]
    F = qr(A)
    Wt = F \ b
    output_layer = Matrix(Wt')
    return output_layer
end

function _train_ridge(
        solver::SciMLLinearSolveAlgorithm, sr::StandardRidge,
        states::AbstractMatrix, targets::AbstractMatrix; kwargs...
    )
    n_samples = size(states, 2)
    n_target_samples = size(targets, 2)
    n_samples == n_target_samples || throw(
        DimensionMismatch(
            "states has $n_samples samples, targets has $n_target_samples"
        )
    )

    λ = convert(eltype(states), sr.reg)
    gram = states * states' + λ * I
    rhs = states * targets'
    solution = solve(LinearProblem(gram, rhs), solver; kwargs...)
    return Matrix(solution.u')
end

function _train_ridge(
        solver, ::StandardRidge, ::AbstractMatrix, ::AbstractMatrix; kwargs...
    )
    return throw(
        ArgumentError(
            "Unsupported ridge solver of type $(typeof(solver)). " *
                "Use QRFactorization(), QRSolver(), or another LinearSolve.jl algorithm."
        )
    )
end

@doc raw"""
    train(rc, train_data, target_data, ps, st;
          objective=StandardRidge(0.0), solver=nothing,
          washout=0, return_states=false)

Train the readout of a reservoir computer.

Collects features from `train_data`, fits the readout to `target_data` using
`objective`, and returns updated parameters and states. Parameters are not
modified in place.

## Arguments

- `rc`: reservoir model (built-in model or [`ReservoirChain`](@ref)) with a
  trainable readout such as [`LinearReadout`](@ref).
- `train_data`: input sequence; columns are time steps.
- `target_data`: targets aligned with `train_data`.
- `ps`: model parameters.
- `st`: model states.

## Keyword arguments

- `objective`: training objective. Default [`StandardRidge`](@ref).
- `solver`: solver for the objective when applicable. For ridge, the default
  (`nothing`) is LinearSolve's `QRFactorization()`; pass [`QRSolver`](@ref)
  or another LinearSolve algorithm to override.
- `washout`: number of initial steps to discard from features and targets.
  Default `0`.
- `return_states`: if `true`, also return the feature matrix used for the fit.
- `kwargs...`: forwarded to the objective backend.

## Returns

- `(ps, st)` normally.
- `((ps, st), states)` if `return_states=true`.

If the readout uses implicit collection, create it with `include_collect=true`
or place an explicit [`Collect`](@ref) earlier in the chain.
"""
function train(
        rc, train_data, target_data, ps, st;
        objective = StandardRidge(0.0),
        solver = nothing,
        washout::Int = 0,
        return_states::Bool = false,
        kwargs...
    )
    raw_states, st_after = collectstates(rc, train_data, ps, st)
    states_wo,
        traindata_wo = washout > 0 ? _apply_washout(raw_states, target_data, washout) :
        (raw_states, target_data)
    output_matrix = _fit_readout(
        objective, states_wo, traindata_wo, solver; kwargs...
    )
    ps2, st_after = addreadout!(rc, output_matrix, ps, st_after)
    return return_states ? ((ps2, st_after), states_wo) : (ps2, st_after)
end

@doc raw"""
    train!(rc, train_data, target_data, ps, st,
           train_method=StandardRidge(0.0);
           washout=0, return_states=false, kwargs...)

Compatibility wrapper around model-level [`train`](@ref).

The positional `train_method` is passed as `objective`. Prefer
`train(rc, train_data, target_data, ps, st; objective=..., solver=...)` for
new code.
"""
function train!(
        rc, train_data, target_data, ps, st,
        train_method = StandardRidge(0.0);
        washout::Int = 0, return_states::Bool = false, kwargs...
    )
    return train(
        rc, train_data, target_data, ps, st;
        objective = train_method,
        washout = washout,
        return_states = return_states,
        kwargs...
    )
end

#_quote_keys(t) = Expr(:tuple, (QuoteNode(s) for s in t)...)

@generated function _setweight_rt(p::NamedTuple{K}, W) where {K}
    keys = K
    Kq = _quote_keys(keys)
    idx = findfirst(==(Symbol(:weight)), keys)

    terms = Any[]
    for i in 1:length(keys)
        push!(terms, (idx === i) ? :(W) : :(getfield(p, $i)))
    end

    if idx === nothing
        newK = _quote_keys((keys..., :weight))
        return :(NamedTuple{$newK}(($(terms...), W)))
    else
        return :(NamedTuple{$Kq}(($(terms...),)))
    end
end

@generated function _addreadout(layers::NamedTuple{K}, ps::NamedTuple{K}, W) where {K}
    if length(K) == 0
        return :(NamedTuple())
    end
    tailK = Base.tail(K)
    Kq = _quote_keys(K)
    tailKq = _quote_keys(tailK)

    head_val = :(
        (getfield(layers, 1) isa LinearReadout)
            ? _setweight_rt(getfield(ps, 1), W)
            : getfield(ps, 1)
    )

    tail_call = :(
        _addreadout(
            NamedTuple{$tailKq}(Base.tail(layers)),
            NamedTuple{$tailKq}(Base.tail(ps)),
            W
        )
    )

    return :(NamedTuple{$Kq}(($head_val, Base.values($tail_call)...)))
end

function addreadout!(
        rc::ReservoirChain,
        W::AbstractMatrix,
        ps::NamedTuple,
        st::NamedTuple
    )
    @assert propertynames(rc.layers) == propertynames(ps)
    new_ps = _addreadout(rc.layers, ps, W)
    return new_ps, st
end
