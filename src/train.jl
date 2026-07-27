@doc raw"""
    RidgeRegression([Type], [reg])

Ridge regression objective for readout training.

Fits weights ``\mathbf{W}`` so that ``\mathbf{Y} \approx \mathbf{W}\mathbf{X}``
with Tikhonov regularization ``\lambda``:

```math
\mathbf{W}^{\top}
=
(\mathbf{X}\mathbf{X}^{\top} + \lambda \mathbf{I})^{-1}
\mathbf{X}\mathbf{Y}^{\top}
```

## Arguments

  - `Type`: element type of ``\lambda`` (optional).
  - `reg`: regularization ``\lambda``. Default `0.0` (ordinary least squares).

Feature and target layouts are `(n_features, T)` and `(n_outputs, T)`; the
fitted weight matrix is `(n_outputs, n_features)`.
"""
struct RidgeRegression
    reg::Number
end

function RidgeRegression(::Type{T}, reg) where {T <: Number}
    return RidgeRegression(T.(reg))
end

function RidgeRegression()
    return RidgeRegression(0.0)
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
    QRFactorization()

Default solver for [`RidgeRegression`](@ref).

From LinearSolve.jl; available via `using ReservoirComputing`. For other
LinearSolve algorithms, load LinearSolve.jl and pass them as `solver`.
"""
const QRFactorization = LinearSolveQRFactorization

@doc raw"""
    QRSolver()

Legacy built-in QR solver for [`RidgeRegression`](@ref).

Prefer [`QRFactorization`](@ref) unless you need this path explicitly.
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

Fit a readout from precomputed features and targets.

## Arguments

  - `objective`: what to fit, e.g. [`RidgeRegression`](@ref), or an MLJ / LIBSVM
    regressor when those packages are loaded.
  - `states`: feature matrix from [`collectstates`](@ref), size
    `(n_features, T)`.
  - `target_data`: targets aligned with `states`, size `(n_outputs, T)`.

## Keyword arguments

  - `solver`: how to solve the fit when the objective needs one. For ridge,
    default `nothing` uses [`QRFactorization`](@ref); also
    [`QRSolver`](@ref) or other LinearSolve algorithms.
  - `kwargs...`: passed to the objective's backend when applicable.

## Returns

Readout weights (ridge) or the backend fit object (e.g. SVM models).
"""
function train(
        objective::RidgeRegression, states::AbstractMatrix, target_data::AbstractMatrix;
        solver = nothing, kwargs...
    )
    ridge_solver = _resolve_ridge_solver(solver)
    return _train_ridge(ridge_solver, objective, states, target_data; kwargs...)
end

function _ridge_augmented_system(
        objective::RidgeRegression,
        states::AbstractMatrix,
        targets::AbstractMatrix,
    )
    n_samples = size(states, 2)
    n_target_samples = size(targets, 2)
    n_samples == n_target_samples || throw(
        DimensionMismatch(
            "states has $n_samples samples, targets has $n_target_samples"
        )
    )

    n_features = size(states, 1)
    n_outputs = size(targets, 1)
    λ = convert(eltype(states), objective.reg)
    λ ≥ zero(λ) || throw(
        ArgumentError(
            "RidgeRegression regularization must be ≥ 0, got reg=$(objective.reg)"
        )
    )
    design = [states'; sqrt(λ) * I(n_features)]
    rhs = [targets'; zeros(eltype(targets), n_features, n_outputs)]
    return design, rhs
end

function _train_ridge(
        ::QRSolver, objective::RidgeRegression,
        states::AbstractMatrix, target_data::AbstractMatrix; kwargs...
    )
    design, rhs = _ridge_augmented_system(objective, states, target_data)
    weight_transpose = qr(design) \ rhs
    return Matrix(weight_transpose')
end

function _train_ridge(
        solver::SciMLLinearSolveAlgorithm, objective::RidgeRegression,
        states::AbstractMatrix, targets::AbstractMatrix; kwargs...
    )
    design, rhs = _ridge_augmented_system(objective, states, targets)
    solution = solve(LinearProblem(design, rhs), solver; kwargs...)
    return Matrix(solution.u')
end

function _train_ridge(
        solver, ::RidgeRegression, ::AbstractMatrix, ::AbstractMatrix; kwargs...
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
          objective=RidgeRegression(0.0), solver=nothing,
          washout=0, return_states=false)

Train the readout of a reservoir computer.

Builds features from `train_data`, fits them to `target_data` with `objective`,
and returns new parameters and states (inputs `ps` / `st` are not mutated).

## Arguments

  - `rc`: model with a trainable readout (e.g. [`ESN`](@ref),
    [`ReservoirChain`](@ref)).
  - `train_data`: inputs; columns are time steps.
  - `target_data`: targets aligned with `train_data`.
  - `ps`: model parameters.
  - `st`: model states.

## Keyword arguments

  - `objective`: what to fit. Default [`RidgeRegression`](@ref).
  - `solver`: how to solve when needed. For ridge, `nothing` uses
    [`QRFactorization`](@ref).
  - `washout`: initial time steps to drop from features and targets. Default `0`.
  - `return_states`: if `true`, also return the feature matrix used for the fit.
  - `kwargs...`: passed to the objective's backend when applicable.

## Returns

  - `(ps, st)`, or `((ps, st), states)` if `return_states=true`.
"""
function train(
        rc, train_data, target_data, ps, st;
        objective = RidgeRegression(0.0),
        solver = nothing,
        washout::Integer = 0,
        return_states::Bool = false,
        kwargs...
    )
    raw_states, st_after = collectstates(rc, train_data, ps, st)
    states_wo,
        targets_wo = washout > 0 ? _apply_washout(raw_states, target_data, washout) :
        (raw_states, target_data)
    output_matrix = _fit_readout(
        objective, states_wo, targets_wo, solver; kwargs...
    )
    ps2, st_after = addreadout!(rc, output_matrix, ps, st_after)
    return return_states ? ((ps2, st_after), states_wo) : (ps2, st_after)
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
