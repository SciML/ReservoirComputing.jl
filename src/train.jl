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
@concrete struct RidgeRegression
    reg <: Number
end

function RidgeRegression(::Type{T}, reg) where {T <: Number}
    return RidgeRegression(T.(reg))
end

function RidgeRegression()
    return RidgeRegression(0.0)
end

function __apply_washout(states::AbstractMatrix, targets::AbstractMatrix, washout::Integer)
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

__set_readout(ps, m::ReservoirChain, W) = first(addreadout!(m, W, ps, NamedTuple()))

"""
    AbstractReservoirComputingSolver

Developer marker for the package's legacy reservoir-training solver family.

## Extension contract

`QRSolver` is the only built-in subtype. The public [`train`](@ref) API also
accepts `LinearSolve.jl` algorithms directly. There is currently no public
generic extension point for arbitrary `AbstractReservoirComputingSolver`
subtypes: a new subtype is rejected by ridge training unless ReservoirComputing
adds a corresponding implementation itself.

For a custom solver, implement the documented `LinearSolve.jl` algorithm
interface and pass that algorithm to `train(...; solver=...)`. Do not extend
private training helpers from another package.

## Example

```julia
weights = train(RidgeRegression(1.0e-3), states, targets;
    solver = QRFactorization())
```
"""
abstract type AbstractReservoirComputingSolver end

@doc raw"""
    QRFactorization()

Default solver for [`RidgeRegression`](@ref). This is ReservoirComputing's
owned solver facade; it dispatches to LinearSolve's QR factorization
implementation. For other algorithms, load LinearSolve.jl and pass a
documented LinearSolve algorithm as `solver`.
"""
struct QRFactorization <: AbstractReservoirComputingSolver end

@doc raw"""
    QRSolver()

Legacy built-in QR solver for [`RidgeRegression`](@ref).

Prefer [`QRFactorization`](@ref) unless you need this path explicitly.
"""
struct QRSolver <: AbstractReservoirComputingSolver end

__default_ridge_solver() = QRFactorization()
__resolve_ridge_solver(::Nothing) = __default_ridge_solver()
__resolve_ridge_solver(solver) = solver

function __fit_readout(
        objective::RidgeRegression, states::AbstractMatrix, target_data::AbstractMatrix;
        solver = nothing, kwargs...
    )
    ridge_solver = __resolve_ridge_solver(solver)
    return __train_ridge(ridge_solver, objective, states, target_data; kwargs...)
end

function __ridge_augmented_system(
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
    n_samples > 0 || throw(
        ArgumentError("ridge regression requires at least one training sample")
    )

    n_features = size(states, 1)
    n_outputs = size(targets, 1)
    T = promote_type(eltype(states), eltype(targets), typeof(objective.reg))
    states = T.(states)
    targets = T.(targets)
    λ = convert(T, objective.reg)
    λ ≥ zero(λ) || throw(
        ArgumentError(
            "RidgeRegression regularization must be ≥ 0, got reg=$(objective.reg)"
        )
    )
    design = [states'; sqrt(λ) * I(n_features)]
    rhs = [targets'; zeros(T, n_features, n_outputs)]
    return design, rhs
end

function __train_ridge(
        ::QRSolver, objective::RidgeRegression,
        states::AbstractMatrix, target_data::AbstractMatrix; kwargs...
    )
    design, rhs = __ridge_augmented_system(objective, states, target_data)
    weight_transpose = qr(design) \ rhs
    return Matrix(weight_transpose')
end

function __train_ridge(
        ::QRFactorization, objective::RidgeRegression,
        states::AbstractMatrix, targets::AbstractMatrix; kwargs...
    )
    return __train_ridge(
        LinearSolveQRFactorization(), objective, states, targets; kwargs...
    )
end

function __train_ridge(
        solver::AbstractReservoirComputingSolver, ::RidgeRegression,
        ::AbstractMatrix, ::AbstractMatrix; kwargs...
    )
    throw(
        ArgumentError(
            "solver $(typeof(solver)) is not supported. Pass QRFactorization(), " *
                "QRSolver(), or a documented LinearSolve.jl algorithm instead."
        )
    )
end

function __train_ridge(
        solver, objective::RidgeRegression,
        states::AbstractMatrix, targets::AbstractMatrix; kwargs...
    )
    solver isa AbstractLinearAlgorithm || throw(
        ArgumentError(
            "solver $(typeof(solver)) is not supported. Pass QRFactorization(), " *
                "QRSolver(), or a documented LinearSolve.jl algorithm instead."
        )
    )
    design, rhs = __ridge_augmented_system(objective, states, targets)
    solution = try
        solve(LinearProblem(design, rhs), solver; kwargs...)
    catch err
        err isa DimensionMismatch || rethrow()
        throw(
            ArgumentError(
                "solver $(typeof(solver)) requires a square matrix, but ridge regression's " *
                    "augmented system is always rectangular (more rows than features). " *
                    "Use QRFactorization(), SVDFactorization(), or NormalCholeskyFactorization() " *
                    "instead."
            )
        )
    end
    successful_retcode(solution) || throw(
        ArgumentError("solver $(typeof(solver)) failed to solve the ridge regression system")
    )
    return Matrix(solution.u')
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
        targets_wo = washout > 0 ? __apply_washout(raw_states, target_data, washout) :
        (raw_states, target_data)
    output_matrix = if isnothing(solver)
        __fit_readout(objective, states_wo, targets_wo; kwargs...)
    else
        __fit_readout(objective, states_wo, targets_wo; solver = solver, kwargs...)
    end
    ps2, st_after = addreadout!(rc, output_matrix, ps, st_after)
    return return_states ? ((ps2, st_after), states_wo) : (ps2, st_after)
end

@generated function __setweight_rt(p::NamedTuple{K}, W) where {K}
    keys = K
    Kq = __quote_keys(keys)
    idx = findfirst(==(Symbol(:weight)), keys)

    terms = Any[]
    for i in 1:length(keys)
        push!(terms, (idx === i) ? :(W) : :(getfield(p, $i)))
    end

    if idx === nothing
        newK = __quote_keys((keys..., :weight))
        return :(NamedTuple{$newK}(($(terms...), W)))
    else
        return :(NamedTuple{$Kq}(($(terms...),)))
    end
end

@generated function __addreadout(layers::NamedTuple{K}, ps::NamedTuple{K}, W) where {K}
    if length(K) == 0
        return :(NamedTuple())
    end
    tailK = Base.tail(K)
    Kq = __quote_keys(K)
    tailKq = __quote_keys(tailK)

    head_val = :(
        (getfield(layers, 1) isa LinearReadout)
            ? __setweight_rt(getfield(ps, 1), W)
            : getfield(ps, 1)
    )

    tail_call = :(
        __addreadout(
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
    propertynames(rc.layers) == propertynames(ps) || throw(
        ArgumentError(
            "parameter keys $(propertynames(ps)) must match ReservoirChain layer keys " *
                "$(propertynames(rc.layers))"
        )
    )
    new_ps = __addreadout(rc.layers, ps, W)
    return new_ps, st
end
