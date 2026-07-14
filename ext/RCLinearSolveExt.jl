module RCLinearSolveExt

using LinearAlgebra: I
using LinearSolve: LinearProblem, solve, SciMLLinearSolveAlgorithm
using ReservoirComputing: StandardRidge
import ReservoirComputing: _train_ridge

function _train_ridge(
        solver::SciMLLinearSolveAlgorithm, sr::StandardRidge,
        states::AbstractMatrix, targets::AbstractMatrix; kwargs...
    )
    n_features, n_samples = size(states)
    n_outputs, n_target_samples = size(targets)
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

end #module
