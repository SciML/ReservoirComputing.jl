module RCLinearSolveExt
using LinearAlgebra: mul!, I
using LinearSolve: LinearProblem, init, solve!, SciMLLinearSolveAlgorithm
using ReservoirComputing: StandardRidge
import ReservoirComputing: _train_ridge

function _train_ridge(solver::SciMLLinearSolveAlgorithm, sr::StandardRidge,
        states::AbstractMatrix, targets::AbstractMatrix; kwargs...)

    nfeat, T = size(states)
    nout,  T2 = size(targets)
    T == T2 || throw(DimensionMismatch("states has T=$T samples, targets has T=$T2"))
    λ = convert(eltype(states), sr.reg)
    A = states * states' + λ * I
    b = zeros(eltype(states), nfeat)
    prob = LinearProblem(A, b)
    linsolve = init(prob, solver; kwargs...)
    Wt = zeros(eltype(states), nfeat, nout)
    for idx in 1:nout
        mul!(linsolve.b, states, targets[idx, :])
        sol = solve!(linsolve)
        Wt[:, idx] .= sol.u
    end

    return permutedims(Wt)  # (n_outputs, n_features)
end

end #module
