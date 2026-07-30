module RCODEReservoirSparseArraysExt

using LinearAlgebra: I
using ReservoirComputing: ReservoirComputing
using SparseArrays: AbstractSparseMatrixCSC

# Adds the sparse-Wr specialisation of the fallback in
# `src/layers/continuous_esn_cell.jl`. `pattern(W_r) ∪ diag` is the exact
# sparsity of the analytic Jacobian of the built-in leaky-integrator ESN
# RHS; custom `equations` deliberately fall through to the generic
# `nothing` fallback so a user RHS with off-`W_r` coupling isn't given a
# too-narrow prototype.
function ReservoirComputing._reservoir_jac_prototype(
        ::typeof(ReservoirComputing._continuous_esn_rhs!),
        reservoir_matrix::AbstractSparseMatrixCSC,
    )
    return reservoir_matrix + I
end

end # module
