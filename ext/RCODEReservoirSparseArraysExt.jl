module RCODEReservoirSparseArraysExt

using LinearAlgebra: I
using ReservoirComputing: ReservoirComputing
using SparseArrays: SparseMatrixCSC

# `pattern(W_r) ∪ diag` is the exact Jacobian sparsity of the built-in
# leaky-integrator ESN RHS. Custom `equations` fall through to the
# generic `nothing` fallback so an off-`W_r` coupling isn't given a
# too-narrow prototype.
function ReservoirComputing._reservoir_jac_prototype(
        ::typeof(ReservoirComputing._continuous_esn_rhs!),
        reservoir_matrix::SparseMatrixCSC,
    )
    return reservoir_matrix + I
end

end # module
