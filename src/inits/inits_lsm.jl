@doc raw"""
    dale_sparse([rng], [T], dims...;
        excitatory_fraction=0.8, sparsity=0.1, std=1.0,
        radius=1.0, ei_weight_ratio=1.0, return_sparse=false)

Dale-compliant sparse reservoir: the first
`round(excitatory_fraction * n)` units are excitatory (non-negative
outgoing weights), the rest inhibitory (non-positive). Spectral radius
is scaled to `radius`. Inhibitory magnitudes are multiplied by
`ei_weight_ratio` before scaling.
"""
function dale_sparse(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        excitatory_fraction::Number = T(0.8),
        sparsity::Number = T(0.1),
        std::Number = T(1.0),
        radius::Number = T(1.0),
        ei_weight_ratio::Number = T(1.0),
        return_sparse::Bool = false
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    n = dims[1]
    dims[1] == dims[2] || throw(ArgumentError("dale_sparse requires a square matrix, got $dims"))
    (0 < excitatory_fraction < 1) || throw(
        ArgumentError("excitatory_fraction must be in (0, 1), got $excitatory_fraction")
    )
    lcl_sparsity = T(1) - T(sparsity)
    W = sparse_init(rng, T, dims...; sparsity = lcl_sparsity, std = T(std))
    n_exc = clamp(round(Int, T(excitatory_fraction) * n), 1, n - 1)
    @inbounds for j in 1:n
        s = j <= n_exc ? one(T) : -T(ei_weight_ratio)
        for i in 1:n
            W[i, j] = abs(W[i, j]) * s
        end
    end
    scale_radius!(W, T(radius))
    check_inf_nan(W)
    return return_init_as(Val(return_sparse), W)
end
