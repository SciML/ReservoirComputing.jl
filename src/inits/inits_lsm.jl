@doc raw"""
    dale_sparse([rng], [T], dims...;
        excitatory_fraction=0.8, sparsity=0.1, std=1.0,
        radius=1.0, ei_weight_ratio=1.0, return_sparse=false)

Create a Dale-compliant sparse reservoir matrix. The first
`round(excitatory_fraction * n)` columns are excitatory (non-negative
outgoing weights); the rest are inhibitory (non-positive). Inhibitory
magnitudes are scaled by `ei_weight_ratio` before the spectral radius is
set to `radius`.

## Arguments

  - `rng`: Random number generator. Default is `Utils.default_rng()` from
    [WeightInitializers](https://lux.csail.mit.edu/stable/api/Building_Blocks/WeightInitializers).
  - `T`: Element type of the matrix. Default is `Float32`.
  - `dims`: Dimensions of the reservoir matrix. Must be square.

## Keyword arguments

  - `excitatory_fraction`: Fraction of excitatory units in `(0, 1)`.
    Default: `0.8`.
  - `sparsity`: Fraction of zero entries. Default: `0.1`.
  - `std`: Standard deviation passed to the sparse initialiser.
    Default: `1.0`.
  - `radius`: Target spectral radius. Default: `1.0`.
  - `ei_weight_ratio`: Multiplier on inhibitory magnitudes before spectral
    scaling. Default: `1.0`.
  - `return_sparse`: Return a sparse matrix when `true` (`SparseArrays`
    required). Default: `false`.

## Examples

```jldoctest dalesparse
julia> rng = MersenneTwister(123);

julia> W = dale_sparse(rng, 10, 10; excitatory_fraction = 0.8, sparsity = 0.3);

julia> size(W) == (10, 10) && all(>=(0), W[:, 1:8]) && all(<=(0), W[:, 9:10])
true
```
"""
function dale_sparse(
        rng::AbstractRNG, ::Type{T}, dims::Integer...;
        excitatory_fraction = T(0.8),
        sparsity = T(0.1),
        std = T(1.0),
        radius = T(1.0),
        ei_weight_ratio = T(1.0),
        return_sparse::Bool = false
    ) where {T <: Number}
    throw_sparse_error(return_sparse)
    check_res_size(dims...)
    n = dims[1]
    dims[1] == dims[2] || throw(
        ArgumentError("dale_sparse requires a square matrix, got $dims")
    )
    (0 < excitatory_fraction < 1) || throw(
        ArgumentError(
            "excitatory_fraction must be in (0, 1), got $excitatory_fraction"
        )
    )
    radius > 0 || throw(ArgumentError("radius must be positive, got $radius"))
    ei_weight_ratio > 0 || throw(
        ArgumentError("ei_weight_ratio must be positive, got $ei_weight_ratio")
    )
    lcl_sparsity = T(1) - T(sparsity)
    W = sparse_init(rng, T, dims...; sparsity = lcl_sparsity, std = T(std))
    n_exc = clamp(round(Int, T(excitatory_fraction) * n), 1, n - 1)
    @inbounds for col in 1:n
        sign = col <= n_exc ? one(T) : -T(ei_weight_ratio)
        for row in 1:n
            W[row, col] = abs(W[row, col]) * sign
        end
    end
    scale_radius!(W, T(radius))
    check_inf_nan(W)
    return return_init_as(Val(return_sparse), W)
end
