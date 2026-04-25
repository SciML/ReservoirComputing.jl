@doc raw"""
    _effective_rank(M; threshold=0.01)

Numerical rank of a matrix `M`: the number of singular values strictly greater
than `threshold * maximum_singular_value`.

The threshold is relative to the largest singular value, making the rank
estimate scale-invariant. Returns `0` for empty or all-zero matrices.
"""
function _effective_rank(M::AbstractMatrix; threshold::Real = 0.01)
    @assert threshold > 0 "threshold must be positive, got $threshold"
    isempty(M) && return 0
    s = LinearAlgebra.svdvals(M)
    s_max = isempty(s) ? zero(eltype(s)) : maximum(s)
    s_max <= 0 && return 0
    return count(>(threshold * s_max), s)
end

@doc raw"""
    kernel_rank(states; threshold=0.01)

Compute the **kernel rank** of a reservoir, defined as the numerical rank of a
matrix whose columns are reservoir final states reached after driving the
reservoir with `n` distinct random input streams.

A higher kernel rank indicates a richer separation property: the reservoir
maps different inputs to linearly independent state vectors, which is a
prerequisite for high computational power.

## Arguments

  - `states::AbstractMatrix`: matrix of size `(n_features, n)` whose columns
    are final reservoir states from `n` independent driving runs.

## Keyword Arguments

  - `threshold::Real=0.01`: relative threshold on singular values.

## Returns

  - `Int`: numerical rank, in `[0, min(n_features, n)]`.

## References

  - Legenstein, R. & Maass, W. (2007). "Edge of chaos and prediction of
    computational performance for neural circuit models." *Neural Networks*,
    20(3), 323–334.
"""
function kernel_rank(states::AbstractMatrix; threshold::Real = 0.01)
    return _effective_rank(states; threshold = threshold)
end

@doc raw"""
    generalization_rank(states; threshold=0.01)

Compute the **generalization rank** of a reservoir, defined as the numerical
rank of a matrix whose columns are reservoir final states reached after
driving the reservoir with `n` slightly perturbed copies of a common input
stream.

A *lower* generalization rank indicates that the reservoir collapses similar
inputs to similar states (good generalization). The difference
`kernel_rank - generalization_rank` peaks at the *edge of chaos* and predicts
computational performance (Legenstein & Maass, 2007).

## Arguments

  - `states::AbstractMatrix`: matrix of size `(n_features, n)` whose columns
    are final reservoir states from `n` perturbed driving runs.

## Keyword Arguments

  - `threshold::Real=0.01`: relative threshold on singular values.

## Returns

  - `Int`: numerical rank, in `[0, min(n_features, n)]`.

## References

  - Legenstein, R. & Maass, W. (2007). "Edge of chaos and prediction of
    computational performance for neural circuit models." *Neural Networks*,
    20(3), 323–334.
"""
function generalization_rank(states::AbstractMatrix; threshold::Real = 0.01)
    return _effective_rank(states; threshold = threshold)
end
