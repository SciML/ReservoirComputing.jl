"""
    _RidgeFactor

Pre-computed Cholesky factorization of the Gram matrix `X'X + λI` for
efficient multi-RHS ridge regression. Construct once, solve many times
with [`_ridge_solve`](@ref).
"""
struct _RidgeFactor{T <: Real}
    factor::LinearAlgebra.Cholesky{T, Matrix{T}}
    Xty_buf::Vector{T}   # scratch buffer for X'y products
end

"""
    _ridge_factor(X; reg=1.0)

Pre-compute the Cholesky factorization of `X'X + reg * I`.

- `X`: (T, n_features) design matrix
- `reg`: regularization coefficient (λ ≥ 0)

Returns a `_RidgeFactor` that can be reused across many targets.
"""
function _ridge_factor(X::AbstractMatrix; reg::Real = 1.0)
    @assert reg >= 0 "Regularization coefficient must be non-negative, got $reg"
    T = promote_type(eltype(X), typeof(reg))
    n = size(X, 2)
    X_T = Matrix{T}(X)
    G_reg = Matrix{T}(Symmetric(X_T' * X_T))
    reg_T = convert(T, reg)
    @inbounds for i in 1:n
        G_reg[i, i] += reg_T
    end
    F = try
        cholesky(Symmetric(G_reg))
    catch e
        if e isa LinearAlgebra.PosDefException
            throw(
                ArgumentError(
                    "Cholesky factorization failed: X'X + reg*I is not positive definite. " *
                        "This can happen when reg is zero or too small for rank-deficient data. " *
                        "Increase reg or provide full-rank features.",
                ),
            )
        end
        rethrow(e)
    end
    buf = Vector{T}(undef, n)
    return _RidgeFactor(F, buf)
end

"""
    _ridge_solve(rf, X, y)

Solve ridge regression using the pre-computed `_RidgeFactor`.
Returns weight vector `w` such that `X * w ≈ y`.
"""
function _ridge_solve(rf::_RidgeFactor, X::AbstractMatrix, y::AbstractVector)
    mul!(rf.Xty_buf, X', y)
    return rf.factor \ rf.Xty_buf
end

"""
    _ridge_regression(X, y; reg=1.0)

Solve ridge regression `X * w ≈ y` via Cholesky factorization of the
Gram matrix `X'X + λI`.

- `X`: (T, n_features) design matrix
- `y`: (T,) target vector
- `reg`: regularization coefficient (λ ≥ 0)

Returns weight vector `w` of length `n_features`.
"""
function _ridge_regression(X::AbstractMatrix, y::AbstractVector; reg::Real = 1.0)
    @assert reg >= 0 "Regularization coefficient must be non-negative, got $reg"
    rf = _ridge_factor(X; reg = reg)
    return _ridge_solve(rf, X, y)
end

"""
    _squared_correlation(y_true, y_pred)

Squared Pearson correlation coefficient between `y_true` and `y_pred`.
Returns 0.0 with a warning if either vector has zero variance.
"""
function _squared_correlation(y_true::AbstractVector, y_pred::AbstractVector)
    r = cor(y_true, y_pred)
    if isnan(r)
        @warn "Squared correlation is NaN (zero-variance input). " *
            "This may indicate a degenerate reservoir or collapsed prediction."
        return 0.0
    end
    return clamp(r^2, 0.0, 1.0)
end

"""
    _nmse(y_true, y_pred)

Normalized Mean Squared Error: `mean((y_true - y_pred).²) / var(y_true)`.
"""
function _nmse(y_true::AbstractVector, y_pred::AbstractVector)
    v = var(y_true)
    if v < eps(typeof(v))
        @warn "NMSE: target variance is near-zero ($v). " *
            "NMSE is undefined for constant targets."
        return NaN
    end
    return mean((y_true .- y_pred) .^ 2) / v
end

"""
    _train_test_split(n, train_ratio)

Return `(train_range, test_range)` index ranges for a temporal split.
"""
function _train_test_split(n::Int, train_ratio::Real)
    @assert 0 < train_ratio < 1 "train_ratio must be in (0, 1), got $train_ratio"
    split = floor(Int, n * train_ratio)
    @assert split >= 1 "Training set is empty (n=$n, train_ratio=$train_ratio). Provide more data."
    @assert split < n "Test set is empty (n=$n, train_ratio=$train_ratio). Reduce train_ratio or provide more data."
    return 1:split, (split + 1):n
end

"""
    _normalize(x, lo, hi)

Min-max normalize vector `x` to the interval `[lo, hi]`.
Returns a constant vector of `lo` if `x` is constant.
"""
function _normalize(x::AbstractVector, lo::Real, hi::Real)
    xmin, xmax = extrema(x)
    xmin == xmax && return fill(convert(eltype(x), lo), length(x))
    return @. (x - xmin) / (xmax - xmin) * (hi - lo) + lo
end
