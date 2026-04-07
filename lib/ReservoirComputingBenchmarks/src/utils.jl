"""
    _RidgeFactor

Pre-computed Cholesky factorization of the Gram matrix `X'X + λI` for
efficient multi-RHS ridge regression. Construct once, solve many times
with [`_ridge_solve`](@ref).
"""
struct _RidgeFactor{T <: Real}
    factor::LinearAlgebra.Cholesky{T, Matrix{T}}
    Xty_buf::Vector{T}
    w_buf::Vector{T}
end

function _ridge_factor(X::AbstractMatrix; reg::Real = 1.0)
    @assert reg >= 0 "Regularization coefficient must be non-negative, got $reg"
    T = promote_type(eltype(X), typeof(reg))
    n = size(X, 2)
    reg_T = convert(T, reg)
    G_reg = Matrix{T}(undef, n, n)
    X_col = convert(Matrix{T}, X)
    LinearAlgebra.copytri!(mul!(G_reg, X_col', X_col), 'U')
    @inbounds for i in 1:n
        G_reg[i, i] += reg_T
    end
    F = try
        cholesky!(Symmetric(G_reg))
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
    Xty_buf = Vector{T}(undef, n)
    w_buf = Vector{T}(undef, n)
    return _RidgeFactor(F, Xty_buf, w_buf)
end

function _ridge_solve!(rf::_RidgeFactor, X::AbstractMatrix, y::AbstractVector)
    mul!(rf.Xty_buf, X', y)
    ldiv!(rf.w_buf, rf.factor, rf.Xty_buf)
    return rf.w_buf
end

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

@inline function _squared_correlation(y_true::AbstractVector, y_pred::AbstractVector)
    r = cor(y_true, y_pred)
    if isnan(r)
        @warn "Squared correlation is NaN (zero-variance input). " *
            "This may indicate a degenerate reservoir or collapsed prediction."
        return 0.0
    end
    return clamp(r^2, 0.0, 1.0)
end

@inline function _nmse(y_true::AbstractVector, y_pred::AbstractVector)
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

@inline function _normalize(x::AbstractVector, lo::Real, hi::Real)
    xmin, xmax = extrema(x)
    xmin == xmax && return fill(convert(eltype(x), lo), length(x))
    return @. (x - xmin) / (xmax - xmin) * (hi - lo) + lo
end

@doc raw"""
    nmse(y_true, y_pred)

Normalized Mean Squared Error: ``\\text{mean}((y_\\text{true} - y_\\text{pred})^2) / \\text{var}(y_\\text{true})``.

Returns `NaN` with a warning if `y_true` has zero variance.
"""
@inline function nmse(y_true::AbstractVector, y_pred::AbstractVector)
    _nmse(y_true, y_pred)
end

@doc raw"""
    rnmse(y_true, y_pred)

Root Normalized Mean Squared Error: ``\\sqrt{\\text{nmse}(y_\\text{true}, y_\\text{pred})}``.

Returns `0.0` if NMSE is negative (due to numerical issues).
"""
@inline function rnmse(y_true::AbstractVector, y_pred::AbstractVector)
    nmse_val = nmse(y_true, y_pred)
    isnan(nmse_val) && return NaN
    nmse_val < 0 ? 0.0 : sqrt(nmse_val)
end

@doc raw"""
    mse(y_true, y_pred)

Mean Squared Error: ``\\text{mean}((y_\\text{true} - y_\\text{pred})^2)``.
"""
@inline function mse(y_true::AbstractVector, y_pred::AbstractVector)
    return mean((y_true .- y_pred) .^ 2)
end
