using Statistics

"""
    nrmse(pred, truth)

Per-channel NRMSE (Lukoševičius-style scale by channel std), then average
over channels. `pred` and `truth` are `(n_channels, T)`.
"""
function nrmse(pred::AbstractMatrix, truth::AbstractMatrix)
    size(pred) == size(truth) ||
        throw(DimensionMismatch("pred $(size(pred)) vs truth $(size(truth))"))
    n_ch = size(pred, 1)
    acc = 0.0
    for c in 1:n_ch
        y = @view truth[c, :]
        ŷ = @view pred[c, :]
        σ = std(y)
        σ = σ > 0 ? σ : one(σ)
        acc += sqrt(mean((ŷ .- y) .^ 2)) / σ
    end
    return acc / n_ch
end

"""
    nrmse_global(pred, truth)

Single scalar NRMSE using global std of `truth` (matches some #456 notes).
"""
function nrmse_global(pred::AbstractMatrix, truth::AbstractMatrix)
    return sqrt(mean((pred .- truth) .^ 2)) / std(truth)
end

"""
    valid_prediction_time(pred, truth; dt, λ_max, threshold=0.5)

First time (in Lyapunov units) where per-step relative error exceeds
`threshold`. Relative error at step k is
`‖pred[:,k] - truth[:,k]‖ / √(mean(truth.^2) + ε)`.
Returns `Inf` if never exceeded.
"""
function valid_prediction_time(
        pred::AbstractMatrix,
        truth::AbstractMatrix;
        dt::Real,
        λ_max::Real,
        threshold::Real = 0.5
    )
    scale = sqrt(mean(truth .^ 2)) + eps(Float64)
    T = size(pred, 2)
    for k in 1:T
        err = sqrt(sum(abs2, @view(pred[:, k]) .- @view(truth[:, k]))) / scale
        if err > threshold
            return (k - 1) * dt * λ_max
        end
    end
    return Inf
end

"""
    horizon_nrmse(pred, truth; horizons)

NRMSE on prefixes of length `h` for each `h` in `horizons`.
"""
function horizon_nrmse(pred::AbstractMatrix, truth::AbstractMatrix; horizons)
    return Dict(string(h) => nrmse(@view(pred[:, 1:h]), @view(truth[:, 1:h])) for h in horizons)
end
