"""
Standard NARMA coefficients from the literature.

  - NARMA-2:  Atiya & Parlos (2000), Eq. 12
  - NARMA-10: Atiya & Parlos (2000), standard benchmark
  - NARMA-30: Schrauwen et al. (2008)
"""
const NARMA_COEFFICIENTS = Dict{Int, NamedTuple{(:alpha, :beta, :gamma, :delta), NTuple{4, Float64}}}(
    2 => (alpha = 0.4, beta = 0.4, gamma = 0.6, delta = 0.1),
    10 => (alpha = 0.3, beta = 0.05, gamma = 1.5, delta = 0.1),
    30 => (alpha = 0.2, beta = 0.04, gamma = 1.5, delta = 0.001),
)

@doc raw"""
    generate_narma(input; order=10, alpha=nothing, beta=nothing, gamma=nothing, delta=nothing, normalize=true)

Generate a NARMA target signal from the given input.

For `order == 2` (Atiya & Parlos 2000, Eq. 12):

```math
y(k{+}1) = \alpha\, y(k) + \beta\, y(k)\, y(k{-}1)
       + \gamma\, u(k)^3 + \delta
```

For `order >= 3` (Atiya & Parlos 2000, general NARMA-N):

```math
y(t{+}1) = \alpha\, y(t) + \beta\, y(t) \sum_{i=0}^{N-1} y(t{-}i)
            + \gamma\, u(t{-}N{+}1)\, u(t) + \delta
```

## Arguments

  - `input::AbstractVector`: driving input signal of length `T`.

## Keyword Arguments

  - `order::Int=10`: NARMA system order ``N``. Must be ``\geq 2``.
  - `alpha`, `beta`, `gamma`, `delta`: recurrence coefficients. When `nothing`
    (default), standard values are used for orders 2, 10, and 30. All four
    must be provided explicitly for other orders.
  - `normalize::Bool=true`: normalize input to ``[0, 0.5]`` before computing
    the recurrence.

## Returns

  - `Vector{Float64}`: NARMA target signal of length `T`.

## References

  - Atiya, A.F. & Parlos, A.G. (2000). "New results on recurrent network
    training." *IEEE Trans. Neural Networks*, 11(3).
  - Schrauwen, B. et al. (2008). "Improving reservoirs using intrinsic
    plasticity." *Neurocomputing*, 71(7–9).
"""
function generate_narma(
        input::AbstractVector;
        order::Int = 10,
        alpha::Union{Real, Nothing} = nothing,
        beta::Union{Real, Nothing} = nothing,
        gamma::Union{Real, Nothing} = nothing,
        delta::Union{Real, Nothing} = nothing,
        normalize::Bool = true,
    )
    @assert order >= 2 "NARMA order must be >= 2, got $order"

    # Resolve coefficients
    if haskey(NARMA_COEFFICIENTS, order)
        c = NARMA_COEFFICIENTS[order]
        alpha = something(alpha, c.alpha)
        beta = something(beta, c.beta)
        gamma = something(gamma, c.gamma)
        delta = something(delta, c.delta)
    else
        @assert !isnothing(alpha) && !isnothing(beta) && !isnothing(gamma) && !isnothing(delta) (
            "No default coefficients for NARMA-$order. " *
                "Provide alpha, beta, gamma, delta explicitly."
        )
    end

    T = length(input)
    @assert T > order "Input length ($T) must be greater than order ($order)"

    u = normalize ? _normalize(input, 0.0, 0.5) : Float64.(input)
    y = zeros(Float64, T)

    if order == 2
        # NARMA-2: y[t] = α y[t-1] + β y[t-1] y[t-2] + γ u[t-1]³ + δ
        for t in 3:T
            y[t] = alpha * y[t - 1] +
                beta * y[t - 1] * y[t - 2] +
                gamma * u[t - 1]^3 +
                delta
        end
    else
        # NARMA-N (N >= 3): standard recurrence
        # y[t] = α y[t-1] + β y[t-1] Σ y[(t-N):(t-1)] + γ u[t-N] u[t-1] + δ
        for t in (order + 1):T
            y[t] = alpha * y[t - 1] +
                beta * y[t - 1] * sum(@view y[(t - order):(t - 1)]) +
                gamma * u[t - order] * u[t - 1] +
                delta
        end
    end

    if any(!isfinite, y)
        idx = findfirst(!isfinite, y)
        @warn "NARMA-$order target diverged at time step $idx (value: $(y[idx])). " *
            "Consider using normalize=true or adjusting coefficients."
    end

    return y
end

@doc raw"""
    narma(input, states; order=10, metric=:nmse, train_ratio=0.8, reg=1.0, washout=nothing, kwargs...)

Evaluate reservoir performance on the NARMA-N task.

Generates the NARMA target from `input`, trains a ridge regression readout
from `states` to the target, and computes the error metric on held-out data.

## Arguments

  - `input::AbstractVector`: driving input signal of length `T`.
  - `states::AbstractMatrix`: reservoir state matrix of size `(n_features, T)`.

## Keyword Arguments

  - `order::Int=10`: NARMA system order.
  - `metric::Symbol=:nmse`: error metric. One of `:nmse` (Normalized MSE),
    `:rnmse` (Root NMSE), or `:mse`.
  - `train_ratio::Real=0.8`: fraction of data used for training.
  - `reg::Real=1.0`: ridge regression regularization coefficient.
  - `washout::Union{Int,Nothing}=nothing`: number of initial time steps to
    discard. Defaults to `order`.
  - Remaining `kwargs` are forwarded to [`generate_narma`](@ref).

## Returns

A `NamedTuple` with fields:

  - `score::Float64`: the computed error metric.
  - `metric::Symbol`: which metric was used.
  - `target::Vector{Float64}`: the full NARMA target signal.
"""
function narma(
        input::AbstractVector,
        states::AbstractMatrix;
        order::Int = 10,
        metric::Symbol = :nmse,
        train_ratio::Real = 0.8,
        reg::Real = 1.0,
        washout::Union{Int, Nothing} = nothing,
        kwargs...,
    )
    T = length(input)
    @assert size(states, 2) == T "states must have $T columns (time steps), got $(size(states, 2))"

    target = generate_narma(input; order = order, kwargs...)

    # Skip initial transient where y ≈ 0
    wo = something(washout, order)
    @assert 0 <= wo < T "washout must be in [0, $T), got $wo"
    valid = (wo + 1):T
    T_valid = length(valid)

    X = collect(states[:, valid]')  # (T_valid, n_features)
    y = target[valid]

    train_idx, test_idx = _train_test_split(T_valid, train_ratio)

    w = _ridge_regression(X[train_idx, :], y[train_idx]; reg = reg)
    y_pred = X[test_idx, :] * w
    y_test = y[test_idx]

    score = if metric === :nmse
        _nmse(y_test, y_pred)
    elseif metric === :rnmse
        nmse_val = _nmse(y_test, y_pred)
        nmse_val < 0 ? 0.0 : sqrt(nmse_val)
    elseif metric === :mse
        mean((y_test .- y_pred) .^ 2)
    else
        throw(ArgumentError("Unknown metric :$metric. Use :nmse, :rnmse, or :mse."))
    end

    return (score = score, metric = metric, target = target)
end
