@doc raw"""
    nonlinear_memory(input, states; f=x->x^2, max_delay=30, train_ratio=0.8, reg=1.0)

Compute the nonlinear memory capacity of a reservoir following Inubushi &
Yoshimura (2017). For each delay ``k = 1, \ldots, K``, a ridge regression
readout is trained to reconstruct ``f(u(t-k))`` from the reservoir state at
time ``t``. The per-delay capacity is the squared Pearson correlation between
target and prediction, and the total capacity is the sum.

When ``f(x) = x`` this reduces to the linear [`memory_capacity`](@ref).

```math
NMC_k = \frac{\text{cov}^2(f(u(t-k)),\, \hat{y}_k(t))}
              {\text{var}(f(u(t-k)))\;\text{var}(\hat{y}_k(t))}
```

## Arguments

  - `input::AbstractVector`: input signal of length `T` (typically i.i.d.
    uniform in ``[-1, 1]``).
  - `states::AbstractMatrix`: reservoir state matrix of size `(n_features, T)`.

## Keyword Arguments

  - `f`: scalar nonlinearity applied to the delayed input. Default ``x \mapsto x^2``.
  - `max_delay::Int=30`: maximum delay ``K``.
  - `train_ratio::Real=0.8`: fraction of valid data used for training.
  - `reg::Real=1.0`: ridge regression regularization coefficient.

## Returns

A `NamedTuple` with fields:

  - `total::Float64`: total nonlinear memory capacity.
  - `delays::Vector{Float64}`: per-delay capacities ``NMC_k``.

## References

  - Inubushi, M. & Yoshimura, K. (2017). "Reservoir computing beyond
    memory-nonlinearity trade-off." *Scientific Reports*, 7, 10199.
"""
function nonlinear_memory(
        input::AbstractVector,
        states::AbstractMatrix;
        f = x -> x^2,
        max_delay::Int = 30,
        train_ratio::Real = 0.8,
        reg::Real = 1.0,
    )
    T = length(input)
    @assert size(states, 2) == T "states must have $T columns (time steps), got $(size(states, 2))"
    @assert max_delay >= 1 "max_delay must be >= 1, got $max_delay"
    @assert max_delay < T "max_delay ($max_delay) must be less than signal length ($T)"

    valid = (max_delay + 1):T
    T_valid = length(valid)
    X = Matrix{Float64}(undef, T_valid, size(states, 1))
    copyto!(X, view(states, :, valid)')

    train_idx, test_idx = _train_test_split(T_valid, train_ratio)
    X_train = view(X, train_idx, :)
    X_test = view(X, test_idx, :)

    rf = _ridge_factor(X_train; reg = reg)

    delay_capacities = zeros(max_delay)
    target = Vector{Float64}(undef, T_valid)

    @inbounds for k in 1:max_delay
        for (i, t) in enumerate(valid)
            target[i] = float(f(input[t - k]))
        end

        y_train = view(target, train_idx)
        y_test = view(target, test_idx)

        w = _ridge_solve!(rf, X_train, y_train)
        y_pred = X_test * w

        delay_capacities[k] = _squared_correlation(y_test, y_pred)
    end

    return (total = sum(delay_capacities), delays = delay_capacities)
end
