@doc raw"""
    memory_capacity(input, states; max_delay=30, train_ratio=0.8, reg=1.0)

Compute the linear memory capacity of a reservoir following Jaeger (2002).

For each delay ``k = 1, \ldots, K``, a ridge regression readout is trained to
reconstruct the input delayed by ``k`` steps. The per-delay capacity is the
squared Pearson correlation between the true delayed input and the prediction:

```math
MC_k = \frac{\text{cov}^2(u(t-k),\, \hat{u}_k(t))}
            {\text{var}(u(t-k))\;\text{var}(\hat{u}_k(t))}
```

The total memory capacity is ``MC = \sum_{k=1}^{K} MC_k``, bounded above by the
number of reservoir nodes ``N`` (Jaeger, 2002).

## Arguments

  - `input::AbstractVector`: input signal of length `T` (typically i.i.d.
    uniform in ``[-1, 1]``).
  - `states::AbstractMatrix`: reservoir state matrix of size `(n_features, T)`.

## Keyword Arguments

  - `max_delay::Int=30`: maximum delay ``K`` to evaluate.
  - `train_ratio::Real=0.8`: fraction of valid data used for training.
  - `reg::Real=1.0`: ridge regression regularization coefficient ``\lambda``.

## Returns

A `NamedTuple` with fields:

  - `total::Float64`: total memory capacity ``MC``.
  - `delays::Vector{Float64}`: per-delay capacities ``MC_k`` for
    ``k = 1, \ldots, K``.

## References

  - Jaeger, H. (2002). "Short term memory in echo state networks."
    GMD Report 152.
"""
function memory_capacity(
        input::AbstractVector,
        states::AbstractMatrix;
        max_delay::Int = 30,
        train_ratio::Real = 0.8,
        reg::Real = 1.0,
    )
    T = length(input)
    @assert size(states, 2) == T "states must have $T columns (time steps), got $(size(states, 2))"
    @assert max_delay >= 1 "max_delay must be >= 1, got $max_delay"
    @assert max_delay < T "max_delay ($max_delay) must be less than signal length ($T)"

    # Discard first max_delay steps to avoid edge effects from delays
    valid = (max_delay + 1):T
    T_valid = length(valid)
    X = collect(states[:, valid]')  # (T_valid, n_features)

    train_idx, test_idx = _train_test_split(T_valid, train_ratio)

    X_train = X[train_idx, :]
    X_test = X[test_idx, :]

    # Pre-compute Cholesky factorization — reused across all delays
    rf = _ridge_factor(X_train; reg = reg)

    delay_capacities = zeros(max_delay)

    for k in 1:max_delay
        # Target: input shifted back by k steps
        target = input[valid .- k]

        y_train = target[train_idx]
        y_test = target[test_idx]

        w = _ridge_solve(rf, X_train, y_train)
        y_pred = X_test * w

        delay_capacities[k] = _squared_correlation(y_test, y_pred)
    end

    return (total = sum(delay_capacities), delays = delay_capacities)
end
