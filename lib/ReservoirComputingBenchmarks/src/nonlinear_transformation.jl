@doc raw"""
    nonlinear_transformation(input, states; f=x->sin(π*x), train_ratio=0.8,
                             reg=1.0, metric=nmse, washout::Integer=0)

Evaluate how well a reservoir reproduces a memoryless nonlinear transformation
``f(u(t))`` of the current input.

A ridge regression readout is trained from `states` to the target `f.(input)`,
and the metric is reported on a held-out tail of the trajectory.

## Arguments

  - `input::AbstractVector`: driving input signal of length `T`.
  - `states::AbstractMatrix`: reservoir state matrix of size `(n_features, T)`.

## Keyword Arguments

  - `f`: scalar nonlinearity applied element-wise to `input`. Default ``\sin(\pi x)``.
  - `train_ratio::Real=0.8`: fraction of valid data used for training.
  - `reg::Real=1.0`: ridge regression regularization coefficient.
  - `metric`: error metric `(y_true, y_pred) -> score`. Built-ins:
    [`nmse`](@ref), [`rnmse`](@ref), [`mse`](@ref).
  - `washout::Integer=0`: number of initial time steps discarded before
    training/testing.

## Returns

A `NamedTuple` with fields:

  - `score::Float64`: error metric on the test split.
  - `target::Vector{Float64}`: full target signal `f.(input)`.

## References

  - Lukoševičius, M. (2012). "A practical guide to applying echo state networks."
    *Neural Networks: Tricks of the Trade*, Springer.
"""
function nonlinear_transformation(
        input::AbstractVector,
        states::AbstractMatrix;
        f = x -> sin(π * x),
        train_ratio::Real = 0.8,
        reg::Real = 1.0,
        metric = nmse,
        washout::Integer = 0,
    )
    T = length(input)
    @assert size(states, 2) == T "states must have $T columns (time steps), got $(size(states, 2))"
    @assert 0 <= washout < T "washout must be in [0, $T), got $washout"

    valid = (washout + 1):T
    T_valid = length(valid)
    @assert T_valid >= 2 "Not enough samples after washout: $T_valid"

    X = Matrix{Float64}(undef, T_valid, size(states, 1))
    copyto!(X, view(states, :, valid)')

    target = Vector{Float64}(undef, T)
    @inbounds for t in 1:T
        target[t] = float(f(input[t]))
    end

    train_idx, test_idx = _train_test_split(T_valid, train_ratio)

    y_valid = view(target, valid)
    w = _ridge_regression(view(X, train_idx, :), view(y_valid, train_idx); reg = reg)
    y_pred = view(X, test_idx, :) * w
    score = metric(view(y_valid, test_idx), y_pred)

    return (score = score, target = target)
end
