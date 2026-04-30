@doc raw"""
    sin_approximation(input, states; freq=π, train_ratio=0.8, reg=1.0,
                      metric=nmse, washout::Integer=0)

Evaluate how well a reservoir approximates ``\sin(\omega\, u(t))``.

This is a memoryless nonlinear task. It is a thin wrapper around
[`nonlinear_transformation`](@ref) with ``f(x) = \sin(\omega x)``.

## Arguments

  - `input::AbstractVector`: driving input signal of length `T`.
  - `states::AbstractMatrix`: reservoir state matrix of size `(n_features, T)`.

## Keyword Arguments

  - `freq::Real=π`: angular frequency ``\omega``.
  - Remaining keyword arguments are forwarded to
    [`nonlinear_transformation`](@ref).

## Returns

A `NamedTuple` with fields:

  - `score::Float64`: error metric on the test split.
  - `target::Vector{Float64}`: full target signal `sin(freq * input)`.
"""
function sin_approximation(
        input::AbstractVector,
        states::AbstractMatrix;
        freq::Real = π,
        kwargs...,
    )
    return nonlinear_transformation(input, states; f = x -> sin(freq * x), kwargs...)
end
