@doc raw"""
    ipc(input, states; max_delay=10, max_degree=3, max_total_degree=nothing, cross_terms=true, train_ratio=0.8, reg=1.0)

Compute the Information Processing Capacity of a reservoir following
Dambre et al. (2012).

The IPC decomposes the total computational capacity of a reservoir into
contributions from orthonormal Legendre polynomial basis functions:

```math
z_l(t) = \prod_i \tilde{P}_{d_i}\!\bigl(u(t - \tau_i)\bigr)
```

where ``\tilde{P}_n(x) = P_n(x)\sqrt{(2n+1)/2}`` is the normalized Legendre
polynomial of degree ``n``.  For each basis function, a ridge regression readout
is trained, and the capacity is the squared correlation between the target and
prediction.

The total capacity satisfies ``C_{\text{total}} \leq N`` where ``N`` is the
number of reservoir nodes.

## Arguments

  - `input::AbstractVector`: input signal of length `T` (should be i.i.d.
    uniform in ``[-1, 1]``).
  - `states::AbstractMatrix`: reservoir state matrix of size `(n_features, T)`.

## Keyword Arguments

  - `max_delay::Int=10`: maximum delay ``\tau`` for basis functions.
  - `max_degree::Int=3`: maximum polynomial degree per variable.
  - `max_total_degree::Union{Int,Nothing}=nothing`: maximum sum of degrees
    in cross-terms. Defaults to `max_degree`.
  - `cross_terms::Bool=true`: include two-variable product basis functions.
  - `train_ratio::Real=0.8`: fraction of valid data used for training.
  - `reg::Real=1.0`: ridge regression regularization coefficient.

## Returns

A `NamedTuple` with fields:

  - `total::Float64`: total information processing capacity.
  - `linear::Float64`: capacity from degree-1 (linear memory) terms.
  - `nonlinear::Float64`: capacity from degree ``\geq 2`` terms.
  - `by_degree::Dict{Int,Float64}`: capacity aggregated by total polynomial
    degree.
  - `by_delay::Dict{Int,Float64}`: capacity aggregated by delay (single-variable
    terms only).
  - `basis_capacities::Vector{NamedTuple}`: per-basis-function results with
    fields `terms`, `degree`, `capacity`.
  - `theoretical_max::Int`: upper bound ``N`` (number of reservoir features).

## References

  - Dambre, J., Verstraeten, D., Schrauwen, B. & Massar, S. (2012).
    "Information processing capacity of dynamical systems."
    *Scientific Reports*, 2, 514.
"""
function ipc(
        input::AbstractVector,
        states::AbstractMatrix;
        max_delay::Int = 10,
        max_degree::Int = 3,
        max_total_degree::Union{Int, Nothing} = nothing,
        cross_terms::Bool = true,
        train_ratio::Real = 0.8,
        reg::Real = 1.0,
    )
    T = length(input)
    n_features = size(states, 1)
    @assert size(states, 2) == T "states must have $T columns (time steps), got $(size(states, 2))"
    @assert max_delay >= 1 "max_delay must be >= 1, got $max_delay"
    @assert max_degree >= 1 "max_degree must be >= 1, got $max_degree"
    @assert max_delay < T "max_delay ($max_delay) must be less than signal length ($T)"

    mtd = something(max_total_degree, max_degree)

    # Precompute normalized Legendre polynomial values for all (degree, delay)
    poly_cache = Dict{Tuple{Int, Int}, Vector{Float64}}()
    for delay in 1:max_delay
        for degree in 1:max_degree
            delayed = @view input[(max_delay + 1 - delay):(T - delay)]
            poly_cache[(delay, degree)] = _normalized_legendre.(degree, delayed)
        end
    end

    # Generate basis functions
    basis_functions = _enumerate_basis_functions(max_delay, max_degree, mtd, cross_terms)

    # Valid time range (after max_delay to avoid edge effects)
    valid = (max_delay + 1):T
    T_valid = length(valid)
    X = collect(states[:, valid]')  # (T_valid, n_features)

    train_idx, test_idx = _train_test_split(T_valid, train_ratio)
    X_train = X[train_idx, :]
    X_test = X[test_idx, :]

    # Pre-compute Cholesky factorization — reused across all basis functions
    rf = _ridge_factor(X_train; reg = reg)

    results = Vector{NamedTuple{(:terms, :degree, :capacity), Tuple{Vector{Tuple{Int, Int}}, Int, Float64}}}()
    by_degree = Dict{Int, Float64}()
    by_delay = Dict{Int, Float64}()

    # Pre-allocate target buffer — reused via fill!
    target = Vector{Float64}(undef, T_valid)

    for terms in basis_functions
        total_degree = sum(d for (_, d) in terms)

        # Compute target: product of normalized Legendre polynomials
        fill!(target, 1.0)
        for (delay, degree) in terms
            target .*= poly_cache[(delay, degree)]
        end

        y_train = target[train_idx]
        y_test = target[test_idx]

        w = _ridge_solve(rf, X_train, y_train)
        y_pred = X_test * w

        cap = _squared_correlation(y_test, y_pred)

        push!(results, (terms = terms, degree = total_degree, capacity = cap))
        by_degree[total_degree] = get(by_degree, total_degree, 0.0) + cap

        # Track per-delay capacity for single-variable terms
        if length(terms) == 1
            delay = terms[1][1]
            by_delay[delay] = get(by_delay, delay, 0.0) + cap
        end
    end

    total_cap = isempty(results) ? 0.0 : sum(r.capacity for r in results)
    linear_cap = get(by_degree, 1, 0.0)
    nonlinear_cap = total_cap - linear_cap

    return (
        total = total_cap,
        linear = linear_cap,
        nonlinear = nonlinear_cap,
        by_degree = by_degree,
        by_delay = by_delay,
        basis_capacities = results,
        theoretical_max = n_features,
    )
end

# ── Legendre polynomials ─────────────────────────────────────────────────────

"""
    _legendre(n, x)

Evaluate the Legendre polynomial ``P_n(x)`` using the three-term recurrence.
"""
function _legendre(n::Int, x::Real)
    @assert n >= 0 "Legendre polynomial degree must be non-negative, got $n"
    n == 0 && return one(float(x))
    n == 1 && return float(x)
    p_prev, p_curr = one(float(x)), float(x)
    for k in 1:(n - 1)
        p_next = ((2k + 1) * x * p_curr - k * p_prev) / (k + 1)
        p_prev, p_curr = p_curr, p_next
    end
    return p_curr
end

"""
    _normalized_legendre(n, x)

Evaluate the orthonormalized Legendre polynomial
``\\tilde{P}_n(x) = P_n(x) \\sqrt{(2n+1)/2}`` on ``[-1,1]``.
"""
function _normalized_legendre(n::Int, x::Real)
    return _legendre(n, x) * sqrt((2n + 1) / 2)
end

# ── Basis function enumeration ───────────────────────────────────────────────

"""
    _enumerate_basis_functions(max_delay, max_degree, max_total_degree, cross_terms)

Return a vector of basis function descriptors. Each descriptor is a
`Vector{Tuple{Int,Int}}` of `(delay, degree)` pairs.
"""
function _enumerate_basis_functions(
        max_delay::Int, max_degree::Int, max_total_degree::Int, cross_terms::Bool
    )
    basis = Vector{Vector{Tuple{Int, Int}}}()

    # Single-variable terms: P_d(u(t - τ))
    for delay in 1:max_delay
        for degree in 1:max_degree
            push!(basis, [(delay, degree)])
        end
    end

    # Two-variable cross-terms: P_{d1}(u(t-τ1)) * P_{d2}(u(t-τ2))
    if cross_terms
        for d1 in 1:max_delay
            for d2 in (d1 + 1):max_delay
                for deg1 in 1:max_degree
                    for deg2 in 1:max_degree
                        deg1 + deg2 <= max_total_degree || continue
                        push!(basis, [(d1, deg1), (d2, deg2)])
                    end
                end
            end
        end
    end

    return basis
end
