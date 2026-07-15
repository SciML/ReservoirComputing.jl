@doc raw"""
    correlation_matrix(states) -> Matrix

State correlation matrix ``R = X X^\top / L`` for a reservoir state collection
`states` of size ``N \times L`` (one state per column). This is the matrix whose
regularized-identity map defines a conceptor.

## Arguments

- `states::AbstractMatrix{<:Real}`: Reservoir states arranged as
  `(reservoir_dimension, sample_count)`.

## Returns

- A dense square matrix with the floating-point element type inferred from `states`.

## Throws

- `ArgumentError`: If `states` has no elements.

## Example

```julia
states = rand(Float32, 50, 1_000)
correlation = correlation_matrix(states)
```
"""
function correlation_matrix(states::AbstractMatrix{<:Real})
    isempty(states) && throw(ArgumentError("states must contain at least one state"))
    element_type = float(eltype(states))
    state_matrix = element_type.(states)
    return (state_matrix * state_matrix') / size(state_matrix, 2)
end

@doc raw"""
    conceptor_matrix(correlation, aperture) -> Matrix

Conceptor derived from a correlation matrix and the given `aperture`
[Jaeger2014conceptors](@cite). The result is symmetric positive semidefinite
with singular values in ``[0, 1)``.

## Arguments

- `correlation::AbstractMatrix{<:Real}`: Symmetric state correlation matrix.
- `aperture::Real`: Finite positive aperture. Larger values admit more state-space
  directions; smaller values suppress more directions.

## Returns

- A dense conceptor matrix with the floating-point element type of `correlation`.

## Throws

- `ArgumentError`: If `aperture` is not finite and positive, or if `correlation`
  is not symmetric.
- `DimensionMismatch`: If `correlation` is not square.

## Example

```julia
correlation = correlation_matrix(states)
conceptor = conceptor_matrix(correlation, 10)
```
"""
function conceptor_matrix(correlation::AbstractMatrix{<:Real}, aperture::Real)
    isfinite(aperture) && aperture > 0 ||
        throw(ArgumentError("aperture must be finite and positive, got $aperture"))
    checksquare(correlation)
    issymmetric(correlation) || throw(ArgumentError("correlation must be symmetric"))
    element_type = float(eltype(correlation))
    correlation_matrix = element_type.(correlation)
    typed_aperture = element_type(aperture)
    inverse_aperture_squared = inv(typed_aperture * typed_aperture)
    conceptor = if element_type <: Union{Float32, Float64}
        decomposition = eigen(Symmetric(correlation_matrix))
        values = max.(decomposition.values, zero(element_type))
        decomposition.vectors *
            Diagonal(values ./ (values .+ inverse_aperture_squared)) *
            decomposition.vectors'
    else
        correlation_matrix /
            Symmetric(correlation_matrix + inverse_aperture_squared * I)
    end
    return (conceptor + conceptor') / element_type(2)
end

"""
    conceptor_from_states(states, aperture) -> Matrix

Convenience composition of [`correlation_matrix`](@ref) and
[`conceptor_matrix`](@ref): the conceptor characterizing a reservoir state cloud
`states` (size `N × L`) at the given `aperture`.

## Arguments

- `states::AbstractMatrix{<:Real}`: Reservoir states, one state per column.
- `aperture::Real`: Finite positive aperture.

## Returns

- The conceptor associated with the sample correlation of `states`.

## Example

```julia
conceptor = conceptor_from_states(states, 10)
```
"""
function conceptor_from_states(states::AbstractMatrix{<:Real}, aperture::Real)
    return conceptor_matrix(correlation_matrix(states), aperture)
end

"""
    conceptor_singular_values(conceptor) -> Vector

Singular values of a (symmetric) conceptor matrix `C`, in descending order. For a
conceptor these coincide with its eigenvalues and lie in `[0, 1]`.

## Arguments

- `conceptor::AbstractMatrix{<:Real}`: Square conceptor matrix.

## Returns

- A vector of singular values in descending order.

## Throws

- `DimensionMismatch`: If `conceptor` is not square.
"""
function conceptor_singular_values(conceptor::AbstractMatrix{<:Real})
    checksquare(conceptor)
    return svdvals(float.(conceptor))
end

"""
    quota(conceptor) -> Real

Mean singular value of a conceptor, `tr(C) / N`. Jaeger's "quota" measures the
fraction of reservoir state space the conceptor leaves open (0 = point, 1 = all).

## Arguments

- `conceptor::AbstractMatrix{<:Real}`: Square conceptor matrix.

## Returns

- The mean singular value. Values near zero indicate a restrictive conceptor;
  values near one indicate a permissive conceptor.
"""
function quota(conceptor::AbstractMatrix{<:Real})
    checksquare(conceptor)
    return tr(conceptor) / size(conceptor, 1)
end

# ======================================================================
#  Aperture adaptation and aperture selection
# ======================================================================

@doc raw"""
    adapt_singular_value(singular_value, aperture_factor) -> Real

Aperture-adapted value for an input `singular_value` in `[0, 1]` and a
nonnegative `aperture_factor` [Jaeger2014conceptors](@cite).
The boundary values zero and one are fixed points.

## Arguments

- `singular_value::Real`: Value to adapt, normally in `[0, 1]`.
- `aperture_factor::Real`: Nonnegative factor applied to the aperture.

## Returns

- The adapted singular value, using the floating-point type inferred from
  `singular_value`.

## Throws

- `ArgumentError`: If `aperture_factor` is negative.
"""
function adapt_singular_value(singular_value::Real, aperture_factor::Real)
    aperture_factor >= 0 ||
        throw(ArgumentError("aperture_factor must be nonnegative, got $aperture_factor"))
    element_type = float(typeof(singular_value))
    typed_value = element_type(singular_value)
    typed_factor = element_type(aperture_factor)
    typed_value <= zero(element_type) && return zero(element_type)
    typed_value >= one(element_type) && return one(element_type)
    iszero(typed_factor) && return zero(element_type)
    isinf(typed_factor) && return one(element_type)
    return typed_value /
        (typed_value + (one(element_type) - typed_value) / typed_factor^2)
end

@doc raw"""
    aperture_adapt(conceptor, aperture_factor) -> Matrix

Aperture adaptation of a `conceptor` by a nonnegative `aperture_factor`
[Jaeger2014conceptors](@cite). This is equivalent to multiplying its aperture by
the supplied factor. Zero and infinite factors are handled exactly.

## Arguments

- `conceptor::AbstractMatrix{<:Real}`: Symmetric square conceptor matrix.
- `aperture_factor::Real`: Nonnegative aperture multiplier.

## Returns

- A conceptor with the same principal directions and adapted singular values.

## Throws

- `ArgumentError`: If the factor is negative or `conceptor` is not symmetric.
- `DimensionMismatch`: If `conceptor` is not square.

## Example

```julia
wider_conceptor = aperture_adapt(conceptor, 2)
projector = aperture_adapt(conceptor, Inf)
```
"""
function aperture_adapt(conceptor::AbstractMatrix{<:Real}, aperture_factor::Real)
    aperture_factor >= 0 ||
        throw(ArgumentError("aperture_factor must be nonnegative, got $aperture_factor"))
    checksquare(conceptor)
    issymmetric(conceptor) || throw(ArgumentError("conceptor must be symmetric"))
    element_type = float(eltype(conceptor))
    conceptor_matrix = element_type.(conceptor)
    typed_factor = element_type(aperture_factor)
    if isfinite(typed_factor) && !iszero(typed_factor)
        inverse_factor_squared = inv(typed_factor * typed_factor)
        adapted = conceptor_matrix /
            Symmetric(conceptor_matrix + inverse_factor_squared * (I - conceptor_matrix))
        return (adapted + adapted') / element_type(2)
    end
    decomposition = eigen(Symmetric(conceptor_matrix))
    adapted_values = adapt_singular_value.(
        clamp.(decomposition.values, zero(element_type), one(element_type)), typed_factor
    )
    return decomposition.vectors * Diagonal(adapted_values) * decomposition.vectors'
end

"""
    reaperture(conceptor, from_aperture, to_aperture) -> Matrix

Re-express a conceptor `C` that currently has aperture `from_aperture` so that it
has aperture `to_aperture` [Jaeger2014conceptors](@cite).

## Arguments

- `conceptor::AbstractMatrix{<:Real}`: Conceptor formed at `from_aperture`.
- `from_aperture::Real`: Finite positive current aperture.
- `to_aperture::Real`: Nonnegative target aperture; `Inf` is allowed.

## Returns

- The aperture-adapted conceptor.
"""
function reaperture(
        conceptor::AbstractMatrix{<:Real}, from_aperture::Real, to_aperture::Real
    )
    isfinite(from_aperture) && from_aperture > 0 ||
        throw(ArgumentError("from_aperture must be finite and positive"))
    to_aperture >= 0 || throw(ArgumentError("to_aperture must be nonnegative"))
    return aperture_adapt(conceptor, to_aperture / from_aperture)
end

@doc raw"""
    attenuation(conceptor, recurrent_weights, bias; steps, washout, init_state, rng) -> Real

Attenuation ``a_C = E[\|z(n) - x(n)\|^2] / E[\|z(n)\|^2]`` of a loaded reservoir
(recurrent weights `W`, bias `b`) run autonomously under conceptor `C`
[Jaeger2014conceptors](@cite), where ``z(n) = \tanh(W x(n-1) + b)`` is the unconstrained
update and ``x(n) = C z(n)`` is the conceptor-constrained state. The attenuation
is the fraction of reservoir signal energy suppressed by `C`; as a function of
aperture it passes through a minimum at the best-reconstructing aperture.

## Arguments

- `conceptor`: Square conceptor matching the recurrent reservoir dimension.
- `recurrent_weights`: Square autonomous recurrent weight matrix.
- `bias`: Reservoir bias vector.

## Keywords

- `steps::Int = 500`: Number of samples included in the energy ratio.
- `washout::Int = 200`: Initial autonomous steps excluded from the ratio.
- `init_state = nothing`: Optional initial reservoir state.
- `rng = Random.default_rng()`: Random number generator used when `init_state` is
  omitted.

## Returns

- The fraction of unconstrained reservoir energy suppressed by the conceptor.
"""
function attenuation(
        conceptor::AbstractMatrix{<:Real},
        recurrent_weights::AbstractMatrix{<:Real},
        bias::AbstractVector{<:Real};
        steps::Int = 500,
        washout::Int = 200,
        init_state::Union{Nothing, AbstractVector} = nothing,
        rng::AbstractRNG = Random.default_rng(),
    )
    checksquare(conceptor)
    checksquare(recurrent_weights)
    size(conceptor) == size(recurrent_weights) ||
        throw(DimensionMismatch("conceptor and recurrent_weights must have equal dimensions"))
    size(recurrent_weights, 1) == length(bias) ||
        throw(DimensionMismatch("bias length must equal the recurrent matrix dimension"))
    steps > 0 || throw(ArgumentError("steps must be positive"))
    washout >= 0 || throw(ArgumentError("washout must be nonnegative"))
    element_type = float(
        promote_type(eltype(conceptor), eltype(recurrent_weights), eltype(bias))
    )
    conceptor_matrix = element_type.(conceptor)
    recurrent_matrix = element_type.(recurrent_weights)
    bias_vector = element_type.(bias)
    reservoir_dimension = size(recurrent_matrix, 1)
    state = init_state === nothing ?
        element_type(0.5) .* randn(rng, element_type, reservoir_dimension) :
        element_type.(init_state)
    length(state) == reservoir_dimension ||
        throw(DimensionMismatch("init_state must have length $reservoir_dimension"))

    suppressed_energy = zero(element_type)
    total_energy = zero(element_type)
    for step in 1:(washout + steps)
        unconstrained_state = tanh.(recurrent_matrix * state .+ bias_vector)
        state = conceptor_matrix * unconstrained_state
        if step > washout
            suppressed_energy += sum(abs2, unconstrained_state .- state)
            total_energy += sum(abs2, unconstrained_state)
        end
    end
    iszero(total_energy) &&
        throw(ArgumentError("attenuation is undefined for a zero-energy rollout"))
    return suppressed_energy / total_energy
end

"""
    optimal_aperture(correlation, recurrent_weights, bias, apertures; kwargs...)

Select the aperture minimizing the [`attenuation`](@ref) criterion over a grid
`apertures`. For each candidate aperture, its conceptor is formed and the
loaded reservoir (`W`, `b`) is run autonomously to measure its attenuation. Returns
the minimizing aperture together with the full vector of attenuations (aligned with
`apertures`) so the characteristic trough can be inspected. `kwargs` are forwarded
to [`attenuation`](@ref).

## Arguments

- `correlation`: Symmetric state correlation matrix.
- `recurrent_weights`: Loaded recurrent weight matrix.
- `bias`: Reservoir bias vector.
- `apertures`: Nonempty vector of finite positive candidate apertures.

## Returns

- `(best_aperture, attenuations)`, where `attenuations` follows the order of
  `apertures`.
"""
function optimal_aperture(
        correlation::AbstractMatrix{<:Real},
        recurrent_weights::AbstractMatrix{<:Real},
        bias::AbstractVector{<:Real},
        apertures::AbstractVector{<:Real};
        kwargs...,
    )
    isempty(apertures) && throw(ArgumentError("apertures must not be empty"))
    attenuations = [
        attenuation(
                conceptor_matrix(correlation, aperture), recurrent_weights, bias; kwargs...
            ) for aperture in apertures
    ]
    return apertures[argmin(attenuations)], attenuations
end

# ======================================================================
#  Boolean operations on conceptors (Jaeger 2014, Section 3.9)
# ======================================================================

@doc raw"""
    conceptor_not(conceptor) -> Matrix

Logical negation of a conceptor [Jaeger2014conceptors](@cite).
Exchanges the roles of the directions the conceptor admits and suppresses.

## Arguments

- `conceptor::AbstractMatrix{<:Real}`: Square conceptor matrix.

## Returns

- The complementary conceptor. Its singular values are one minus those of
  `conceptor`.
"""
function conceptor_not(conceptor::AbstractMatrix{<:Real})
    checksquare(conceptor)
    element_type = float(eltype(conceptor))
    conceptor_matrix = element_type.(conceptor)
    return Matrix{element_type}(I, size(conceptor_matrix)) - conceptor_matrix
end

@doc raw"""
    conceptor_and(first_conceptor, second_conceptor) -> Matrix

Logical conjunction of two conceptors [Jaeger2014conceptors](@cite). For
full-rank conceptors this equals ``(C^{-1} + B^{-1} - I)^{-1}``; the implementation
uses the range-intersection projector form so that singular conceptors are handled
robustly. The result admits exactly the reservoir directions admitted by *both*
`C` and `B`.

## Arguments

- `first_conceptor::AbstractMatrix{<:Real}`: First symmetric conceptor.
- `second_conceptor::AbstractMatrix{<:Real}`: Second symmetric conceptor of the
  same size.

## Returns

- The conjunction of the two conceptors, including when either input is
  rank-deficient.

## Throws

- `DimensionMismatch`: If the conceptors are not square or have different sizes.
- `ArgumentError`: If either conceptor is not symmetric.
"""
function conceptor_and(
        first_conceptor::AbstractMatrix{<:Real},
        second_conceptor::AbstractMatrix{<:Real}
    )
    checksquare(first_conceptor)
    checksquare(second_conceptor)
    element_type = float(promote_type(eltype(first_conceptor), eltype(second_conceptor)))
    first_matrix = element_type.(first_conceptor)
    second_matrix = element_type.(second_conceptor)
    size(first_matrix) == size(second_matrix) ||
        throw(DimensionMismatch("conceptors must have equal size"))
    issymmetric(first_matrix) || throw(ArgumentError("first_conceptor must be symmetric"))
    issymmetric(second_matrix) || throw(ArgumentError("second_conceptor must be symmetric"))

    # Basis of R(C) ∩ R(B) = N(P_{N(C)} + P_{N(B)}), via the null-space projectors.
    first_nullspace = nullspace(first_matrix)
    second_nullspace = nullspace(second_matrix)
    nullspace_projector =
        first_nullspace * first_nullspace' + second_nullspace * second_nullspace'
    intersection_basis = nullspace(nullspace_projector)

    if size(intersection_basis, 2) == 0
        return zeros(element_type, size(first_matrix))
    end
    core = intersection_basis' * (pinv(first_matrix) + pinv(second_matrix) - I) *
        intersection_basis
    result = intersection_basis *
        (core \ Matrix{element_type}(I, size(core))) *
        intersection_basis'
    return (result + result') / element_type(2)
end

@doc raw"""
    conceptor_or(first_conceptor, second_conceptor) -> Matrix

Logical disjunction of two conceptors [Jaeger2014conceptors](@cite), defined
via De Morgan's law. The result admits
every reservoir direction admitted by `C` or by `B`.

## Arguments

- `first_conceptor::AbstractMatrix{<:Real}`: First symmetric conceptor.
- `second_conceptor::AbstractMatrix{<:Real}`: Second symmetric conceptor of the
  same size.

## Returns

- The disjunction of the two conceptors.
"""
function conceptor_or(
        first_conceptor::AbstractMatrix{<:Real},
        second_conceptor::AbstractMatrix{<:Real}
    )
    return conceptor_not(
        conceptor_and(conceptor_not(first_conceptor), conceptor_not(second_conceptor))
    )
end

# ======================================================================
#  The Conceptor reservoir-computer wrapper and its parameter/state plumbing
# ======================================================================

@doc raw"""
    Conceptor(model)

Wrap a ReservoirComputing `model` (e.g. an [`ESN`](@ref)) so that conceptor
matrices can be derived from it, stored by name, combined with the conceptor
algebra, and used to constrain autonomous generation [Jaeger2014conceptors](@cite).

The wrapped `model` provides the reservoir update; the `Conceptor` adds a
conceptor library to the model state. Parameters and states are nested under the
`model` field, leaving room for conceptor bookkeeping at the top level.

## Arguments

- `model`: Reservoir computer to wrap. Pattern loading expects an ESN-compatible
  parameter layout with reservoir, input, bias, and readout weights.

## Returns

- A conceptor-enabled reservoir computer used with [`initialparameters`](@ref),
  [`initialstates`](@ref), [`loadpatterns`](@ref), and [`generate`](@ref).

## Example

```julia
model = Conceptor(ESN(1, 100, 1; use_bias = true))
parameters = initialparameters(rng, model)
states = initialstates(rng, model)
```
"""
@concrete struct Conceptor <: AbstractReservoirComputer{(:model,)}
    model
end

function initialparameters(rng::AbstractRNG, concept::Conceptor)
    return (; model = initialparameters(rng, concept.model), readout = nothing)
end

function initialstates(rng::AbstractRNG, concept::Conceptor)
    return (;
        model = initialstates(rng, concept.model),
        conceptors = Dict{Symbol, Matrix}(),
        apertures = Dict{Symbol, Real}(),
        active_conceptor = nothing,
    )
end

"""
    has_conceptor(st, name) -> Bool

Whether a conceptor called `name` has been stored in the state `st`.

Returns `true` when `name` is present and `false` otherwise.
"""
has_conceptor(st::NamedTuple, name::Symbol) = haskey(st.conceptors, name)

"""
    get_conceptor(st, name) -> Matrix or nothing

The stored conceptor matrix called `name`, or `nothing` if it is absent.

Use [`has_conceptor`](@ref) when absence should be checked separately.
"""
get_conceptor(st::NamedTuple, name::Symbol) = get(st.conceptors, name, nothing)

"""
    store_conceptor(st, name, conceptor, aperture) -> st

Record conceptor matrix `C` (formed at `aperture`) under `name` in the state's
conceptor library. The input state is left untouched; an updated state is
returned.

## Arguments

- `st::NamedTuple`: State returned by `initialstates` for a [`Conceptor`](@ref).
- `name::Symbol`: Name used to retrieve the conceptor later.
- `conceptor::AbstractMatrix{<:Real}`: Square conceptor matrix.
- `aperture::Real`: Finite positive aperture associated with the matrix.

## Returns

- A state with the conceptor and aperture recorded in its library.
"""
function store_conceptor(
        st::NamedTuple,
        name::Symbol,
        conceptor::AbstractMatrix{<:Real},
        aperture::Real,
    )
    checksquare(conceptor)
    isfinite(aperture) && aperture > 0 ||
        throw(ArgumentError("aperture must be finite and positive, got $aperture"))
    conceptors = copy(st.conceptors)
    apertures = copy(st.apertures)
    conceptors[name] = Matrix(conceptor)
    apertures[name] = convert(float(eltype(conceptor)), aperture)
    return merge(st, (; conceptors, apertures))
end

"""
    set_active_conceptor(st, name) -> st

Mark the conceptor `name` (or `nothing`) as the active one, returning an updated
state. The active conceptor is the default constraint for generation.

Passing `nothing` clears the active selection. A `KeyError` is thrown when a
nonexistent name is selected.
"""
function set_active_conceptor(st::NamedTuple, name::Union{Symbol, Nothing})
    name !== nothing && !has_conceptor(st, name) && throw(KeyError(name))
    return merge(st, (; active_conceptor = name))
end

"""
    active_conceptor(st) -> Matrix

The currently active conceptor matrix. Throws if none has been set.

Use [`set_active_conceptor`](@ref) to select or clear the active conceptor.
"""
function active_conceptor(st::NamedTuple)
    name = st.active_conceptor
    name === nothing &&
        throw(ArgumentError("no active conceptor; call set_active_conceptor"))
    return st.conceptors[name]
end

# Resolve a conceptor argument that is either an explicit matrix or a stored name.
resolve_conceptor(::NamedTuple, conceptor::AbstractMatrix{<:Real}) = conceptor
function resolve_conceptor(st::NamedTuple, conceptor::Symbol)
    stored_conceptor = get_conceptor(st, conceptor)
    stored_conceptor === nothing && throw(KeyError(conceptor))
    return stored_conceptor
end

"""
    collectstates(concept::Conceptor, signal, ps, st; conceptor=nothing) -> (states, st)

Stream `signal` through the wrapped reservoir and collect its state sequence. When
`conceptor` (a stored `Symbol` or an explicit matrix) is given, the collected
states are filtered through it, `C x(n)`, realizing the conceptor as a state
modifier.
"""
function collectstates(
        concept::Conceptor,
        signal::AbstractMatrix,
        ps::NamedTuple,
        st::NamedTuple;
        conceptor::Union{Symbol, AbstractMatrix, Nothing} = nothing,
    )
    states, st_model = collectstates(concept.model, signal, ps.model, st.model)
    new_st = merge(st, (; model = st_model))
    conceptor === nothing && return states, new_st
    return resolve_conceptor(new_st, conceptor) * states, new_st
end

"""
    addreadout!(concept::Conceptor, weights, ps, st) -> (ps, st)

Install trained readout weights `W` into the wrapped model's readout layer
(`ps.model.readout.weight`), keeping the generation and classification paths on a
single readout convention.
"""
function addreadout!(
        concept::Conceptor, weights::AbstractMatrix, ps::NamedTuple, st::NamedTuple
    )
    return set_readout_weight(ps, weights), st
end

reservoir_params(ps::NamedTuple) = ps.model.reservoir
readout_params(ps::NamedTuple) = ps.model.readout

function reservoir_bias(ps::NamedTuple)
    rps = reservoir_params(ps)
    element_type = eltype(rps.reservoir_matrix)
    return haskey(rps, :bias) ?
        vec(rps.bias) : zeros(element_type, size(rps.reservoir_matrix, 1))
end

function set_reservoir_matrix(ps::NamedTuple, weights::AbstractMatrix)
    reservoir = merge(ps.model.reservoir, (; reservoir_matrix = weights))
    return merge(ps, (; model = merge(ps.model, (; reservoir = reservoir))))
end

function set_readout_weight(ps::NamedTuple, weights::AbstractMatrix)
    readout = merge(ps.model.readout, (; weight = weights))
    return merge(ps, (; model = merge(ps.model, (; readout = readout))))
end

# Tikhonov-regularized linear map M minimizing |M features - targets|^2 + reg |M|^2.
# `features` is n_in × T, `targets` is n_out × T; returns M of size n_out × n_in.
function ridge_map(
        features::AbstractMatrix{<:Real},
        targets::AbstractMatrix{<:Real},
        reg::Real,
    )
    element_type = float(promote_type(eltype(features), eltype(targets)))
    feature_matrix = element_type.(features)
    target_matrix = element_type.(targets)
    return train(StandardRidge(element_type, reg), feature_matrix, target_matrix)
end

# Per-pattern aperture lookup: a scalar applies to every pattern; a dictionary
# supplies a value per name.
aperture_for(aperture::Real, ::Symbol) = aperture
function aperture_for(aperture::AbstractDict{Symbol, <:Real}, name::Symbol)
    haskey(aperture, name) || throw(ArgumentError("no aperture provided for :$name"))
    return aperture[name]
end

# loading patterns

function _drive_pattern(
        rng::AbstractRNG,
        concept::Conceptor,
        signal::AbstractMatrix,
        ps::NamedTuple,
        st::NamedTuple,
        washout::Int,
    )
    washout >= 1 ||
        throw(ArgumentError("washout must be at least 1, got $washout: fitting the \
                             recurrent weights pairs each state with its predecessor"))
    st_reset = resetcarry!(rng, concept.model, st.model; init_carry = nothing)
    states, st_model = collectstates(concept.model, signal, ps.model, st_reset)
    signal_length = size(states, 2)
    washout < signal_length ||
        throw(ArgumentError("washout=$washout exceeds signal length=$signal_length"))
    current_states = states[:, (washout + 1):signal_length]
    lagged_states = states[:, washout:(signal_length - 1)]
    aligned_signal = signal[:, (washout + 1):signal_length]
    return current_states, lagged_states, aligned_signal, merge(st, (; model = st_model))
end

@doc raw"""
    loadpatterns(rng, concept, named_signals, ps, st;
          aperture=10.0, washout=500, reg_recurrent=1e-4, reg_readout=1e-2)

Load named driving patterns into the reservoir wrapped by `concept`
[Jaeger2014conceptors](@cite). For each `name => signal` pair the reservoir is run
from a cleared carry, the first `washout` states are discarded, and the remaining
states are used to

  * derive and store a conceptor from the state correlation matrix, and
  * accumulate data for a single shared input-internalizing recurrent matrix `W`
    and readout `W_out`.

`W` is fitted (ridge, `reg_recurrent`) so that
``W x(n-1) \approx W^* x(n-1) + W_\text{in} p(n)``, absorbing the input drive into
the recurrent weights; `W_out` is fitted (ridge, `reg_readout`) so that
``W_\text{out} x(n) \approx p(n)`` across all patterns. `aperture` is either a
scalar or a dictionary mapping each name to an aperture.

Returns `(ps, st)` with `reservoir_matrix` and `readout.weight` replaced and the
conceptor library populated.

## Arguments

- `rng::AbstractRNG`: Random number generator used when reservoir carries reset.
- `concept::Conceptor`: Conceptor-wrapped reservoir.
- `named_signals`: Iterable of `Symbol => signal` pairs. A vector signal is treated
  as one-dimensional; a matrix has one input channel per row and time per column.
- `ps::NamedTuple`: Current parameters.
- `st::NamedTuple`: Current states.

## Keywords

- `aperture = 10.0`: One aperture for every pattern, or a dictionary keyed by
  pattern name.
- `washout::Int = 500`: Initial samples excluded from fitting; at least 1 and
  shorter than every signal.
- `reg_recurrent::Real = 1.0e-4`: Ridge penalty for recurrent weights.
- `reg_readout::Real = 1.0e-2`: Ridge penalty for readout weights.

## Returns

- `(parameters, states)`: Updated parameters and a state containing one conceptor
  per named signal.

## Throws

- `ArgumentError`: If a name is not a `Symbol`, an aperture is missing, or washout
  is not in `1:(signal length - 1)`.
"""
function loadpatterns(
        rng::AbstractRNG,
        concept::Conceptor,
        named_signals,
        ps::NamedTuple,
        st::NamedTuple;
        aperture = 10.0,
        washout::Int = 500,
        reg_recurrent::Real = 1.0e-4,
        reg_readout::Real = 1.0e-2,
    )
    rps = reservoir_params(ps)
    element_type = float(eltype(rps.reservoir_matrix))
    original_recurrent_weights = element_type.(rps.reservoir_matrix)
    input_weights = element_type.(rps.input_matrix)

    state_sets = Matrix{element_type}[]
    lagged_state_sets = Matrix{element_type}[]
    signal_sets = Matrix{element_type}[]
    st_acc = st

    for (name, signal) in named_signals
        name isa Symbol ||
            throw(ArgumentError("signal name must be a Symbol, got $(typeof(name))"))
        signal_matrix = signal isa AbstractMatrix ?
            element_type.(signal) : reshape(element_type.(signal), 1, :)
        current_states, lagged_states, aligned_signal, st_acc =
            _drive_pattern(rng, concept, signal_matrix, ps, st_acc, washout)
        pattern_aperture = aperture_for(aperture, name)
        st_acc = store_conceptor(
            st_acc, name, conceptor_from_states(current_states, pattern_aperture),
            pattern_aperture
        )
        push!(state_sets, current_states)
        push!(lagged_state_sets, lagged_states)
        push!(signal_sets, aligned_signal)
    end

    all_states = reduce(hcat, state_sets)
    all_lagged_states = reduce(hcat, lagged_state_sets)
    all_signals = reduce(hcat, signal_sets)

    readout_weights = ridge_map(all_states, all_signals, reg_readout)
    recurrent_target =
        original_recurrent_weights * all_lagged_states .+ input_weights * all_signals
    recurrent_weights = ridge_map(all_lagged_states, recurrent_target, reg_recurrent)

    updated_parameters = set_reservoir_matrix(ps, element_type.(recurrent_weights))
    updated_parameters =
        set_readout_weight(updated_parameters, element_type.(readout_weights))
    return updated_parameters, st_acc
end

# ======================================================================
#  Autonomous pattern generation (Jaeger 2014, Section 3.4)
# ======================================================================

@doc raw"""
    generate(concept, ps, st; conceptor, steps, washout=200,
             init_state=nothing, rng=Random.default_rng()) -> (Y, X)

Run the loaded reservoir autonomously under `conceptor` and read out the observer
signal:

```math
x(n) = C \tanh(W x(n-1) + b), \qquad y(n) = W_\text{out} x(n).
```

`conceptor` is either a stored conceptor `Symbol` or an explicit conceptor matrix
(e.g. the output of [`morph_conceptor`](@ref) or the Boolean operations). The first
`washout` steps let the autonomous orbit settle and are discarded. Returns `(Y, X)`:
the post-washout observer outputs (`out_dims × steps`) and reservoir states
(`res_dims × steps`). The rollout uses the element type of the reservoir parameters.

## Arguments

- `concept::Conceptor`: Loaded conceptor model.
- `ps::NamedTuple`: Parameters returned by [`loadpatterns`](@ref).
- `st::NamedTuple`: State containing the requested conceptor.

## Keywords

- `conceptor`: Stored conceptor name or an explicit square conceptor matrix.
- `steps::Int`: Number of returned time steps; must be positive.
- `washout::Int = 200`: Autonomous steps discarded before recording.
- `init_state = nothing`: Optional reservoir state vector. A random state is used
  when omitted.
- `rng = Random.default_rng()`: Random number generator for the initial state.

## Returns

- `(outputs, states)`: Observer outputs and reservoir states, with time along
  columns.

## Throws

- `KeyError`: If a requested conceptor name is absent.
- `DimensionMismatch`: If the conceptor or initial state does not match the
  reservoir dimension.
- `ArgumentError`: If `steps` is not positive or `washout` is negative.

## Example

```julia
outputs, states = generate(
    concept, parameters, model_state;
    conceptor = :sine, steps = 500, washout = 100, rng
)
```
"""
function generate(
        concept::Conceptor,
        ps::NamedTuple,
        st::NamedTuple;
        conceptor::Union{Symbol, AbstractMatrix},
        steps::Int,
        washout::Int = 200,
        init_state::Union{Nothing, AbstractVector} = nothing,
        rng::AbstractRNG = Random.default_rng(),
    )
    rps = reservoir_params(ps)
    element_type = float(eltype(rps.reservoir_matrix))
    recurrent_weights = element_type.(rps.reservoir_matrix)
    bias = reservoir_bias(ps)
    readout_weights = element_type.(readout_params(ps).weight)
    conceptor_matrix = element_type.(resolve_conceptor(st, conceptor))
    reservoir_dimension = size(recurrent_weights, 1)
    checksquare(conceptor_matrix)
    size(conceptor_matrix, 1) == reservoir_dimension ||
        throw(
        DimensionMismatch(
            "conceptor size must match the reservoir dimension $reservoir_dimension"
        )
    )
    steps > 0 || throw(ArgumentError("steps must be positive"))
    washout >= 0 || throw(ArgumentError("washout must be nonnegative"))

    state = init_state === nothing ?
        element_type(0.5) .* randn(rng, element_type, reservoir_dimension) :
        element_type.(init_state)
    length(state) == reservoir_dimension ||
        throw(DimensionMismatch("init_state must have length $reservoir_dimension"))
    output_dimension = size(readout_weights, 1)
    outputs = zeros(element_type, output_dimension, steps)
    states = zeros(element_type, reservoir_dimension, steps)

    for step in 1:(washout + steps)
        state = conceptor_matrix * tanh.(recurrent_weights * state .+ bias)
        if step > washout
            output_index = step - washout
            @views states[:, output_index] = state
            @views outputs[:, output_index] = readout_weights * state
        end
    end
    return outputs, states
end

# morphing

@doc raw"""
    morph_conceptor(st, weights) -> Matrix

Linearly combine stored conceptors into a morphed conceptor
[Jaeger2014conceptors](@cite). `weights` is an iterable of name-to-weight pairs, a `NamedTuple`,
or a `Dict{Symbol,<:Real}`. Coefficients summing to one
interpolate between the named prototypes; coefficients outside `[0, 1]` extrapolate.

## Arguments

- `st::NamedTuple`: State containing stored conceptors.
- `weights`: Named tuple, dictionary, or iterable of name-to-weight pairs.

## Returns

- The weighted sum of the named conceptors.

## Throws

- `KeyError`: If a named conceptor is absent.
- `ArgumentError`: If `weights` is empty.
"""
function morph_conceptor(st::NamedTuple, weights)
    # `pairs` on a plain iterable of Pairs would enumerate index => pair.
    weight_pairs = weights isa Union{NamedTuple, AbstractDict} ? pairs(weights) : weights
    morphed_conceptor = nothing
    for (name, weight) in weight_pairs
        stored_conceptor = get_conceptor(st, Symbol(name))
        stored_conceptor === nothing && throw(KeyError(Symbol(name)))
        contribution = convert(eltype(stored_conceptor), weight) .* stored_conceptor
        morphed_conceptor = morphed_conceptor === nothing ?
            contribution : morphed_conceptor .+ contribution
    end
    morphed_conceptor === nothing && throw(ArgumentError("no weights provided"))
    return morphed_conceptor
end

# ======================================================================
#  Conceptors as state filters for supervised readout training
# ======================================================================

# Normalize a target into a row-major matrix (1 × T for a vector target).
_as_matrix(target::AbstractMatrix) = target
_as_matrix(target::AbstractVector) = reshape(target, 1, :)

"""
    store_conceptors(rng, concept, named_signals, ps, st;
                      aperture=1.0, init_carry=nothing) -> st

Derive and store one conceptor per named signal without modifying the reservoir
weights. For each `name => signal` the reservoir is run from a cleared carry and
the conceptor of its state cloud is stored. Use
this to build a conceptor library for classification, where each class pattern
gets its own conceptor. `aperture` is a scalar or a dictionary keyed by name.

## Returns

- The updated state. Reservoir parameters are unchanged.

## See also

[`loadpatterns`](@ref), [`train!`](@ref), [`store_conceptor`](@ref)
"""
function store_conceptors(
        rng::AbstractRNG,
        concept::Conceptor,
        named_signals,
        ps::NamedTuple,
        st::NamedTuple;
        aperture = 1.0,
        init_carry = nothing,
    )
    st_acc = st
    for (name, signal) in named_signals
        name isa Symbol ||
            throw(ArgumentError("signal name must be a Symbol, got $(typeof(name))"))
        sig = signal isa AbstractMatrix ? signal : reshape(signal, 1, :)
        st_reset = resetcarry!(rng, concept.model, st_acc.model; init_carry = init_carry)
        states, st_model = collectstates(concept.model, sig, ps.model, st_reset)
        st_acc = merge(st_acc, (; model = st_model))
        pattern_aperture = aperture_for(aperture, name)
        st_acc = store_conceptor(
            st_acc, name, conceptor_from_states(states, pattern_aperture), pattern_aperture
        )
    end
    return st_acc
end

"""
    train!(rng, concept::Conceptor, named_signals, named_targets, ps, st,
           train_method=StandardRidge(0.0);
           washout=0, return_states=false, init_carry=nothing) -> (ps, st)

Train a single readout on conceptor-filtered reservoir features. For each
`name => signal` the states are collected through the conceptor named `name`
(stored beforehand via [`store_conceptors`](@ref)), the first `washout` columns
are dropped, and features and the matching `named_targets` are pooled across
patterns before the readout is fit with `train_method`. With `return_states=true`
the pooled feature matrix is also returned.

## Arguments

- `named_signals`: Iterable of named input sequences.
- `named_targets`: Iterable containing one target sequence for every signal name.
- `train_method`: Readout fitting method; defaults to unregularized
  [`StandardRidge`](@ref).

## Keywords

- `washout::Int = 0`: Initial columns removed from every feature and target.
- `return_states::Bool = false`: Return pooled conceptor-filtered features.
- `init_carry = nothing`: Initial carry passed when resetting each sequence.
- `kwargs...`: Additional options forwarded to `train`.

## Returns

- `(parameters, states)` by default.
- `((parameters, states), features)` when `return_states = true`.
"""
function train!(
        rng::AbstractRNG,
        concept::Conceptor,
        named_signals,
        named_targets,
        ps::NamedTuple,
        st::NamedTuple,
        train_method = StandardRidge(0.0);
        washout::Int = 0,
        return_states::Bool = false,
        init_carry = nothing,
        kwargs...,
    )
    targets = Dict{Symbol, AbstractMatrix}(
        Symbol(name) => _as_matrix(target) for (name, target) in named_targets
    )

    all_states = AbstractMatrix[]
    all_targets = AbstractMatrix[]
    for (name, signal) in named_signals
        name isa Symbol ||
            throw(ArgumentError("signal name must be a Symbol, got $(typeof(name))"))
        haskey(targets, name) || throw(ArgumentError("no target data for pattern :$name"))
        st_reset = resetcarry!(rng, concept.model, st.model; init_carry = init_carry)
        st_tmp = merge(st, (; model = st_reset))
        states, _ = collectstates(concept, signal, ps, st_tmp; conceptor = name)
        tgt = targets[name]
        if washout > 0
            states = states[:, (washout + 1):end]
            tgt = tgt[:, (washout + 1):end]
        end
        push!(all_states, states)
        push!(all_targets, tgt)
    end

    features = reduce(hcat, all_states)
    pooled_targets = reduce(hcat, all_targets)
    W_out = train(train_method, features, pooled_targets; kwargs...)
    ps2, st2 = addreadout!(concept, W_out, ps, st)
    return return_states ? ((ps2, st2), features) : (ps2, st2)
end
