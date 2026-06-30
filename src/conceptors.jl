# Conceptors for recurrent neural networks, after Jaeger (2014), "Controlling
# Recurrent Neural Networks by Conceptors" (arXiv:1403.3369).
#
# A conceptor is a positive semidefinite matrix C = R (R + α^{-2} I)^{-1} derived
# from the correlation matrix R of a reservoir's driven states. It acts as a soft
# projector onto the linear subspace a driving pattern excites in the reservoir.
# Conceptors can be aperture-adapted, combined with a Boolean algebra, loaded into
# a reservoir to autonomously regenerate the loaded patterns, and morphed to
# interpolate or extrapolate between patterns.
#
# Conceptor matrices are always computed and stored in `Float64`: the defining
# operation is a matrix inversion whose conditioning degrades quickly for the
# near-singular correlation matrices produced by periodic drivers, so conceptor
# precision is decoupled from the (often `Float32`) reservoir precision.

const ConceptorMatrix = Matrix{Float64}

# ======================================================================
#  Core conceptor matrices
# ======================================================================

@doc raw"""
    correlation_matrix(states) -> Matrix{Float64}

State correlation matrix ``R = X X^\top / L`` for a reservoir state collection
`states` of size ``N \times L`` (one state per column). This is the matrix whose
regularized-identity map defines a conceptor.
"""
function correlation_matrix(states::AbstractMatrix{<:Real})
    X = Float64.(states)
    return (X * X') / size(X, 2)
end

@doc raw"""
    conceptor_matrix(R, aperture) -> Matrix{Float64}

Conceptor ``C = R (R + \alpha^{-2} I)^{-1}`` derived from a correlation matrix `R`
with aperture ``\alpha`` = `aperture` (Jaeger 2014, Eq. 7). The result is
symmetric positive semidefinite with singular values in ``[0, 1)``.
"""
function conceptor_matrix(R::AbstractMatrix{<:Real}, aperture::Real)
    aperture > 0 || throw(ArgumentError("aperture must be positive, got $aperture"))
    Rd = Float64.(R)
    inv_sq = 1.0 / aperture^2
    return Rd / (Rd + inv_sq * I)
end

"""
    conceptor_from_states(states, aperture) -> Matrix{Float64}

Convenience composition of [`correlation_matrix`](@ref) and
[`conceptor_matrix`](@ref): the conceptor characterizing a reservoir state cloud
`states` (size `N × L`) at the given `aperture`.
"""
function conceptor_from_states(states::AbstractMatrix{<:Real}, aperture::Real)
    return conceptor_matrix(correlation_matrix(states), aperture)
end

"""
    conceptor_singular_values(C) -> Vector{Float64}

Singular values of a (symmetric) conceptor matrix `C`, in descending order. For a
conceptor these coincide with its eigenvalues and lie in `[0, 1]`.
"""
function conceptor_singular_values(C::AbstractMatrix{<:Real})
    return svdvals(Float64.(C))
end

"""
    quota(C) -> Float64

Mean singular value of a conceptor, `tr(C) / N`. Jaeger's "quota" measures the
fraction of reservoir state space the conceptor leaves open (0 = point, 1 = all).
"""
function quota(C::AbstractMatrix{<:Real})
    return tr(Float64.(C)) / size(C, 1)
end

# ======================================================================
#  Aperture adaptation and aperture selection
# ======================================================================

@doc raw"""
    adapt_singular_value(s, γ) -> Float64

Aperture-adapted singular value ``s_\gamma`` for an input singular value
``s \in [0, 1]`` and adaptation factor ``\gamma \in [0, \infty]`` (Jaeger 2014,
Prop. 3, Eq. 19). Hard values `s = 0` and `s = 1` are fixed points; intermediate
values are pushed toward 0 as ``\gamma \to 0`` and toward 1 as ``\gamma \to \infty``.
"""
function adapt_singular_value(s::Float64, γ::Float64)
    (s <= 0.0) && return 0.0
    (s >= 1.0) && return 1.0
    iszero(γ) && return 0.0
    isinf(γ) && return 1.0
    return s / (s + (1.0 - s) / γ^2)
end

@doc raw"""
    aperture_adapt(C, γ) -> Matrix{Float64}

Aperture adaptation ``\varphi(C, \gamma)`` of a conceptor `C` by factor
``\gamma \in [0, \infty]`` (Jaeger 2014, Def. 3). Equivalent to multiplying the
aperture of `C` by ``\gamma``; in particular ``C(R, \alpha) = \varphi(C(R, 1), \alpha)``.
Applied via the eigendecomposition so the limiting cases ``\gamma = 0`` (hard
zero) and ``\gamma = \infty`` (hardening to a projector) are exact.
"""
function aperture_adapt(C::AbstractMatrix{<:Real}, γ::Real)
    γ >= 0 || throw(ArgumentError("aperture factor γ must be ≥ 0, got $γ"))
    F = eigen(Symmetric(Float64.(C)))
    s_adapted = adapt_singular_value.(clamp.(F.values, 0.0, 1.0), Float64(γ))
    return F.vectors * Diagonal(s_adapted) * F.vectors'
end

"""
    reaperture(C, from_aperture, to_aperture) -> Matrix{Float64}

Re-express a conceptor `C` that currently has aperture `from_aperture` so that it
has aperture `to_aperture`, using `φ(C, to/from)` (Jaeger 2014, Prop. 5).
"""
function reaperture(C::AbstractMatrix{<:Real}, from_aperture::Real, to_aperture::Real)
    from_aperture > 0 || throw(ArgumentError("from_aperture must be positive"))
    return aperture_adapt(C, to_aperture / from_aperture)
end

@doc raw"""
    attenuation(C, W, b; steps, washout, init_state, rng) -> Float64

Attenuation ``a_C = E[\|z(n) - x(n)\|^2] / E[\|z(n)\|^2]`` of a loaded reservoir
(recurrent weights `W`, bias `b`) run autonomously under conceptor `C`
(Jaeger 2014, Eq. 23), where ``z(n) = \tanh(W x(n-1) + b)`` is the unconstrained
update and ``x(n) = C z(n)`` is the conceptor-constrained state. The attenuation
is the fraction of reservoir signal energy suppressed by `C`; as a function of
aperture it passes through a minimum at the best-reconstructing aperture.
"""
function attenuation(
        C::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, b::AbstractVector{<:Real};
        steps::Int = 500, washout::Int = 200,
        init_state::Union{Nothing, AbstractVector} = nothing,
        rng::AbstractRNG = Random.default_rng()
    )
    Cd = Float64.(C)
    Wd = Float64.(W)
    bd = Float64.(b)
    N = size(Wd, 1)
    x = init_state === nothing ? 0.5 .* randn(rng, N) : Float64.(init_state)

    num = 0.0
    den = 0.0
    for n in 1:(washout + steps)
        z = tanh.(Wd * x .+ bd)
        x = Cd * z
        if n > washout
            num += sum(abs2, z .- x)
            den += sum(abs2, z)
        end
    end
    return num / den
end

"""
    optimal_aperture(R, W, b, apertures; kwargs...) -> (best_aperture, attenuations)

Select the aperture minimizing the [`attenuation`](@ref) criterion over a grid
`apertures`. For each candidate `α` the conceptor `C(R, α)` is formed and the
loaded reservoir (`W`, `b`) is run autonomously to measure its attenuation. Returns
the minimizing aperture together with the full vector of attenuations (aligned with
`apertures`) so the characteristic trough can be inspected. `kwargs` are forwarded
to [`attenuation`](@ref).
"""
function optimal_aperture(
        R::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, b::AbstractVector{<:Real},
        apertures::AbstractVector{<:Real}; kwargs...
    )
    atts = [attenuation(conceptor_matrix(R, α), W, b; kwargs...) for α in apertures]
    return apertures[argmin(atts)], atts
end

# ======================================================================
#  Boolean operations on conceptors (Jaeger 2014, Section 3.9)
# ======================================================================

@doc raw"""
    conceptor_not(C) -> Matrix{Float64}

Logical negation ``\neg C = I - C`` of a conceptor (Jaeger 2014, Eq. 28).
Exchanges the roles of the directions the conceptor admits and suppresses.
"""
function conceptor_not(C::AbstractMatrix{<:Real})
    Cd = Float64.(C)
    return Matrix(I, size(Cd)) - Cd
end

# Orthonormal basis of the range of a symmetric PSD matrix, using a tolerance on
# the singular values to decide the numerical rank.
function _range_basis(C::Matrix{Float64}; tol::Float64 = 1.0e-12)
    F = svd(C)
    rank = count(>(tol), F.S)
    return F.U[:, 1:rank]
end

@doc raw"""
    conceptor_and(C, B) -> Matrix{Float64}

Logical conjunction ``C \wedge B`` of two conceptors (Jaeger 2014, Eq. 32). For
full-rank conceptors this equals ``(C^{-1} + B^{-1} - I)^{-1}``; the implementation
uses the range-intersection projector form so that singular conceptors are handled
robustly. The result admits exactly the reservoir directions admitted by *both*
`C` and `B`.
"""
function conceptor_and(C::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
    Cd = Float64.(C)
    Bd = Float64.(B)
    size(Cd) == size(Bd) || throw(DimensionMismatch("conceptors must have equal size"))

    # Basis of R(C) ∩ R(B) = N(P_{N(C)} + P_{N(B)}), via the null-space projectors.
    UC0 = nullspace(Cd)
    UB0 = nullspace(Bd)
    M = UC0 * UC0' + UB0 * UB0'
    Fm = svd(M)
    rank_M = count(>(1.0e-12), Fm.S)
    Bgk = Fm.U[:, (rank_M + 1):end]            # basis of the range intersection

    if size(Bgk, 2) == 0
        return zeros(size(Cd))                  # disjoint supports → all-zero conceptor
    end
    core = Bgk' * (pinv(Cd) + pinv(Bd) - I) * Bgk
    return Bgk * (core \ Matrix(I, size(core))) * Bgk'
end

@doc raw"""
    conceptor_or(C, B) -> Matrix{Float64}

Logical disjunction ``C \vee B`` of two conceptors (Jaeger 2014, Eq. 31), defined
via De Morgan's law ``C \vee B = \neg(\neg C \wedge \neg B)``. The result admits
every reservoir direction admitted by `C` or by `B`.
"""
function conceptor_or(C::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
    return conceptor_not(conceptor_and(conceptor_not(C), conceptor_not(B)))
end

# Infix sugar mirroring the paper's ¬ / ∧ / ∨ notation. `∧` and `∨` are Julia infix
# operators; `¬` is a prefix-style function (`¬(C)`). Not exported, to avoid
# polluting the package namespace; use the named functions or `ReservoirComputing.:∧`.
const ¬ = conceptor_not
∧(C::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real}) = conceptor_and(C, B)
∨(C::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real}) = conceptor_or(C, B)

# ======================================================================
#  The Conceptor reservoir-computer wrapper and its parameter/state plumbing
# ======================================================================

@doc raw"""
    Conceptor(model)

Wrap a ReservoirComputing `model` (e.g. an [`ESN`](@ref)) so that conceptor
matrices can be derived from it, stored by name, combined with the conceptor
algebra, and used to constrain autonomous generation (Jaeger 2014).

The wrapped `model` provides the reservoir update; the `Conceptor` adds a
conceptor library to the model state. Parameters and states are nested under the
`model` field, leaving room for conceptor bookkeeping at the top level.
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
        conceptors = Dict{Symbol, ConceptorMatrix}(),
        apertures = Dict{Symbol, Float64}(),
        active_conceptor = nothing,
    )
end

"""
    has_conceptor(st, name) -> Bool

Whether a conceptor called `name` has been stored in the state `st`.
"""
has_conceptor(st::NamedTuple, name::Symbol) = haskey(st.conceptors, name)

"""
    get_conceptor(st, name) -> Matrix{Float64} or nothing

The stored conceptor matrix called `name`, or `nothing` if it is absent.
"""
get_conceptor(st::NamedTuple, name::Symbol) = get(st.conceptors, name, nothing)

"""
    store_conceptor!(st, name, C, aperture) -> st

Record conceptor matrix `C` (formed at `aperture`) under `name` in the state's
conceptor library. Mutates the library dictionaries in place and returns `st`.
"""
function store_conceptor!(st::NamedTuple, name::Symbol, C::AbstractMatrix{<:Real}, aperture::Real)
    st.conceptors[name] = Float64.(C)
    st.apertures[name] = Float64(aperture)
    return st
end

"""
    set_active_conceptor(st, name) -> st

Mark the conceptor `name` (or `nothing`) as the active one, returning an updated
state. The active conceptor is the default constraint for generation.
"""
function set_active_conceptor(st::NamedTuple, name::Union{Symbol, Nothing})
    name !== nothing && !has_conceptor(st, name) && throw(KeyError(name))
    return merge(st, (; active_conceptor = name))
end

"""
    active_conceptor(st) -> Matrix{Float64}

The currently active conceptor matrix. Throws if none has been set.
"""
function active_conceptor(st::NamedTuple)
    name = st.active_conceptor
    name === nothing && throw(ArgumentError("no active conceptor; call set_active_conceptor"))
    return st.conceptors[name]
end

# Resolve a conceptor argument that is either an explicit matrix or a stored name.
resolve_conceptor(st::NamedTuple, conceptor::AbstractMatrix{<:Real}) = Float64.(conceptor)
function resolve_conceptor(st::NamedTuple, conceptor::Symbol)
    C = get_conceptor(st, conceptor)
    C === nothing && throw(KeyError(conceptor))
    return C
end

"""
    collectstates(concept::Conceptor, signal, ps, st; conceptor=nothing) -> (states, st)

Stream `signal` through the wrapped reservoir and collect its state sequence. When
`conceptor` (a stored `Symbol` or an explicit matrix) is given, the collected
states are filtered through it, `C x(n)`, realizing the conceptor as a state
modifier.
"""
function collectstates(
        concept::Conceptor, signal::AbstractMatrix, ps::NamedTuple, st::NamedTuple;
        conceptor::Union{Symbol, AbstractMatrix, Nothing} = nothing
    )
    states, st_model = collectstates(concept.model, signal, ps.model, st.model)
    new_st = merge(st, (; model = st_model))
    conceptor === nothing && return states, new_st
    return resolve_conceptor(new_st, conceptor) * states, new_st
end

"""
    addreadout!(concept::Conceptor, W, ps, st) -> (ps, st)

Install trained readout weights `W` into the wrapped model's readout layer
(`ps.model.readout.weight`), keeping the generation and classification paths on a
single readout convention.
"""
function addreadout!(concept::Conceptor, W::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    return set_readout_weight(ps, W), st
end

# ======================================================================
#  Parameter-tree access for the wrapped model (ps.model.{reservoir,readout})
# ======================================================================

reservoir_params(ps::NamedTuple) = ps.model.reservoir
readout_params(ps::NamedTuple) = ps.model.readout

function reservoir_bias(ps::NamedTuple)
    rps = reservoir_params(ps)
    return haskey(rps, :bias) ? Float64.(vec(rps.bias)) : zeros(Float64, size(rps.reservoir_matrix, 1))
end

function set_reservoir_matrix(ps::NamedTuple, W::AbstractMatrix)
    reservoir = merge(ps.model.reservoir, (; reservoir_matrix = W))
    return merge(ps, (; model = merge(ps.model, (; reservoir = reservoir))))
end

function set_readout_weight(ps::NamedTuple, W::AbstractMatrix)
    readout = merge(ps.model.readout, (; weight = W))
    return merge(ps, (; model = merge(ps.model, (; readout = readout))))
end

@doc raw"""
    ridge_map(features, targets, reg) -> Matrix

Tikhonov-regularized linear map `M` minimizing
``\|M \,\text{features} - \text{targets}\|^2 + \text{reg}\,\|M\|^2``. `features` is
``n_\text{in} \times T``, `targets` is ``n_\text{out} \times T``, and the returned
`M` has size ``n_\text{out} \times n_\text{in}``.
"""
function ridge_map(features::AbstractMatrix{<:Real}, targets::AbstractMatrix{<:Real}, reg::Real)
    T = float(promote_type(eltype(features), eltype(targets)))
    F = T.(features)
    Y = T.(targets)
    A = F * F' + T(reg) * I
    B = F * Y'
    return Matrix((A \ B)')
end

# Per-pattern aperture lookup: a scalar applies to every pattern; a dictionary
# supplies a value per name.
aperture_for(aperture::Real, ::Symbol) = aperture
function aperture_for(aperture::AbstractDict{Symbol, <:Real}, name::Symbol)
    haskey(aperture, name) || throw(ArgumentError("no aperture provided for :$name"))
    return aperture[name]
end

# ======================================================================
#  Loading patterns into a reservoir (Jaeger 2014, Section 3.2)
# ======================================================================

# Collect one pattern's reservoir states from a cleared carry, returning the
# post-washout states x(n), the lagged states x(n-1), and the aligned driver p(n).
function _drive_pattern(
        rng::AbstractRNG, concept::Conceptor, signal::AbstractMatrix,
        ps::NamedTuple, st::NamedTuple, washout::Int
    )
    st_reset = resetcarry!(rng, concept.model, st.model; init_carry = nothing)
    states, st_model = collectstates(concept.model, signal, ps.model, st_reset)
    L = size(states, 2)
    washout < L || throw(ArgumentError("washout=$washout ≥ signal length=$L"))
    X = states[:, (washout + 1):L]          # x(n)
    Xlag = states[:, washout:(L - 1)]       # x(n-1)
    P = signal[:, (washout + 1):L]          # driver aligned with x(n)
    return X, Xlag, P, merge(st, (; model = st_model))
end

@doc raw"""
    load!(rng, concept, named_signals, ps, st;
          aperture=10.0, washout=500, reg_recurrent=1e-4, reg_readout=1e-2)

Load named driving patterns into the reservoir wrapped by `concept`
(Jaeger 2014, Section 3.2). For each `name => signal` pair the reservoir is run
from a cleared carry, the first `washout` states are discarded, and the remaining
states are used to

  * derive and store a conceptor ``C = R (R + \alpha^{-2} I)^{-1}``, and
  * accumulate data for a single shared input-internalizing recurrent matrix `W`
    and readout `W_out`.

`W` is fitted (ridge, `reg_recurrent`) so that
``W x(n-1) \approx W^* x(n-1) + W_\text{in} p(n)``, absorbing the input drive into
the recurrent weights; `W_out` is fitted (ridge, `reg_readout`) so that
``W_\text{out} x(n) \approx p(n)`` across all patterns. `aperture` is either a
scalar or a `name => α` dictionary.

Returns `(ps, st)` with `reservoir_matrix` and `readout.weight` replaced and the
conceptor library populated.
"""
function load!(
        rng::AbstractRNG, concept::Conceptor, named_signals, ps::NamedTuple, st::NamedTuple;
        aperture = 10.0, washout::Int = 500, reg_recurrent::Real = 1.0e-4, reg_readout::Real = 1.0e-2
    )
    rps = reservoir_params(ps)
    W_star = Float64.(rps.reservoir_matrix)
    W_in = Float64.(rps.input_matrix)

    Xs = Matrix{Float64}[]
    Xlags = Matrix{Float64}[]
    Ps = Matrix{Float64}[]
    st_acc = st

    for (name, signal) in named_signals
        name isa Symbol || throw(ArgumentError("signal name must be a Symbol, got $(typeof(name))"))
        sig = signal isa AbstractMatrix ? Float64.(signal) : reshape(Float64.(signal), 1, :)
        X, Xlag, P, st_acc = _drive_pattern(rng, concept, sig, ps, st_acc, washout)
        α = aperture_for(aperture, name)
        store_conceptor!(st_acc, name, conceptor_from_states(X, α), α)
        push!(Xs, X)
        push!(Xlags, Xlag)
        push!(Ps, P)
    end

    Xall = reduce(hcat, Xs)
    Xlagall = reduce(hcat, Xlags)
    Pall = reduce(hcat, Ps)

    W_out = ridge_map(Xall, Pall, reg_readout)
    recurrent_target = W_star * Xlagall .+ W_in * Pall
    W = ridge_map(Xlagall, recurrent_target, reg_recurrent)

    T = eltype(rps.reservoir_matrix)
    ps2 = set_reservoir_matrix(ps, T.(W))
    ps2 = set_readout_weight(ps2, T.(W_out))
    return ps2, st_acc
end

# ======================================================================
#  Autonomous pattern generation (Jaeger 2014, Section 3.2)
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
(`res_dims × steps`). The rollout runs in `Float64` for numerical stability.
"""
function generate(
        concept::Conceptor, ps::NamedTuple, st::NamedTuple;
        conceptor::Union{Symbol, AbstractMatrix}, steps::Int, washout::Int = 200,
        init_state::Union{Nothing, AbstractVector} = nothing,
        rng::AbstractRNG = Random.default_rng()
    )
    rps = reservoir_params(ps)
    W = Float64.(rps.reservoir_matrix)
    b = reservoir_bias(ps)
    W_out = Float64.(readout_params(ps).weight)
    C = resolve_conceptor(st, conceptor)
    N = size(W, 1)

    x = init_state === nothing ? 0.5 .* randn(rng, N) : Float64.(init_state)
    out_dims = size(W_out, 1)
    Y = Matrix{Float64}(undef, out_dims, steps)
    X = Matrix{Float64}(undef, N, steps)

    for n in 1:(washout + steps)
        x = C * tanh.(W * x .+ b)
        if n > washout
            k = n - washout
            @views X[:, k] = x
            @views Y[:, k] = W_out * x
        end
    end
    return Y, X
end

# ======================================================================
#  Conceptor morphing (Jaeger 2014, Section 3.2)
# ======================================================================

@doc raw"""
    morph_conceptor(st, weights) -> Matrix{Float64}

Linearly combine stored conceptors into a morphed conceptor
``M = \sum_j \mu_j C_j`` (Jaeger 2014). `weights` is an iterable of `name => μ`
pairs, a `NamedTuple`, or a `Dict{Symbol,<:Real}`. Coefficients summing to one
interpolate between the named prototypes; coefficients outside `[0, 1]` extrapolate.
"""
function morph_conceptor(st::NamedTuple, weights)
    M = nothing
    for (name, μ) in pairs(weights)
        C = get_conceptor(st, Symbol(name))
        C === nothing && throw(KeyError(Symbol(name)))
        contribution = Float64(μ) .* C
        M = M === nothing ? contribution : M .+ contribution
    end
    M === nothing && throw(ArgumentError("no weights provided"))
    return M
end

# ======================================================================
#  Conceptors as state filters for supervised readout training
# ======================================================================

# Normalize a target into a row-major matrix (1 × T for a vector target).
_as_matrix(t::AbstractMatrix) = t
_as_matrix(t::AbstractVector) = reshape(t, 1, :)

"""
    store_conceptors!(rng, concept, named_signals, ps, st;
                      aperture=1.0, init_carry=nothing) -> st

Derive and store one conceptor per named signal without modifying the reservoir
weights. For each `name => signal` the reservoir is run from a cleared carry and
the conceptor `C(name) = R (R + α^{-2} I)^{-1}` of its state cloud is stored. Use
this to build a conceptor library for classification, where each class pattern
gets its own conceptor. `aperture` is a scalar or a `name => α` dictionary.
"""
function store_conceptors!(
        rng::AbstractRNG, concept::Conceptor, named_signals, ps::NamedTuple, st::NamedTuple;
        aperture = 1.0, init_carry = nothing
    )
    st_acc = st
    for (name, signal) in named_signals
        name isa Symbol || throw(ArgumentError("signal name must be a Symbol, got $(typeof(name))"))
        sig = signal isa AbstractMatrix ? signal : reshape(signal, 1, :)
        st_reset = resetcarry!(rng, concept.model, st_acc.model; init_carry = init_carry)
        states, st_model = collectstates(concept.model, sig, ps.model, st_reset)
        st_acc = merge(st_acc, (; model = st_model))
        α = aperture_for(aperture, name)
        store_conceptor!(st_acc, name, conceptor_from_states(states, α), α)
    end
    return st_acc
end

"""
    train!(rng, concept::Conceptor, named_signals, named_targets, ps, st,
           train_method=StandardRidge(0.0);
           washout=0, return_states=false, init_carry=nothing) -> (ps, st)

Train a single readout on conceptor-filtered reservoir features. For each
`name => signal` the states are collected through the conceptor named `name`
(stored beforehand via [`store_conceptors!`](@ref)), the first `washout` columns
are dropped, and features and the matching `named_targets` are pooled across
patterns before the readout is fit with `train_method`. With `return_states=true`
the pooled feature matrix is also returned.
"""
function train!(
        rng::AbstractRNG, concept::Conceptor, named_signals, named_targets,
        ps::NamedTuple, st::NamedTuple, train_method = StandardRidge(0.0);
        washout::Int = 0, return_states::Bool = false, init_carry = nothing, kwargs...
    )
    targets = Dict{Symbol, AbstractMatrix}(Symbol(name) => _as_matrix(t) for (name, t) in named_targets)

    all_states = AbstractMatrix[]
    all_targets = AbstractMatrix[]
    for (name, signal) in named_signals
        name isa Symbol || throw(ArgumentError("signal name must be a Symbol, got $(typeof(name))"))
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
