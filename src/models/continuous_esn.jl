@doc raw"""
    ContinuousESN(in_dims, res_dims, tspan, args...; kwargs...)

Continuous-time leaky-integrator Echo State Network ([Lukosevicius2012](@cite)
§3.2.6, eq 5). `ContinuousESN <: AbstractSciMLProblemReservoir`, so it reuses
the continuous-time `_collectstates` / `_predict` machinery in the
`RCODEReservoirExt` package extension and pre-bakes the leaky-integrator
reservoir ODE

```math
\dot{\mathbf{x}}(t) = \alpha \left(
    -\mathbf{x}(t) + \tanh\!\left(
        \mathbf{W}_{\text{in}}\,\mathbf{u}(t) + \mathbf{W}_r\,\mathbf{x}(t) + \mathbf{b}
    \right)
\right)
```

with reservoir matrix `W_r`, input matrix `W_in`, optional bias `b`, and
scalar leaking rate `α` exposed as `leak_coefficient`. Forward-Euler
discretisation at step `Δt = 1` collapses the ODE to the discrete leaky
update `x(n+1) = (1-α)·x(n) + α·tanh(W_in·u(n) + W_r·x(n) + b)`.

This is *not* a port of the parametric-surrogate CTESN of
[Anantharaman2021](@cite); that model uses `ṙ = tanh(A·r + W_hyb·u)` without
a decay term and is trained via RBF interpolation of `W_out(p)` over a
parameter space. A separate type would land for it in a future PR.

## Arguments

  - `in_dims`: Input dimension.
  - `res_dims`: Reservoir (hidden state) dimension.
  - `tspan`: Integration interval for `collectstates`. Must be a length-2,
    strictly-increasing, finite tuple/pair. Mirrors the DiffEqFlux
    `NeuralODE` convention.
  - `args...`: Positional arguments forwarded to `solve`. The solver
    algorithm (e.g. `Tsit5()`, `Euler()`) is the first element by
    convention.

## Keyword arguments

  - `leak_coefficient`: Leaking rate `α`. Default: `1.0`. Pairing `Δt = 1`
    with `leak_coefficient = α` recovers the discrete leaky ESN under
    forward Euler.
  - `use_bias`: Whether to include a bias term `b`. Default: `false`.
  - `init_reservoir`: Initializer for `W_r`. Default:
    [`rand_sparse`](@ref). Receives `(rng, T, res_dims, res_dims)`.
  - `init_input`: Initializer for `W_in`. Default: [`scaled_rand`](@ref).
    Receives `(rng, T, res_dims, in_dims)`.
  - `init_bias`: Initializer for `b`. Used only when `use_bias = true`.
    Default: a type-aware zeros initialiser. Receives `(rng, T, res_dims)`.
  - `T`: Element type for the reservoir matrices and initial state.
    Default: `Float32`.
  - `kwargs...`: Forwarded to `solve`. The three keys `saveat`,
    `save_everystep`, and `dense` are owned by the extension helper and
    rejected at construction.

## Continuous-time stability note

`scale_radius!` and the default reservoir initialisers control the
discrete-time spectral radius `ρ(W_r)`. For the α-factored ODE used here,
a log-norm contraction argument gives the sufficient condition
`‖W_r‖_2 < 1` (operator 2-norm bound), independent of `α`: the leaking
rate scales the rate of decay but not the asymptotic bound. The condition
is strictly stronger than `ρ(W_r) < 1` because `‖·‖_2 ≥ ρ(·)`. Published
continuous ESN work treats `ρ` as an empirical hyperparameter — adjust
accordingly.

!!! note
    This constructor errors unless `RCODEReservoirExt` is loaded. Load
    `SciMLBase`, `DataInterpolations`, and a concrete OrdinaryDiffEq
    solver package (e.g. `OrdinaryDiffEqTsit5`, `OrdinaryDiffEq`) to
    enable it.
"""
@concrete struct ContinuousESN <: AbstractSciMLProblemReservoir
    prob
    sampler
    tspan
    args
    kwargs
    in_dims
    res_dims
    leak_coefficient
    use_bias
    init_reservoir
    init_input
    init_bias
    matrix_type
end

# Public factory — the real implementation lives in `RCODEReservoirExt`.
# Loading `SciMLBase` + `DataInterpolations` adds the typed method that
# actually builds the `ODEProblem`; without the extension, this fallback
# fires and produces a clear error.
function ContinuousESN(::Any, ::Any, ::Any, ::Any...; kwargs...)
    return error(
        "ContinuousESN requires the RCODEReservoirExt extension and an " *
            "OrdinaryDiffEq solver package. Load `SciMLBase`, `DataInterpolations`, " *
            "and a solver package (e.g. `OrdinaryDiffEqTsit5`) to enable it."
    )
end

function Base.show(io::IO, ce::ContinuousESN)
    print(io, "ContinuousESN(")
    print(io, "in_dims = ", ce.in_dims)
    print(io, ", res_dims = ", ce.res_dims)
    print(io, ", leak_coefficient = ", ce.leak_coefficient)
    print(io, ", use_bias = ", ce.use_bias)
    print(io, ", tspan = ")
    show(io, ce.tspan)
    print(io, ")")
    return
end

function initialparameters(rng::AbstractRNG, ce::ContinuousESN)
    T = ce.matrix_type
    W_r = ce.init_reservoir(rng, T, ce.res_dims, ce.res_dims)
    W_in = ce.init_input(rng, T, ce.res_dims, ce.in_dims)
    leak_coefficient = T(ce.leak_coefficient)
    isfinite(leak_coefficient) || throw(
        ArgumentError(
            "leak_coefficient converted to $T overflowed to $leak_coefficient; " *
                "pick a smaller value or widen `T`."
        )
    )
    ps = (W_r = W_r, W_in = W_in, leak_coefficient = leak_coefficient)
    if ce.use_bias
        ps = merge(ps, (b = ce.init_bias(rng, T, ce.res_dims),))
    end
    return ps
end

function initialstates(::AbstractRNG, ::ContinuousESN)
    return NamedTuple()
end
