"""
    AbstractSampler

Developer interface for a sampler that extracts a discrete state matrix from a
continuous-time reservoir trajectory.

## Fields

The marker itself requires no fields. A sampler may carry configuration such as
a window statistic or a within-window sampling rule.

## Extension contract

The continuous-reservoir extension calls
`__sample(sampler, sol)` after solving with one saved endpoint per input
column. A concrete method must return an `AbstractMatrix` with one column per
saved sample and a stable row dimension for the readout. It may inspect the
SciML solution object, but it must not change the solution or the input data.

Subtyping `AbstractSampler` alone is not enough: the matching `__sample`
method must be defined in the extension that owns the continuous-reservoir
implementation. This is a developer hook; ordinary users should use
[`TerminalStateSampling`](@ref).

## Example

```julia
struct WindowMean <: AbstractSampler
    width::Int
end

# Define `__sample(::WindowMean, sol)` in the continuous-reservoir extension.
```
"""
abstract type AbstractSampler end

"""
    TerminalStateSampling()

Sampler that records the reservoir state at the *end* of each input window.
With `T` input columns and `tspan = (t0, t1)`, `collectstates` splits
`tspan` into `T` equal-width windows; input column `k` is applied at the
start of window `k` and the state at the end of that window becomes column
`k` of the returned state matrix.

## Fields

None.

## Returns

The continuous-reservoir extension returns the saved solution values as an
`(state_dimension, T)` matrix. `states[:, k]` is the reservoir state after
processing input column `k`.

## Example

```julia
sampler = TerminalStateSampling()
reservoir = SciMLProblemReservoir(prob, sampler, (0.0, 1.0), Tsit5())
```
"""
struct TerminalStateSampling <: AbstractSampler end

"""
    AbstractSciMLProblemReservoir <: AbstractLuxLayer

Developer interface for a Lux layer whose dynamics are defined by an
`AbstractSciMLProblem` (typically `ODEProblem`, `SDEProblem`, or `DDEProblem`).

## Required fields

Concrete subtypes used with the built-in continuous-reservoir extension should
provide:

- `prob`: the SciML problem template;
- `sampler`: an [`AbstractSampler`](@ref);
- `tspan`: the integration interval used for each input sequence;
- `args`: positional solver arguments; and
- `kwargs`: keyword solver arguments that do not conflict with the sampling
  machinery.

## Extension contract

Implement `LuxCore.initialparameters` and `LuxCore.initialstates` for any
additional layer parameters or state. The continuous-reservoir extension then
dispatches the following developer hooks:

- `__collectstates(res, rc, data, ps, st) -> (states, st′)`, where `states` is
  an `(state_dimension, n_samples)` matrix;
- `__predict(res, rc, data, ps, st) -> (outputs, st′)` for teacher-forced
  prediction; and
- `__predict(res, rc, steps, ps, st; initialdata) -> (outputs, st′)` for
  autoregressive prediction.

The hooks must preserve the parameter/state layout expected by the enclosing
[`AbstractReservoirComputer`](@ref). Subtyping this type without implementing
the hooks only provides the default empty Lux parameter/state containers; it
does not make a new reservoir solvable.

## Example

```julia
struct MyContinuousReservoir <: AbstractSciMLProblemReservoir
    prob
    sampler
    tspan
    args
    kwargs
end
```

The continuous-time `__collectstates` implementation lives in the
`RCODEReservoirExt` package extension and requires `SciMLBase` and
`DataInterpolations` to be loaded. Pick any concrete solver package
separately (e.g. `OrdinaryDiffEqTsit5`, `OrdinaryDiffEq`) — its solver
types are what `SciMLProblemReservoir`'s `args[1]` consumes.
"""
abstract type AbstractSciMLProblemReservoir <: AbstractLuxLayer end

"""
    SciMLProblemReservoir(prob, sampler, tspan, args...; kwargs...)

Generic continuous-time reservoir layer wrapping any `AbstractSciMLProblem`.
Following the [DiffEqFlux NeuralDE pattern](https://github.com/SciML/DiffEqFlux.jl/blob/master/src/neural_de.jl)
the solver positional arguments and keyword arguments are captured at
construction time and forwarded to `solve` when `collectstates` runs.

## Arguments

- `prob`: an `AbstractSciMLProblem` (ODE/SDE/DDE) defining the reservoir
  dynamics. Left untyped so that any problem subtype can plug in without
  changes to the core types.
- `sampler`: an [`AbstractSampler`](@ref) controlling how the continuous
  trajectory is mapped to a discrete state matrix.
- `tspan`: the integration interval for `collectstates`. Overrides
  `prob.tspan` via `remake` at solve time, mirroring DiffEqFlux's NeuralODE.
- `args...`: positional arguments forwarded to `solve`. The solver algorithm
  (e.g. `Tsit5()`) is the first element by convention.
- `kwargs...`: keyword arguments forwarded to `solve`. The continuous helper
  owns three protected keys — `saveat`, `save_everystep`, and `dense` — because
  `collectstates` needs to synthesise a sample grid from `tspan` and the input
  width. Passing any of them at construction errors immediately.

The real `__collectstates` implementation lives in the `RCODEReservoirExt`
package extension. Without it loaded, calling `collectstates` on a
reservoir computer holding a `SciMLProblemReservoir` will error with a
message instructing the user to load `SciMLBase` and `DataInterpolations`
(plus a concrete solver package — `OrdinaryDiffEqTsit5`, `OrdinaryDiffEq`,
…).
"""
@concrete struct SciMLProblemReservoir <: AbstractSciMLProblemReservoir
    prob
    sampler
    tspan
    args
    kwargs
end

# Keyword arguments owned by the continuous `__collectstates` helper:
#   - `saveat` is derived from `tspan` and the input width, so a user value
#     would silently desync the sample grid from the input grid.
#   - `save_everystep` and `dense` are hardcoded to `false` because the
#     sampler only ever reads `sol.u` at the `saveat` points; allocating
#     the full trajectory would waste memory without changing the result.
# All three are rejected at construction so the user finds out immediately
# rather than getting a wrong-shape state matrix at solve time. Internal —
# not a docstring so Documenter's `:missing_docs` check leaves it alone.
const __PROTECTED_SOLVE_KWARGS = (:saveat, :save_everystep, :dense)

function __check_protected_kwargs(kwargs)
    collisions = filter(key -> key in __PROTECTED_SOLVE_KWARGS, keys(kwargs))
    isempty(collisions) && return nothing
    return throw(
        ArgumentError(
            "SciMLProblemReservoir rejects $(collect(collisions)) in `kwargs`: " *
                "these keys are set by `collectstates` from `tspan` and the input " *
                "data width. Drop them from the constructor call."
        )
    )
end

function SciMLProblemReservoir(prob, sampler, tspan, args...; kwargs...)
    # No type constraint on `sampler` here: a constrained outer constructor
    # makes this method strictly more specific than the inner constructor
    # generated by `@concrete`, causing infinite recursion at the 5-arg
    # call below. The DiffEqFlux NeuralDE pattern keeps these arguments
    # untyped for the same reason.
    __check_protected_kwargs(kwargs)
    return SciMLProblemReservoir(prob, sampler, tspan, args, kwargs)
end

# Empty parameters/state by default. Concrete subtypes (e.g. `ContinuousESN`)
# override these to expose reservoir matrices and any solver caches.
function initialparameters(::AbstractRNG, ::AbstractSciMLProblemReservoir)
    return NamedTuple()
end

function initialstates(::AbstractRNG, ::AbstractSciMLProblemReservoir)
    return NamedTuple()
end

function Base.show(io::IO, res::SciMLProblemReservoir)
    print(io, "SciMLProblemReservoir(")
    print(io, "prob = ")
    show(io, res.prob)
    print(io, ", sampler = ")
    show(io, res.sampler)
    print(io, ", tspan = ")
    show(io, res.tspan)
    print(io, ")")
    return
end
