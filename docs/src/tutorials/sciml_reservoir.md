# Continuous-Time Reservoirs from a `SciMLProblem`

ReservoirComputing.jl exposes a continuous-time reservoir layer,
[`SciMLProblemReservoir`](@ref), that wraps any
`AbstractSciMLProblem` (`ODEProblem`, `SDEProblem`, `DDEProblem`) and
plugs it into the standard `collectstates` / `predict` pipeline. The
implementation lives in the `RCODEReservoirExt` package extension, so
the extension is loaded automatically once `OrdinaryDiffEq`,
`SciMLBase`, and `DataInterpolations` are in scope.

This page walks through the core type, how time is laid out internally,
and a worked example that checks the continuous reservoir against a
closed-form analytic solution.

## Loading the extension

```julia
using ReservoirComputing
using OrdinaryDiffEq
using SciMLBase
using DataInterpolations
```

All three are required: `OrdinaryDiffEq` brings the concrete solver
types (e.g. `Tsit5()`), `SciMLBase` provides `solve` / `remake`, and
`DataInterpolations` is used for the per-window input signal in the
autoregressive `predict` path.

## Constructing a reservoir from an ODE problem

The constructor follows the
[DiffEqFlux `NeuralODE` pattern](https://github.com/SciML/DiffEqFlux.jl/blob/master/src/neural_de.jl):

```julia
SciMLProblemReservoir(prob, sampler, tspan, args...; kwargs...)
```

* `prob` — any `AbstractSciMLProblem`. The reservoir's initial state is
  taken from `prob.u0`; the ODE right-hand side reads the time-varying
  input through `p.input(t)` (injected by the extension at solve time).
* `sampler` — an [`AbstractSampler`](@ref). The bundled
  [`TerminalStateSampling`](@ref) records the reservoir state at the
  end of each input window.
* `tspan` — overrides `prob.tspan` via `remake` at solve time. The
  input column grid is synthesised from `tspan` and the input width.
* `args...` — forwarded to `solve` positionally; the solver algorithm
  is the first element (`Tsit5()`, `Euler()`, …).
* `kwargs...` — forwarded to `solve`. The three keys `saveat`,
  `save_everystep`, and `dense` are owned by the helper and rejected
  at construction.

The user's ODE can carry static parameters as a `NamedTuple` (or
`nothing` / `NullParameters()` if there are none). Anything else is
rejected with an `ArgumentError`. The reserved name `:input` cannot
appear in `prob.p` because the extension uses it to inject the
interpolated input signal.

## How time is laid out

`collectstates` receives a discrete `data::AbstractMatrix` of shape
`(channels, n_samples)` and turns it into a continuous-time input:

1. `tspan` is split into `n_samples` equal-width windows of width
   `Δt = (t1 - t0) / n_samples`.
2. Input column `k` is held over window `k`
   (`t0 + (k-1)Δt ≤ t < t0 + kΔt`) via zero-order hold.
3. The reservoir state at the **end** of window `k`
   (`t = t0 + kΔt`) is sampled and becomes `states[:, k]`.

This input-at-start / sample-at-end alignment matches the discrete
update semantics: `states[:, k]` is the reservoir state after
processing input `k`, with no off-by-one offset.

## A worked example: linear ODE with closed-form solution

The scalar linear ODE `dx/dt = -x + u(t)` with `u(t) = 1` and
`x(0) = 0` has the closed form `x(t) = 1 - exp(-t)`. Running it
through the continuous reservoir at a tight solver tolerance recovers
that curve to within ~1e-6:

```julia
using ReservoirComputing
using OrdinaryDiffEq
using SciMLBase
using DataInterpolations
using Random

function linear_rhs!(dx, x, p, t)
    u_val = p.input(t)
    dx .= .-x .+ u_val
end

n_samples = 10
tspan = (0.0, 1.0)
Δt = (tspan[2] - tspan[1]) / n_samples
sample_ts = collect(range(tspan[1] + Δt, tspan[2]; length = n_samples))

u_const = 1.0
data = fill(u_const, 1, n_samples)
initial_state = [0.0]

prob = ODEProblem(linear_rhs!, initial_state, tspan, (;))
res = SciMLProblemReservoir(
    prob, TerminalStateSampling(), tspan, Tsit5();
    reltol = 1.0e-10, abstol = 1.0e-12
)
rc = ReservoirComputer(res, LinearReadout(1 => 1))
ps, st = setup(MersenneTwister(0), rc)

states, _ = collectstates(rc, data, ps, st)
analytic = u_const .* (1 .- exp.(.-sample_ts))

@assert states[1, :] ≈ analytic atol = 1.0e-6
```

## Calling `predict`

Both `predict` signatures route through the same continuous helper:

```julia
predict(rc, data, ps, st)                  # teacher-forced
predict(rc, steps, ps, st; initialdata)    # autoregressive rollout
```

* The teacher-forced path solves the full `tspan` once and applies the
  readout column-by-column to the sampled states.
* The autoregressive path splits `tspan` into `steps` sub-intervals,
  feeds the previous readout output back as the constant input on
  the next sub-interval, and stitches the per-window readouts into
  the returned output matrix.

In both cases the reservoir's initial state is `prob.u0`. To continue
from a previously computed trajectory, `remake(prob; u0 = …)` before
constructing the reservoir.

## Adding your own sampler

The reservoir state sequence the readout sees is produced by an
[`AbstractSampler`](@ref). To plug in a custom strategy (window mean,
sub-sampling within a window, etc.), define a concrete subtype and a
matching `_sample(::YourSampler, sol)` method inside an extension that
also loads `OrdinaryDiffEq` and `SciMLBase`. The method should return
a `(state_dim, n_samples)` matrix; everything downstream (state
modifiers, readout, predict) is sampler-agnostic.
