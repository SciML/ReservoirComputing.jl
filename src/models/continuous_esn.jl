@doc raw"""
    ContinuousESN(in_dims, res_dims, out_dims; kwargs...)

Continuous-time leaky-integrator Echo State Network ([Lukosevicius2012](@cite)
§3.2.6, eq 5). **Work in progress — PR3 of the SciML 2026 fellowship roadmap
(#397).** This is a stub; the real constructor lives in the `RCODEReservoirExt`
package extension and will be filled in once the design is settled.

`ContinuousESN <: AbstractSciMLProblemReservoir`, so it reuses the
continuous-time `_collectstates` / `_predict` machinery landed in PR #450
(`add-ode-reservoir-ext`) and pre-bakes the leaky-integrator reservoir ODE
of [Lukosevicius2012](@cite) eq (5):

```math
\dot{\mathbf{x}}(t) = -\mathbf{x}(t) + \tanh\!\left(
    \mathbf{W}^{\text{in}}\,\mathbf{u}(t) + \mathbf{W}_r\,\mathbf{x}(t) + \mathbf{b}
\right)
```

The leak rate `α` familiar from the discrete leaky ESN is, in the continuous
formulation, exactly the Euler discretisation step: take `Δt = α` and the
update collapses to `x(n+1) = (1-α)x(n) + α·tanh(W_in u(n+1) + W_r x(n) + b)`.
It is therefore controlled in `ContinuousESN` via the integration `tspan` and
the input-window count, not as a separate parameter of the ODE.

This is *not* a port of the parametric-surrogate CTESN of
[Anantharaman2021](@cite); that model uses
`\dot{\mathbf{r}} = \tanh(A\,\mathbf{r} + W_{\text{hyb}}\,\mathbf{x}(t))` without
a decay term and is trained via RBF interpolation of `W_out(p)` over a
parameter space. It would land as a separate type in a future PR.

!!! note
    This constructor errors unless the `RCODEReservoirExt` extension is loaded
    (`SciMLBase` + `DataInterpolations`) **and** a concrete `OrdinaryDiffEq`
    solver package is available for `args[1]`.

## Status

- PR3 work in progress on branch `add-ctesn`, stacked on top of PR #450.
  Will be rebased onto `master` once #450 merges.
- Open design questions tracked in the pull-request description.
"""
ContinuousESN(::Any...) = error(
    "ContinuousESN requires the RCODEReservoirExt extension and an " *
        "OrdinaryDiffEq solver package. Load `SciMLBase`, `DataInterpolations`, " *
        "and a solver package (e.g. `OrdinaryDiffEqTsit5`) to enable it. PR3 is " *
        "a work-in-progress — the real constructor is not yet wired up."
)
