@doc raw"""
    CTESN(in_dims, res_dims, out_dims; kwargs...)

Continuous-Time Echo State Network ([Anantharaman2021](@cite)). **Work in progress
— PR3 of the SciML 2026 fellowship roadmap (#397).** This is a stub: the real
constructor lives in the `RCODEReservoirExt` package extension and will be
filled in once the design questions on the open pull request are resolved.

`CTESN <: AbstractSciMLProblemReservoir`, so it reuses the continuous-time
`_collectstates` / `_predict` machinery landed in PR #450 (`add-ode-reservoir-ext`).
It specialises that machinery to a concrete reservoir ODE.

The canonical reservoir ODE from [Anantharaman2021](@cite) eq. (3) is:

```math
\dot{\mathbf{r}}(t) = \tanh\!\left(A\,\mathbf{r}(t) + W_{\text{hyb}}\,\mathbf{x}(t)\right)
```

with no leak term and no bias — `f = tanh`, `g = id` hardcoded by the paper.
The implementation may also expose an opt-in leaky form
`ṙ = -α r + tanh(W_in u(t) + W_r r + b)` (the continuous limit of the
discrete leaky ESN of [Lukosevicius2012](@cite)); see the PR thread for the
default-form decision.

!!! note
    This constructor errors unless the `RCODEReservoirExt` extension is loaded
    (`SciMLBase` + `DataInterpolations`) **and** a concrete `OrdinaryDiffEq`
    solver package is available for `args[1]`.

## Status

- PR3 work in progress on branch `add-ctesn`, stacked on top of PR #450
  (`add-ode-reservoir-ext`). Will be rebased onto `master` once PR #450 merges.
- Open design questions are in the pull-request description.
"""
CTESN(::Any...) = error(
    "CTESN requires the RCODEReservoirExt extension and an OrdinaryDiffEq " *
        "solver package. Load `SciMLBase`, `DataInterpolations`, and a solver " *
        "package (e.g. `OrdinaryDiffEqTsit5`) to enable it. PR3 is a " *
        "work-in-progress — the real constructor is not yet wired up."
)
