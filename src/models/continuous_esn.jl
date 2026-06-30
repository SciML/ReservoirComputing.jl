@doc raw"""
    ContinuousESN(in_dims, res_dims, out_dims, activation = tanh,
                  tspan, args...;
                  use_bias = false,
                  init_reservoir = rand_sparse, init_input = scaled_rand,
                  init_bias = zeros32, init_state = randn32,
                  state_modifiers = (), readout_activation = identity,
                  equations = _continuous_esn_rhs!,
                  kwargs...)

Continuous-time leaky-integrator Echo State Network ([Lukosevicius2012](@cite)
§3.2.6, eq 5). Thin convenience wrapper that composes:

  1) a [`ContinuousESNCell`](@ref) carrying the reservoir matrices and the
     ODE right-hand side,
  2) zero or more `state_modifiers` applied to the sampled reservoir
     states, and
  3) a [`LinearReadout`](@ref) mapping the modified states to outputs.

Structurally identical to [`ESN`](@ref) — three fields
`(reservoir, states_modifiers, readout)` — so it slots into the same
training and prediction pipeline. The reservoir field is the
`ContinuousESNCell` rather than a `StatefulLayer(ESNCell)`, which routes
`collectstates` / `predict` through the continuous-time
`RCODEReservoirExt` extension.

## Equations

```math
\begin{aligned}
    \dot{\mathbf{x}}(t) &= -\mathbf{x}(t) + \tanh\!\left(
        \mathbf{W}_{\text{in}}\,\mathbf{u}(t) + \mathbf{W}_r\,\mathbf{x}(t)
        + \mathbf{b}\right) \\
    \mathbf{z}(t) &= \mathrm{Mods}\!\left(\mathbf{x}(t)\right) \\
    \mathbf{y}(t) &= \rho\!\left(\mathbf{W}_{\text{out}}\,\mathbf{z}(t)
        + \mathbf{b}_{\text{out}}\right)
\end{aligned}
```

No leaking-rate term `α` appears in the ODE. `α` emerges only when the
ODE is forward-Euler discretised with step `Δt = α`, recovering the
discrete leaky update of [`ESN`](@ref). To change the effective leak,
adjust `tspan` so the per-input-window width matches the desired `Δt`.

## Arguments

  - `in_dims`: Input dimension.
  - `res_dims`: Reservoir (hidden state) dimension.
  - `out_dims`: Output dimension.
  - `activation`: Reservoir activation passed to the cell. Default: `tanh`.
  - `tspan`: Integration interval `(t0, t1)`. Length-2, strictly
    increasing, finite.
  - `args...`: Forwarded to `solve` positionally. The solver algorithm
    (`Tsit5()`, `Euler()`, …) is the first element by convention.

## Keyword arguments

Reservoir (passed to [`ContinuousESNCell`](@ref)):

  - `use_bias`: Whether the reservoir uses a bias term. Default: `false`.
  - `init_reservoir`: Initialiser for `W_r`. Default: [`rand_sparse`](@ref).
  - `init_input`: Initialiser for `W_in`. Default: [`scaled_rand`](@ref).
  - `init_bias`: Initialiser for `b`. Default: `zeros32`.
  - `init_state`: Initialiser for the initial hidden state.
    Default: `randn32`.
  - `equations`: ODE right-hand side. Default:
    [`_continuous_esn_rhs!`](@ref).

Composition:

  - `state_modifiers`: A layer or collection of layers applied to the
    sampled reservoir states before the readout. Accepts a single layer,
    an `AbstractVector`, or a `Tuple`. Default: empty `()`.
  - `readout_activation`: Activation for the linear readout. Default:
    `identity`.

Solve metadata:

  - `kwargs...`: Forwarded to `solve`. The three keys `saveat`,
    `save_everystep`, and `dense` are rejected at construction.

## Parameters

  - `reservoir` — parameters of the internal [`ContinuousESNCell`](@ref):
      - `input_matrix :: (res_dims × in_dims)` — `W_in`
      - `reservoir_matrix :: (res_dims × res_dims)` — `W_r`
      - `bias :: (res_dims,)` — present only if `use_bias = true`
  - `states_modifiers` — a `Tuple` with parameters for each modifier
    layer (may be empty).
  - `readout` — parameters of [`LinearReadout`](@ref), typically:
      - `weight :: (out_dims × res_dims)` — `W_out`
      - `bias :: (out_dims,)` — `b_out` (if the readout uses bias)

!!! note
    This constructor errors unless `RCODEReservoirExt` is loaded. Load
    `SciMLBase`, `DataInterpolations`, and a concrete `OrdinaryDiffEq`
    solver package (e.g. `OrdinaryDiffEqTsit5`, `OrdinaryDiffEq`) to
    enable it.
"""
@concrete struct ContinuousESN <:
    AbstractEchoStateNetwork{(:reservoir, :states_modifiers, :readout)}
    reservoir
    states_modifiers
    readout
end

# Public factory — the real implementation lives in `RCODEReservoirExt`.
# Loading `SciMLBase` + `DataInterpolations` adds the typed method that
# actually builds the cell; without the extension, this fallback fires
# and produces a clear error.
function ContinuousESN(::Any, ::Any, ::Any, ::Any...; kwargs...)
    return error(
        "ContinuousESN requires the RCODEReservoirExt extension and an " *
            "OrdinaryDiffEq solver package. Load `SciMLBase`, `DataInterpolations`, " *
            "and a solver package (e.g. `OrdinaryDiffEqTsit5`) to enable it."
    )
end

function Base.show(io::IO, esn::ContinuousESN)
    print(io, "ContinuousESN(\n")

    print(io, "    reservoir = ")
    show(io, esn.reservoir)
    print(io, ",\n")

    print(io, "    state_modifiers = ")
    if isempty(esn.states_modifiers)
        print(io, "()")
    else
        print(io, "(")
        for (idx, mod) in enumerate(esn.states_modifiers)
            idx > 1 && print(io, ", ")
            show(io, mod)
        end
        print(io, ")")
    end
    print(io, ",\n")

    print(io, "    readout = ")
    show(io, esn.readout)
    print(io, "\n)")

    return
end
