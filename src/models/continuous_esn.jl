@doc raw"""
    ContinuousESN(in_dims, res_dims, out_dims, [activation,] tspan, args...;
        use_bias = false,
        init_reservoir = rand_sparse, init_input = scaled_rand,
        init_bias = zeros32, init_state = randn32,
        equations = _continuous_esn_rhs!,
        state_modifiers = (), readout_activation = identity,
        kwargs...)

Continuous-time Echo State Network
([Lukosevicius2012](@cite)). Composes a [`ContinuousESNCell`](@ref),
optional `state_modifiers`, and a [`LinearReadout`](@ref).

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

## Arguments

  - `in_dims`: Input dimension.
  - `res_dims`: Reservoir (hidden state) dimension.
  - `out_dims`: Output dimension.
  - `activation`: Reservoir activation. Default: `tanh`.
  - `tspan`: Integration interval `(t0, t1)` for `collectstates`.
    Length-2, strictly increasing, finite.
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

  - `state_modifiers`: Layers applied to reservoir states before the
    readout. Accepts a single layer, an `AbstractVector`, or a `Tuple`.
    Default: empty `()`.
  - `readout_activation`: Activation for the linear readout.
    Default: `identity`.

Solve metadata:

  - `kwargs...`: Forwarded to `solve`. The keys `saveat`,
    `save_everystep`, and `dense` are reserved and rejected at
    construction.

## Parameters

  - `reservoir` — parameters of the internal [`ContinuousESNCell`](@ref):
      - `input_matrix :: (res_dims × in_dims)` — `W_in`
      - `reservoir_matrix :: (res_dims × res_dims)` — `W_r`
      - `bias :: (res_dims,)` — present only if `use_bias = true`
  - `states_modifiers` — parameters for each modifier layer (may be empty).
  - `readout` — parameters of [`LinearReadout`](@ref):
      - `weight :: (out_dims × res_dims)` — `W_out`
      - `bias :: (out_dims,)` — `b_out` (if the readout uses bias)

## States

  - `reservoir` — states for the internal [`ContinuousESNCell`](@ref).
  - `states_modifiers` — states for each modifier layer (may be empty).
  - `readout` — states for [`LinearReadout`](@ref).

!!! note
    The `RCODEReservoirExt` extension must be loaded for this constructor
    to succeed. Load a solver package such as `OrdinaryDiffEqTsit5`
    alongside `SciMLBase` and `DataInterpolations`.
"""
@concrete struct ContinuousESN <:
    AbstractEchoStateNetwork{(:reservoir, :states_modifiers, :readout)}
    reservoir
    states_modifiers
    readout
end

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
