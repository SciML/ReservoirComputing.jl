@doc raw"""
    LSM(in_dims, res_dims, out_dims, tspan, args...;
        neuron=LIFNeuron(), encoder=CurrentInjection(),
        feature_map=ExponentialSpikeFilter(),
        use_bias=false, init_reservoir=dale_sparse, init_input=scaled_rand,
        init_bias=zeros32, init_state=zeros32,
        state_modifiers=(), readout_activation=identity, kwargs...)

Liquid State Machine ([Maass2002](@cite)): [`LSMCell`](@ref), optional
`state_modifiers`, and [`LinearReadout`](@ref).

## Arguments

  - `in_dims`: Input dimension.
  - `res_dims`: Reservoir dimension.
  - `out_dims`: Output dimension.
  - `tspan`: `(t0, t1)` for `collectstates`. Length-2, strictly
    increasing, finite.
  - `args...`: Positional `solve` arguments. Solver first by convention.

## Keyword arguments

Reservoir (to [`LSMCell`](@ref)):

  - `neuron`: Default [`LIFNeuron`](@ref).
  - `encoder`: Default [`CurrentInjection`](@ref).
  - `feature_map`: Default [`ExponentialSpikeFilter`](@ref).
  - `use_bias`: Default `false`.
  - `init_reservoir`: Default [`dale_sparse`](@ref).
  - `init_input`: Default [`scaled_rand`](@ref).
  - `init_bias`: Default `zeros32`.
  - `init_state`: Default `zeros32`.

Composition:

  - `state_modifiers`: Layer, vector, or tuple. Default: `()`.
  - `readout_activation`: Default: `identity`.

Solve:

  - `kwargs...`: Forwarded to `solve`. Rejected: `saveat`,
    `save_everystep`, `dense`, `callback`. Use `dtmax ≈ tau_ref/4`.

## Parameters

  - `reservoir` — [`LSMCell`](@ref)
  - `state_modifiers`
  - `readout` — [`LinearReadout`](@ref)

## States

  - `reservoir`
  - `state_modifiers`
  - `readout`

!!! note
    Requires `RCODEReservoirExt` (`SciMLBase`, `DataInterpolations`, and a
    solver package such as `OrdinaryDiffEqTsit5`).
"""
@concrete struct LSM <:
    AbstractReservoirComputer{(:reservoir, :state_modifiers, :readout)}
    reservoir
    state_modifiers
    readout
end

function LSM(::Any, ::Any, ::Any, ::Any...; kwargs...)
    return error(
        "LSM requires the RCODEReservoirExt extension and an " *
            "OrdinaryDiffEq solver package. Load `SciMLBase`, `DataInterpolations`, " *
            "and a solver package (e.g. `OrdinaryDiffEqTsit5`) to enable it."
    )
end

function Base.show(io::IO, lsm::LSM)
    print(io, "LSM(\n")

    print(io, "    reservoir = ")
    show(io, lsm.reservoir)
    print(io, ",\n")

    print(io, "    state_modifiers = ")
    if isempty(lsm.state_modifiers)
        print(io, "()")
    else
        print(io, "(")
        for (idx, mod) in enumerate(lsm.state_modifiers)
            idx > 1 && print(io, ", ")
            show(io, mod)
        end
        print(io, ")")
    end
    print(io, ",\n")

    print(io, "    readout = ")
    show(io, lsm.readout)
    print(io, "\n)")

    return
end
