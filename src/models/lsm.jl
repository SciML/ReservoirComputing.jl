@doc raw"""
    LSM(in_dims, res_dims, out_dims, tspan, args...;
        neuron=LIFCell(), encoder=CurrentInjection(),
        spike_readout=ExponentialFilterReadout(),
        use_bias=false, init_reservoir=dale_sparse, init_input=scaled_rand,
        init_bias=zeros32, init_state=zeros32,
        state_modifiers=(), readout_activation=identity, kwargs...)

Liquid State Machine ([Maass2002](@cite)). Composes an [`LSMCell`](@ref),
optional `state_modifiers`, and a [`LinearReadout`](@ref).

## Arguments

  - `in_dims`, `res_dims`, `out_dims`: input / reservoir / output sizes.
  - `tspan`: `(t0, t1)` for `collectstates`.
  - `args...`: Forwarded to `solve` (solver first).

## Keyword arguments

  - Reservoir: `neuron`, `encoder`, `spike_readout`, `use_bias`,
    `init_reservoir` (default [`dale_sparse`](@ref)), `init_input`,
    `init_bias`, `init_state` — see [`LSMCell`](@ref).
  - `state_modifiers`: Default `()`.
  - `readout_activation`: Default `identity`.
  - `kwargs...`: Forwarded to `solve`. Use `dtmax ≈ τ_ref/4`.
    `saveat`, `save_everystep`, `dense`, and `callback` are rejected.

## Parameters / states

Same three-field layout as other models: `reservoir`, `state_modifiers`,
`readout`.

!!! note
    Requires `RCODEReservoirExt` (`SciMLBase`, `DataInterpolations`, and a
    solver package such as `OrdinaryDiffEqTsit5`).
"""
@concrete struct LSM <:
    AbstractEchoStateNetwork{(:reservoir, :state_modifiers, :readout)}
    reservoir
    state_modifiers
    readout
end

function LSM(::Any, ::Any, ::Any, ::Any...; kwargs...)
    return error(
        "LSM requires the RCODEReservoirExt extension and an OrdinaryDiffEq " *
            "solver package. Load `SciMLBase`, `DataInterpolations`, and a solver " *
            "package (e.g. `OrdinaryDiffEqTsit5`) to enable it."
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
