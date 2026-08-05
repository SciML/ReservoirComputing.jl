@doc raw"""
    LSM(in_dims, res_dims, out_dims, tspan, args...;
        neuron=LIFNeuron(), encoder=CurrentInjection(),
        feature_map=ExponentialSpikeFilter(),
        use_bias=false, init_reservoir=dale_sparse, init_input=scaled_rand,
        init_bias=zeros32, init_state=zeros32,
        state_modifiers=(), readout_activation=identity, kwargs...)

Liquid State Machine ([Maass2002](@cite)). Composes an [`LSMCell`](@ref),
optional `state_modifiers`, and a [`LinearReadout`](@ref).

## Arguments

  - `in_dims`: Input dimension.
  - `res_dims`: Reservoir (spiking unit) dimension.
  - `out_dims`: Output dimension.
  - `tspan`: Integration interval `(t0, t1)` for `collectstates`.
    Length-2, strictly increasing, finite.
  - `args...`: Forwarded to `solve` positionally. The solver algorithm
    (`Tsit5()`, `Euler()`, …) is the first element by convention.

## Keyword arguments

Reservoir (passed to [`LSMCell`](@ref)):

  - `neuron`: [`AbstractSpikingNeuron`](@ref). Default: [`LIFNeuron`](@ref).
  - `encoder`: [`AbstractInputEncoder`](@ref). Default:
    [`CurrentInjection`](@ref).
  - `feature_map`: [`AbstractSpikeFeature`](@ref). Default:
    [`ExponentialSpikeFilter`](@ref).
  - `use_bias`: Whether the reservoir uses a bias term. Default: `false`.
  - `init_reservoir`: Initialiser for `W_r`. Default: [`dale_sparse`](@ref).
  - `init_input`: Initialiser for `W_in`. Default: [`scaled_rand`](@ref).
  - `init_bias`: Initialiser for `b`. Default: `zeros32`.
  - `init_state`: Initialiser for the membrane voltage. Default: `zeros32`.

Composition:

  - `state_modifiers`: Layers applied to features before the readout.
    Accepts a single layer, an `AbstractVector`, or a `Tuple`.
    Default: empty `()`.
  - `readout_activation`: Activation for the linear readout.
    Default: `identity`.

Solve metadata:

  - `kwargs...`: Forwarded to `solve`. Prefer an explicit
    `dtmax ≈ tau_ref/4`. The keys `saveat`, `save_everystep`, `dense`,
    and `callback` are reserved and rejected at construction.

## Parameters

  - `reservoir` — parameters of the internal [`LSMCell`](@ref):
      - `input_matrix :: (res_dims × in_dims)` — `W_in`
      - `reservoir_matrix :: (res_dims × res_dims)` — `W_r`
      - `bias :: (res_dims,)` — present only if `use_bias = true`
  - `state_modifiers` — parameters for each modifier layer (may be empty).
  - `readout` — parameters of [`LinearReadout`](@ref):
      - `weight :: (out_dims × feature_dims)` — `W_out`
      - `bias :: (out_dims,)` — `b_out` (if the readout uses bias)

## States

  - `reservoir` — states for the internal [`LSMCell`](@ref).
  - `state_modifiers` — states for each modifier layer (may be empty).
  - `readout` — states for [`LinearReadout`](@ref).

!!! note
    The `RCODEReservoirExt` extension must be loaded for this constructor
    to succeed. Load a solver package such as `OrdinaryDiffEqTsit5`
    alongside `SciMLBase` and `DataInterpolations`.
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
