"""
    AbstractSpikingNeuron

Supertype for continuous-time spiking neuron models used by [`LSMCell`](@ref).
"""
abstract type AbstractSpikingNeuron end

"""
    AbstractInputEncoder

Supertype for [`LSMCell`](@ref) input encodings.
"""
abstract type AbstractInputEncoder end

"""
    AbstractSpikeReadout

Supertype for spike-side feature maps of [`LSMCell`](@ref) (before the
linear ridge readout).
"""
abstract type AbstractSpikeReadout end

@doc raw"""
    LIFCell(; τ_m=0.02, V_rest=0.0, V_reset=0.0, V_th=1.0,
            τ_ref=0.002, R_m=1.0, τ_syn=0.005)

Leaky integrate-and-fire population with exponential synaptic current
([GerstnerKistler2002](@cite)). Not related to [`LIFESN`](@ref).

```math
\begin{aligned}
    \tau_m \dot V_i &= -(V_i - V_{\mathrm{rest}})
        + R_m (I_i + I^{\mathrm{ext}}_i)
        && t \ge t^{\mathrm{ref}}_i \\
    \tau_{\mathrm{syn}} \dot I_i &= -I_i \\
    V_i &\leftarrow V_{\mathrm{reset}},\quad
    I \mathrel{+}= W_{:,i}
        && V_i \ge V_{\mathrm{th}}
\end{aligned}
```
"""
@concrete struct LIFCell <: AbstractSpikingNeuron
    τ_m <: Number
    V_rest <: Number
    V_reset <: Number
    V_th <: Number
    τ_ref <: Number
    R_m <: Number
    τ_syn <: Number
end

function LIFCell(;
        τ_m::Number = 0.02, V_rest::Number = 0.0, V_reset::Number = 0.0,
        V_th::Number = 1.0, τ_ref::Number = 0.002, R_m::Number = 1.0,
        τ_syn::Number = 0.005
    )
    return LIFCell(τ_m, V_rest, V_reset, V_th, τ_ref, R_m, τ_syn)
end

"""
    CurrentInjection()

``I^{\\mathrm{ext}}(t) = W_{\\mathrm{in}} u(t)`` (+ optional bias).
"""
struct CurrentInjection <: AbstractInputEncoder end

@doc raw"""
    PoissonRateEncoder(; scale=50.0, weight=1.0)

Poisson spikes at rate ``\lambda_k = \mathrm{scale}\,\max(u_k, 0)``.
Times are drawn once per `collectstates` from `st.encoder.rng`.
Teacher-forced `predict` only (no AR).
"""
@concrete struct PoissonRateEncoder <: AbstractInputEncoder
    scale <: Number
    weight <: Number
end

function PoissonRateEncoder(; scale::Number = 50.0, weight::Number = 1.0)
    return PoissonRateEncoder(scale, weight)
end

"""
    SpikeCountReadout()

Per-window spike counts. Teacher-forced `predict` only.
"""
struct SpikeCountReadout <: AbstractSpikeReadout end

@doc raw"""
    ExponentialFilterReadout(τ=0.03)

``\dot s = -s/\tau + \sum_f \delta(t-t_f)``, sampled at window ends.
"""
@concrete struct ExponentialFilterReadout <: AbstractSpikeReadout
    τ <: Number
end

ExponentialFilterReadout() = ExponentialFilterReadout(0.03)

"""
    FilteredVoltageReadout()

Membrane voltage at each window end.
"""
struct FilteredVoltageReadout <: AbstractSpikeReadout end

_feature_dim(::AbstractSpikeReadout, n_units::Integer) = Int(n_units)
_supports_ar(::AbstractSpikeReadout) = true
_supports_ar(::SpikeCountReadout) = false

@doc raw"""
    LSMCell(in_dims => out_dims; tspan, args=(),
        neuron=LIFCell(), encoder=CurrentInjection(),
        spike_readout=ExponentialFilterReadout(),
        use_bias=false, init_bias=zeros32, init_reservoir=dale_sparse,
        init_input=scaled_rand, init_state=zeros32, kwargs...)

Spiking reservoir cell for [`LSM`](@ref) ([Maass2002](@cite)).
State is ``(V, I_{\mathrm{syn}})`` of length `2 * out_dims`.

## Arguments

  - `in_dims`: Input dimension.
  - `out_dims`: Number of units.

## Keyword arguments

  - `tspan`: `(t0, t1)`, strictly increasing and finite.
  - `args`: Forwarded to `solve` (solver first).
  - `neuron`: [`AbstractSpikingNeuron`](@ref). Default: [`LIFCell`](@ref).
  - `encoder`: [`AbstractInputEncoder`](@ref). Default:
    [`CurrentInjection`](@ref).
  - `spike_readout`: [`AbstractSpikeReadout`](@ref). Default:
    [`ExponentialFilterReadout`](@ref).
  - `use_bias`: Default `false`.
  - `init_reservoir`: Default [`dale_sparse`](@ref).
  - `init_input`: Default [`scaled_rand`](@ref).
  - `init_bias` / `init_state`: Default `zeros32`.
  - `kwargs...`: Forwarded to `solve`. Use an explicit `dtmax ≈ τ_ref/4`.
    `saveat`, `save_everystep`, `dense`, and `callback` are rejected.

## Parameters

  - `input_matrix :: (out_dims × in_dims)`
  - `reservoir_matrix :: (out_dims × out_dims)`
  - `bias :: (out_dims,)` if `use_bias=true`

## States

  - `rng`, `encoder`
"""
@concrete struct LSMCell <: AbstractSciMLProblemReservoir
    neuron <: AbstractSpikingNeuron
    encoder <: AbstractInputEncoder
    spike_readout <: AbstractSpikeReadout
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_reservoir
    init_input
    init_state
    use_bias <: StaticBool
    tspan
    args
    kwargs
end

function LSMCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        tspan, args = (),
        neuron::AbstractSpikingNeuron = LIFCell(),
        encoder::AbstractInputEncoder = CurrentInjection(),
        spike_readout::AbstractSpikeReadout = ExponentialFilterReadout(),
        use_bias::BoolType = False(),
        init_bias = zeros32, init_reservoir = dale_sparse,
        init_input = scaled_rand, init_state = zeros32, kwargs...
    )
    return LSMCell(
        neuron, encoder, spike_readout, in_dims, out_dims,
        init_bias, init_reservoir, init_input, init_state,
        static(use_bias), tspan, args, kwargs
    )
end

function initialparameters(rng::AbstractRNG, cell::LSMCell)
    ps = (
        input_matrix = cell.init_input(rng, cell.out_dims, cell.in_dims),
        reservoir_matrix = cell.init_reservoir(rng, cell.out_dims, cell.out_dims),
    )
    if has_bias(cell)
        ps = merge(ps, (bias = cell.init_bias(rng, cell.out_dims),))
    end
    return ps
end

function initialstates(rng::AbstractRNG, cell::LSMCell)
    return (
        rng = replicate(rng),
        encoder = _init_encoder_st(rng, cell.encoder),
    )
end

_init_encoder_st(::AbstractRNG, ::CurrentInjection) = NamedTuple()
function _init_encoder_st(rng::AbstractRNG, ::PoissonRateEncoder)
    return (rng = replicate(rng),)
end

function Base.show(io::IO, cell::LSMCell)
    print(io, "LSMCell(", cell.in_dims, " => ", cell.out_dims)
    print(io, ", tspan = ")
    show(io, cell.tspan)
    return print(io, ")")
end
