"""
    AbstractSpikingNeuron

Developer marker for a neuron model used by the spiking reservoir interface.

## Fields

The marker itself requires no fields. A concrete neuron type must document the
state variables, parameter fields, and differential equations that its
extension uses.

## Extension contract

The current LSM implementation accepts only [`LIFNeuron`](@ref). Subtyping
`AbstractSpikingNeuron` does not by itself make a neuron valid for [`LSMCell`](@ref):
an extension must add the solver right-hand side, event handling, and feature
extraction methods for the new neuron. Those hooks are developer APIs, not
generic end-user dispatches.

## Example

```julia
struct MyNeuron <: AbstractSpikingNeuron
    tau_m::Float64
end
```

The example is only a type declaration; it is not accepted by `LSMCell` until
the corresponding extension contract is implemented.
"""
abstract type AbstractSpikingNeuron end

"""
    AbstractInputEncoder

Developer marker for an input-to-spike encoding used by [`LSMCell`](@ref).

## Fields

The marker itself requires no fields. A concrete encoder owns the parameters
needed to turn an input vector into the external current or event process
consumed by the neuron model.

## Extension contract

The current implementation accepts [`CurrentInjection`](@ref) and
[`PoissonRateEncoder`](@ref) only. A custom subtype must be integrated by an
extension that defines its encoder state initialization, input/event generation,
and solver callback behavior. Subtyping this marker alone does not make the
encoder accepted by `LSMCell`.

## Example

```julia
struct MyEncoder <: AbstractInputEncoder
    scale::Float64
end
```
"""
abstract type AbstractInputEncoder end

"""
    AbstractSpikeFeature

Developer marker for a feature map that converts a spiking trajectory into
reservoir features for [`LSMCell`](@ref).

## Fields

The marker itself requires no fields. A concrete feature map should document
its state, output feature dimension, and how each sample window is computed.

## Extension contract

The current implementation accepts [`SpikeCountFeatures`](@ref),
[`ExponentialSpikeFilter`](@ref), and [`MembraneVoltageFeature`](@ref). A
custom feature map must provide extension methods for its feature dimension,
autoregressive support, and sampled feature calculation. The extension passes
the spike times and unit indices together with the requested sample times; the
implementation must return one feature column per sample time. Subtyping this
marker alone does not add a feature map to `LSMCell`.

## Example

```julia
struct MySpikeFeature <: AbstractSpikeFeature
    window::Float64
end
```
"""
abstract type AbstractSpikeFeature end

@doc raw"""
    LIFNeuron(; tau_m=0.02, v_rest=0.0, v_reset=0.0, v_th=1.0,
              tau_ref=0.002, r_m=1.0, tau_syn=0.005)

Leaky integrate-and-fire neuron with exponential synaptic current
([GerstnerKistler2002](@cite)). Not [`LIFESN`](@ref) (Local Information Flow).

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

## Keyword arguments

  - `tau_m`: Membrane time constant. Default: `0.02`.
  - `v_rest`: Resting potential. Default: `0.0`.
  - `v_reset`: Reset potential. Default: `0.0`.
  - `v_th`: Spike threshold. Default: `1.0`.
  - `tau_ref`: Absolute refractory period. Default: `0.002`.
  - `r_m`: Membrane resistance. Default: `1.0`.
  - `tau_syn`: Synaptic decay time. Default: `0.005`.
"""
@concrete struct LIFNeuron <: AbstractSpikingNeuron
    tau_m
    v_rest
    v_reset
    v_th
    tau_ref
    r_m
    tau_syn
end

function LIFNeuron(;
        tau_m = 0.02, v_rest = 0.0, v_reset = 0.0, v_th = 1.0,
        tau_ref = 0.002, r_m = 1.0, tau_syn = 0.005
    )
    tau_m > 0 || throw(ArgumentError("tau_m must be positive, got $tau_m"))
    r_m > 0 || throw(ArgumentError("r_m must be positive, got $r_m"))
    tau_syn > 0 || throw(ArgumentError("tau_syn must be positive, got $tau_syn"))
    tau_ref >= 0 || throw(ArgumentError("tau_ref must be non-negative, got $tau_ref"))
    return LIFNeuron(tau_m, v_rest, v_reset, v_th, tau_ref, r_m, tau_syn)
end

"""
    CurrentInjection()

Stateless input encoder that supplies the instantaneous external current
``I^{\\mathrm{ext}}(t) = W_{\\mathrm{in}} u(t)`` (plus an optional bias).

## Fields

None.

## Returns

The LSM solver receives the current obtained by multiplying the input by the
reservoir input matrix. The encoder has no additional state.

## Example

```julia
LSMCell(3 => 20; tspan = (0.0, 1.0), encoder = CurrentInjection())
```
"""
struct CurrentInjection <: AbstractInputEncoder end

@doc raw"""
    PoissonRateEncoder(; rate_scale=50.0, synaptic_weight=1.0)

Poisson spikes at rate ``\lambda_k = \mathrm{rate\_scale}\,\max(u_k, 0)``.
No autoregressive `predict`.

## Keyword arguments

  - `rate_scale`: Input-to-rate scale. Default: `50.0`.
  - `synaptic_weight`: Weight into ``I_{\mathrm{syn}}``. Default: `1.0`.
"""
@concrete struct PoissonRateEncoder <: AbstractInputEncoder
    rate_scale
    synaptic_weight
end

function PoissonRateEncoder(; rate_scale = 50.0, synaptic_weight = 1.0)
    rate_scale >= 0 || throw(
        ArgumentError("rate_scale must be non-negative, got $rate_scale")
    )
    return PoissonRateEncoder(rate_scale, synaptic_weight)
end

"""
    SpikeCountFeatures()

Per-window spike counts, with one feature for each spiking unit. This feature
map does not support autoregressive `predict` because it represents event
counts over a sampled time window.

## Fields

None.

## Returns

For `n_units` neurons and `n_samples` sampling times, the extension returns an
`(n_units, n_samples)` feature matrix.

## Example

```julia
LSMCell(3 => 20; tspan = (0.0, 1.0), feature_map = SpikeCountFeatures())
```
"""
struct SpikeCountFeatures <: AbstractSpikeFeature end

@doc raw"""
    ExponentialSpikeFilter(; filter_tau=0.03)

```math
\dot s = -s/\tau + \sum_f \delta(t - t_f)
```

Sampled at each window end.

## Keyword arguments

  - `filter_tau`: Filter time constant. Default: `0.03`.
"""
@concrete struct ExponentialSpikeFilter <: AbstractSpikeFeature
    filter_tau
end

function ExponentialSpikeFilter(; filter_tau = 0.03)
    filter_tau > 0 || throw(
        ArgumentError("filter_tau must be positive, got $filter_tau")
    )
    return ExponentialSpikeFilter(filter_tau)
end

"""
    MembraneVoltageFeature()

Membrane voltage at each sampled window end. This feature map supports
autoregressive `predict`.

## Fields

None.

## Returns

For `n_units` neurons and `n_samples` sampling times, the extension returns an
`(n_units, n_samples)` matrix of sampled membrane voltages.

## Example

```julia
LSMCell(3 => 20; tspan = (0.0, 1.0), feature_map = MembraneVoltageFeature())
```
"""
struct MembraneVoltageFeature <: AbstractSpikeFeature end

__feature_dim(::AbstractSpikeFeature, n_units::Integer) = Int(n_units)
__supports_ar(::AbstractSpikeFeature) = false
__supports_ar(::ExponentialSpikeFilter) = true
__supports_ar(::MembraneVoltageFeature) = true

const __LSM_ENCODERS = Union{CurrentInjection, PoissonRateEncoder}
const __LSM_FEATURES = Union{
    SpikeCountFeatures, ExponentialSpikeFilter, MembraneVoltageFeature,
}

function __check_lsm_components(neuron, encoder, feature_map)
    neuron isa LIFNeuron || throw(
        ArgumentError(
            "LSM v1 supports only LIFNeuron; got $(typeof(neuron))"
        )
    )
    encoder isa __LSM_ENCODERS || throw(
        ArgumentError(
            "LSM v1 supports only CurrentInjection and PoissonRateEncoder; " *
                "got $(typeof(encoder))"
        )
    )
    feature_map isa __LSM_FEATURES || throw(
        ArgumentError(
            "LSM v1 supports only SpikeCountFeatures, ExponentialSpikeFilter, " *
                "and MembraneVoltageFeature; got $(typeof(feature_map))"
        )
    )
    return nothing
end

@doc raw"""
    LSMCell(in_dims => out_dims; tspan, args=(),
        neuron=LIFNeuron(), encoder=CurrentInjection(),
        feature_map=ExponentialSpikeFilter(),
        use_bias=false, init_bias=zeros32, init_reservoir=dale_sparse,
        init_input=scaled_rand, init_state=zeros32, kwargs...)

Spiking reservoir cell for [`LSM`](@ref) ([Maass2002](@cite)).
State is ``(V, I_{\mathrm{syn}})`` of length `2 * out_dims`.

## Arguments

  - `in_dims`: Input dimension.
  - `out_dims`: Reservoir dimension.

## Keyword arguments

  - `tspan`: Integration interval `(t0, t1)`. Length-2, strictly
    increasing, finite.
  - `args`: Positional `solve` arguments. Solver first by convention.
    Default: `()`.
  - `neuron`: [`LIFNeuron`](@ref) only in v1. Default: [`LIFNeuron`](@ref).
  - `encoder`: [`CurrentInjection`](@ref) or [`PoissonRateEncoder`](@ref).
    Default: [`CurrentInjection`](@ref).
  - `feature_map`: [`SpikeCountFeatures`](@ref),
    [`ExponentialSpikeFilter`](@ref), or [`MembraneVoltageFeature`](@ref).
    Default: [`ExponentialSpikeFilter`](@ref).
  - `use_bias`: Include bias. Default: `false`.
  - `init_reservoir`: For `W_r`. Default: [`dale_sparse`](@ref).
  - `init_input`: For `W_in`. Default: [`scaled_rand`](@ref).
  - `init_bias`: For bias when `use_bias=true`. Default: `zeros32`.
  - `init_state`: For membrane voltage. Default: `zeros32`.
  - `kwargs...`: Forwarded to `solve`. Rejected: `saveat`,
    `save_everystep`, `dense`, `callback`. Use `dtmax ≈ tau_ref/4`.

## Parameters

  - `input_matrix :: (out_dims × in_dims)` — `W_in`
  - `reservoir_matrix :: (out_dims × out_dims)` — `W_r`
  - `bias :: (out_dims,)` — if `use_bias = true`
"""
@concrete struct LSMCell <: AbstractSciMLProblemReservoir
    neuron
    encoder
    feature_map
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

function __check_lsm_tspan(tspan)
    length(tspan) == 2 || throw(
        ArgumentError(
            "tspan must be a length-2 tuple/pair (t0, t1), got length $(length(tspan))"
        )
    )
    (isfinite(tspan[1]) && isfinite(tspan[2])) || throw(
        ArgumentError("tspan endpoints must be finite, got $tspan")
    )
    tspan[2] > tspan[1] || throw(
        ArgumentError("LSM requires `tspan[2] > tspan[1]`, got tspan = $tspan")
    )
    return nothing
end

function LSMCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        tspan, args = (),
        neuron = LIFNeuron(),
        encoder = CurrentInjection(),
        feature_map = ExponentialSpikeFilter(),
        use_bias::BoolType = False(),
        init_bias = zeros32, init_reservoir = dale_sparse,
        init_input = scaled_rand, init_state = zeros32, kwargs...
    )
    in_dims > 0 || throw(ArgumentError("in_dims must be positive, got $in_dims"))
    out_dims > 0 || throw(ArgumentError("out_dims must be positive, got $out_dims"))
    __check_lsm_components(neuron, encoder, feature_map)
    __check_lsm_tspan(tspan)
    __check_protected_kwargs(kwargs)
    haskey(kwargs, :callback) && throw(
        ArgumentError("LSM owns the solve callback; drop `callback` from kwargs.")
    )
    return LSMCell(
        neuron, encoder, feature_map, in_dims, out_dims,
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
        encoder = __init_encoder_st(rng, cell.encoder),
    )
end

__init_encoder_st(::AbstractRNG, ::CurrentInjection) = NamedTuple()
function __init_encoder_st(rng::AbstractRNG, ::PoissonRateEncoder)
    return (rng = replicate(rng),)
end

function Base.show(io::IO, cell::LSMCell)
    print(io, "LSMCell(", cell.in_dims, " => ", cell.out_dims)
    print(io, ", tspan = ")
    show(io, cell.tspan)
    return print(io, ")")
end
