"""
    AbstractSpikingNeuron

Abstract supertype for continuous-time spiking neurons used by [`LSMCell`](@ref).
"""
abstract type AbstractSpikingNeuron end

"""
    AbstractInputEncoder

Abstract supertype for [`LSMCell`](@ref) input encodings.
"""
abstract type AbstractInputEncoder end

"""
    AbstractSpikeFeature

Abstract supertype for spike-side feature maps of [`LSMCell`](@ref).
"""
abstract type AbstractSpikeFeature end

@doc raw"""
    LIFNeuron(; tau_m=0.02, v_rest=0.0, v_reset=0.0, v_th=1.0,
              tau_ref=0.002, r_m=1.0, tau_syn=0.005)

Leaky integrate-and-fire neuron with exponential synaptic current
([GerstnerKistler2002](@cite)). Distinct from [`LIFESN`](@ref)
(Local Information Flow ESN).

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

``I^{\\mathrm{ext}}(t) = W_{\\mathrm{in}} u(t)`` (+ optional bias).
"""
struct CurrentInjection <: AbstractInputEncoder end

@doc raw"""
    PoissonRateEncoder(; rate_scale=50.0, synaptic_weight=1.0)

Poisson input spikes at rate ``\lambda_k = \mathrm{rate\_scale}\,\max(u_k, 0)``.
Teacher-forced `predict` only.

## Keyword arguments

  - `rate_scale`: Input-to-rate scale. Default: `50.0`.
  - `synaptic_weight`: Spike weight into ``I_{\mathrm{syn}}``. Default: `1.0`.
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

Per-window spike counts. Teacher-forced `predict` only.
"""
struct SpikeCountFeatures <: AbstractSpikeFeature end

@doc raw"""
    ExponentialSpikeFilter(; filter_tau=0.03)

Exponential spike filter sampled at each window end:

```math
\dot s = -s/\tau + \sum_f \delta(t - t_f)
```

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

Membrane voltage at each window end.
"""
struct MembraneVoltageFeature <: AbstractSpikeFeature end

__feature_dim(::AbstractSpikeFeature, n_units::Integer) = Int(n_units)
__supports_ar(::AbstractSpikeFeature) = true
__supports_ar(::SpikeCountFeatures) = false

@doc raw"""
    LSMCell(in_dims => out_dims; tspan, args=(),
        neuron=LIFNeuron(), encoder=CurrentInjection(),
        feature_map=ExponentialSpikeFilter(),
        use_bias=false, init_bias=zeros32, init_reservoir=dale_sparse,
        init_input=scaled_rand, init_state=zeros32, kwargs...)

Spiking reservoir cell for [`LSM`](@ref) ([Maass2002](@cite)). Continuous
state is ``(V, I_{\mathrm{syn}})`` of length `2 * out_dims`.

## Arguments

  - `in_dims`: Input dimension.
  - `out_dims`: Reservoir (spiking unit) dimension.

## Keyword arguments

  - `tspan`: Integration interval `(t0, t1)`. Length-2, strictly
    increasing, finite.
  - `args`: Tuple of positional arguments forwarded to `solve`. The solver
    algorithm (`Tsit5()`, `Euler()`, …) is the first element by convention.
    Default: `()`.
  - `neuron`: [`AbstractSpikingNeuron`](@ref). Default: [`LIFNeuron`](@ref).
  - `encoder`: [`AbstractInputEncoder`](@ref). Default:
    [`CurrentInjection`](@ref).
  - `feature_map`: [`AbstractSpikeFeature`](@ref). Default:
    [`ExponentialSpikeFilter`](@ref).
  - `use_bias`: Whether to include a bias term. Default: `false`.
  - `init_reservoir`: Initialiser for `W_r`. Default: [`dale_sparse`](@ref).
  - `init_input`: Initialiser for `W_in`. Default: [`scaled_rand`](@ref).
  - `init_bias`: Initialiser for the bias. Only used if `use_bias=true`.
    Default: `zeros32`.
  - `init_state`: Initialiser for the membrane voltage. Default: `zeros32`.
  - `kwargs...`: Forwarded to `solve`. The keys `saveat`, `save_everystep`,
    `dense`, and `callback` are reserved and rejected at construction.
    Prefer an explicit `dtmax ≈ tau_ref/4`.

## Parameters

  - `input_matrix :: (out_dims × in_dims)` — `W_in`
  - `reservoir_matrix :: (out_dims × out_dims)` — `W_r`
  - `bias :: (out_dims,)` — present only if `use_bias = true`
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
