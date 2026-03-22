using DifferentialEquations
using LinearAlgebra
using Random
using Statistics
using DataInterpolations

# Maybe have a struct for ReservoirParams as well for cleaner ps

@concrete struct CTESNCell <: AbstractReservoirRecurrentCell
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_reservoir
    init_input
    init_state
    leak_coefficient
    use_bias <: StaticBool
end

# This struct for CTESNCell should have extra attributes like solver, sampler, etc.
function CTESNCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType},
        activation = tanh_fast; use_bias::BoolType = False(), init_bias = zeros32,
        init_reservoir = rand_sparse, init_input = scaled_rand,
        init_state = randn32,
        leak_coefficient::Union{AbstractFloat, AbstractVector} = 1.0
    )

    if isa(leak_coefficient, AbstractVector)
        @assert length(leak_coefficient) == out_dims "leak_coefficient must match reservoir size"
    end

    return CTESNCell(
        activation, in_dims, out_dims, init_bias, init_reservoir,
        init_input, init_state, leak_coefficient, static(use_bias)
    )
end

# forcing function ODE
function (ctesn!::CTESNCell)(dx, x, ps, t)
    bias = safe_getproperty(ps, Val(:bias))

    if hasproperty(ps, :input)
        # --- teacher forcing ---
        u = ps.input(t)
    else
        # --- autoregressive ---
        u = ps.readout.weight * x
    end

    win_inp = dense_bias(ps.reservoir.input_matrix, u, nothing)
    w_state = dense_bias(ps.reservoir.reservoir_matrix, x, bias)
    candidate_h = ctesn!.activation.(win_inp .+ w_state)

    lc = ctesn!.leak_coefficient

    dx .= lc .* (candidate_h .- x)
end

# autoregressive ODE
# dx/dt​=λ(f((Wr+Win​Wout​)x+b)−x)

# non uniform in time input data is not handled


# ======================================================================================================================

function initialparameters(rng::AbstractRNG, ctesn::AbstractReservoirRecurrentCell)
    ps = (
        input_matrix = ctesn.init_input(rng, ctesn.out_dims, ctesn.in_dims),
        reservoir_matrix = ctesn.init_reservoir(rng, ctesn.out_dims, ctesn.out_dims),
    )
    if has_bias(ctesn)
        ps = merge(ps, (bias = ctesn.init_bias(rng, ctesn.out_dims),))
    end
    return ps
end

function initialstates(rng::AbstractRNG, ctesn::AbstractReservoirRecurrentCell)
    return (rng = sample_replicate(rng),)
end

function init_hidden_states(rng::AbstractRNG, cell::AbstractReservoirRecurrentCell, inp::AbstractArray)
    return (init_hidden_state(rng, cell, inp),)
end

function (ctesn::AbstractReservoirRecurrentCell)(inp::AbstractArray, ps, st::NamedTuple)
    rng = replicate(st.rng)
    hidden_state = init_hidden_state(rng, ctesn, inp)
    return ctesn((inp, (hidden_state,)), ps, merge(st, (; rng)))
end

# ======================================================================================================================
