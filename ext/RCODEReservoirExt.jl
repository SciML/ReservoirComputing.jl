module RCODEReservoirExt

using DataInterpolations: ConstantInterpolation
using LinearAlgebra: mul!
using LuxCore: apply, replicate
using Random: AbstractRNG, randexp
using SciMLBase: FullSpecialize, ODEFunction, ODEProblem, init, reinit!, remake, solve,
    solve!, NullParameters, VectorContinuousCallback, DiscreteCallback, CallbackSet,
    successful_retcode
using Static: known, static
using WeightInitializers: randn32, zeros32

using ReservoirComputing: ReservoirComputing,
    AbstractReservoirComputer,
    AbstractSciMLProblemReservoir,
    ContinuousESN,
    ContinuousESNCell,
    CurrentInjection,
    ExponentialSpikeFilter,
    LIFNeuron,
    LinearReadout,
    LSM,
    LSMCell,
    MembraneVoltageFeature,
    PoissonRateEncoder,
    SpikeCountFeatures,
    TerminalStateSampling,
    __feature_dim,
    __supports_ar,
    _reservoir_jac_prototype,
    _wrap_layers,
    collectstates,
    dale_sparse,
    rand_sparse,
    scaled_rand
import ReservoirComputing: _collectstates, _predict

_to_namedtuple(prob_p::NamedTuple) = prob_p
_to_namedtuple(::NullParameters) = NamedTuple()
_to_namedtuple(::Nothing) = NamedTuple()
function _to_namedtuple(prob_p)
    return throw(
        ArgumentError(
            "SciMLProblemReservoir requires `prob.p` to be a NamedTuple, " *
                "`nothing`, or `SciMLBase.NullParameters()`, got $(typeof(prob_p)). " *
                "Wrap your parameters in a NamedTuple — the extension injects " *
                "`input` on top before calling `solve`."
        )
    )
end

function _build_solve_params(prob_p, ps_reservoir, input_interp)
    base = _to_namedtuple(prob_p)
    # `:input` is the reserved key the extension injects so the user's ODE
    # right-hand side can read `p.input(t)`. A pre-existing `:input` field
    # in either `prob.p` or `ps.reservoir` would be silently shadowed below,
    # which is exactly the silent-failure surface we want to avoid.
    if haskey(base, :input)
        throw(
            ArgumentError(
                "`prob.p` already contains an `:input` field. The continuous " *
                    "reservoir extension reserves that name for the interpolated " *
                    "input signal it injects at solve time. Rename the field in " *
                    "your ODE problem before constructing the reservoir."
            )
        )
    end
    if !isempty(ps_reservoir) && haskey(ps_reservoir, :input)
        throw(
            ArgumentError(
                "`ps.reservoir` already contains an `:input` field. That name is " *
                    "reserved for the extension's interpolated input signal — " *
                    "rename your reservoir parameter."
            )
        )
    end
    merged = isempty(ps_reservoir) ? base : merge(base, ps_reservoir)
    return merge(merged, (input = input_interp,))
end

"""
    ZeroOrderHoldInterp(data, ts)

Piecewise-constant input signal for the continuous reservoir. Holds a
`data::AbstractMatrix` of shape `(channels, T)` alongside the matching
time-stamp vector `ts`. For `t` in window `k` (i.e. `ts[k] ≤ t < ts[k+1]`)
the call returns `view(data, :, k)`; out-of-range times clamp to the
nearest endpoint.

We pick zero-order hold (ZOH) over linear interpolation deliberately:
under linear interpolation the reservoir state at sample time `sample_ts[k]`
depends on both `data[:, k]` and `data[:, k+1]` for any non-Euler solver,
which is a one-step lookahead that contradicts the documented "state
after processing input k" semantics. With ZOH, `data[:, k]` is the only
input column that influences `states[:, k]`, regardless of solver — and
the autoregressive `predict` path already uses ZOH for its per-window
input function, so the two paths now use the same scheme.

Why not `DataInterpolations.ConstantInterpolation`: matrix-valued `u`
has no `_integral` method, so `cache_parameters=true` fails at
construction; the default `cache_parameters=false` leaves unused cache
fields typed as `Vector{Union{}}`, which SciMLBase's dual-eltype probing
crashes on while preparing `solve` (observed on DataInterpolations v8 /
SciMLBase v2, 2026-06). A bespoke struct with concrete fields and a
view-returning call sidesteps both paths and is allocation-free in the
ODE hot path. Revisit if/when DataInterpolations supports matrix-`u`
non-cached construction without the bottom-type fallout.
"""
struct ZeroOrderHoldInterp{D <: AbstractMatrix, T <: AbstractVector}
    data::D
    ts::T
end

function (interp::ZeroOrderHoldInterp)(t)
    ts = interp.ts
    n_samples = length(ts)
    t < ts[1] && return view(interp.data, :, 1)
    t ≥ ts[end] && return view(interp.data, :, n_samples)
    window_idx = searchsortedlast(ts, t)
    return view(interp.data, :, clamp(window_idx, 1, n_samples))
end

function _make_input_fn(data::AbstractMatrix, ts::AbstractVector)
    return ZeroOrderHoldInterp(data, ts)
end

function _make_const_input_fn(u_vec::AbstractVector, t_lo, t_hi)
    return ConstantInterpolation([u_vec, u_vec], [t_lo, t_hi]; cache_parameters = true)
end

mutable struct ConstantInputWindow{T <: AbstractVector}
    u_vec::T
end

(input::ConstantInputWindow)(t) = input.u_vec

function _sample(::TerminalStateSampling, sol)
    return reduce(hcat, sol.u)
end

function _apply_modifiers_continuous(
        modifiers::Tuple, states_matrix::AbstractMatrix, ps_mods, st_mods
    )
    isempty(modifiers) && return states_matrix, st_mods
    n_samples = size(states_matrix, 2)
    src_cols = eachcol(states_matrix)

    first_col, new_st = ReservoirComputing._apply_seq(
        modifiers, first(src_cols), ps_mods, st_mods
    )
    # `similar(first_col, ...)` — not `similar(states_matrix, ...)` — so the
    # output matrix takes the modifier output's eltype. If a modifier
    # promotes/demotes (e.g. Float32 → Float64), we want that to surface,
    # not be silently truncated back to the reservoir state's eltype.
    output = similar(first_col, length(first_col), n_samples)
    output[:, 1] .= first_col
    for (idx, src_col) in Iterators.drop(enumerate(src_cols), 1)
        modified_col, new_st = ReservoirComputing._apply_seq(
            modifiers, src_col, ps_mods, new_st
        )
        output[:, idx] .= modified_col
    end
    return output, new_st
end

function _collectstates(
        res::AbstractSciMLProblemReservoir,
        rc::AbstractReservoirComputer,
        data::AbstractMatrix,
        ps::NamedTuple,
        st::NamedTuple
    )
    n_samples = size(data, 2)
    n_samples ≥ 2 || throw(
        ArgumentError(
            "SciMLProblemReservoir collectstates needs at least 2 input " *
                "columns to define a time grid; got $n_samples."
        )
    )

    t0, t1 = res.tspan
    t1 > t0 || throw(
        ArgumentError(
            "SciMLProblemReservoir requires `tspan[2] > tspan[1]`, got " *
                "tspan = ($t0, $t1). Continuous integration is only defined " *
                "over a strictly positive interval."
        )
    )

    Δt = (t1 - t0) / n_samples
    input_ts = collect(range(t0, t1 - Δt; length = n_samples))
    sample_ts = collect(range(t0 + Δt, t1; length = n_samples))

    input_interp = _make_input_fn(data, input_ts)
    solve_p = _build_solve_params(res.prob.p, ps.reservoir, input_interp)

    prob_remade = remake(res.prob; tspan = res.tspan, p = solve_p)

    sol = solve(
        prob_remade, res.args...;
        saveat = sample_ts,
        save_everystep = false,
        dense = false,
        res.kwargs...
    )

    raw_states = _sample(res.sampler, sol)
    modified_states, st_mods = _apply_modifiers_continuous(
        rc.state_modifiers, raw_states, ps.state_modifiers, st.state_modifiers
    )

    newst = (
        reservoir = st.reservoir,
        state_modifiers = st_mods,
        readout = st.readout,
    )
    return modified_states, newst
end

function _predict(
        ::AbstractSciMLProblemReservoir,
        rc::AbstractReservoirComputer,
        data::AbstractMatrix,
        ps::NamedTuple,
        st::NamedTuple
    )
    states, new_st = collectstates(rc, data, ps, st)
    n_samples = size(states, 2)
    st_ro = new_st.readout
    state_cols = eachcol(states)
    first_output, st_ro = apply(rc.readout, first(state_cols), ps.readout, st_ro)
    outputs = similar(first_output, size(first_output, 1), n_samples)
    outputs[:, 1] .= first_output
    for (idx, state_col) in Iterators.drop(enumerate(state_cols), 1)
        current_output, st_ro = apply(rc.readout, state_col, ps.readout, st_ro)
        outputs[:, idx] .= current_output
    end
    return outputs, merge(new_st, (readout = st_ro,))
end

function _predict(
        res::AbstractSciMLProblemReservoir,
        rc::AbstractReservoirComputer,
        steps::Integer,
        ps::NamedTuple,
        st::NamedTuple;
        initialdata::AbstractVector
    )
    steps ≥ 1 || throw(ArgumentError("steps must be ≥ 1, got $steps"))

    t0, t1 = res.tspan
    t1 > t0 || throw(
        ArgumentError(
            "Autoregressive predict requires `tspan[2] > tspan[1]`, got " *
                "tspan = ($t0, $t1)."
        )
    )
    ts = collect(range(t0, t1; length = steps + 1))
    window_starts = @view ts[1:(end - 1)]
    window_ends = @view ts[2:end]

    # Preserve `u0`'s original type — `collect` would degrade `SVector` /
    # `ComponentArray` / scalar states into a plain `Vector` and either
    # error (no `collect(::Number)` method) or silently flatten the
    # user's chosen representation. We only ever read `current_state`,
    # never mutate it in place, so a direct reference is safe.
    current_state = res.prob.u0
    current_input = initialdata

    st_mods = st.state_modifiers
    st_ro = st.readout

    input_fn = ConstantInputWindow(current_input)
    solve_p = _build_solve_params(res.prob.p, ps.reservoir, input_fn)
    sub_prob = remake(
        res.prob;
        tspan = (window_starts[1], window_ends[1]),
        p = solve_p,
        u0 = current_state,
    )
    integrator = init(
        sub_prob, res.args...;
        save_everystep = false, dense = false,
        save_start = false, save_end = true,
        res.kwargs...,
    )

    # `outputs` is allocated *after* the first readout call so its element
    # type and row count come from `apply(rc.readout, …)` rather than
    # `initialdata`. Otherwise a readout returning a different eltype
    # (e.g. Float64 vs the Float32 input) would force a silent
    # conversion at the column assignment.
    local outputs
    for (step_idx, (t_lo, t_hi)) in enumerate(zip(window_starts, window_ends))
        input_fn.u_vec = convert(typeof(input_fn.u_vec), current_input)
        reinit!(
            integrator, current_state;
            t0 = t_lo,
            tf = t_hi,
            erase_sol = true,
            reset_dt = true,
        )
        solve!(integrator)
        current_state = integrator.u

        if !isempty(rc.state_modifiers)
            state_after_mods, st_mods = ReservoirComputing._apply_seq(
                rc.state_modifiers, current_state, ps.state_modifiers, st_mods
            )
        else
            state_after_mods = current_state
        end

        current_output, st_ro = apply(rc.readout, state_after_mods, ps.readout, st_ro)
        if step_idx == 1
            outputs = similar(current_output, length(current_output), steps)
        end
        outputs[:, step_idx] .= current_output
        current_input = current_output
    end

    newst = (
        reservoir = st.reservoir,
        state_modifiers = st_mods,
        readout = st_ro,
    )
    return outputs, newst
end

function ReservoirComputing.ContinuousESN(
        in_dims::Integer, res_dims::Integer, out_dims::Integer,
        activation, tspan, args...;
        use_bias::Bool = false,
        init_bias = zeros32,
        init_reservoir = rand_sparse,
        init_input = scaled_rand,
        init_state = randn32,
        equations = ReservoirComputing._continuous_esn_rhs!,
        state_modifiers = (),
        readout_activation = identity,
        use_jac_prototype::Bool = false,
        kwargs...
    )
    in_dims > 0 || throw(ArgumentError("in_dims must be positive, got $in_dims"))
    res_dims > 0 || throw(ArgumentError("res_dims must be positive, got $res_dims"))
    out_dims > 0 || throw(ArgumentError("out_dims must be positive, got $out_dims"))
    length(tspan) == 2 || throw(
        ArgumentError(
            "tspan must be a length-2 tuple/pair (t0, t1), got length $(length(tspan))"
        )
    )
    (isfinite(tspan[1]) && isfinite(tspan[2])) || throw(
        ArgumentError("tspan endpoints must be finite, got $tspan")
    )
    tspan[2] > tspan[1] || throw(
        ArgumentError(
            "ContinuousESN requires `tspan[2] > tspan[1]`, got tspan = $tspan"
        )
    )
    ReservoirComputing._check_protected_kwargs(kwargs)

    cell = ContinuousESNCell(
        activation, in_dims, res_dims,
        init_bias, init_reservoir, init_input, init_state,
        static(use_bias),
        equations, tspan, args, kwargs,
        static(use_jac_prototype)
    )

    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
        Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)

    readout = LinearReadout(res_dims => out_dims, readout_activation)
    return ContinuousESN(cell, mods, readout)
end

function ReservoirComputing.ContinuousESN(
        in_dims::Integer, res_dims::Integer, out_dims::Integer,
        tspan::Union{Tuple, Pair}, args...; kwargs...
    )
    return ContinuousESN(in_dims, res_dims, out_dims, tanh, tspan, args...; kwargs...)
end

function _collectstates(
        cell::ContinuousESNCell,
        rc::AbstractReservoirComputer,
        data::AbstractMatrix,
        ps::NamedTuple,
        st::NamedTuple
    )
    n_samples = size(data, 2)
    n_samples ≥ 2 || throw(
        ArgumentError(
            "ContinuousESN collectstates needs at least 2 input columns " *
                "to define a time grid; got $n_samples."
        )
    )

    t0, t1 = cell.tspan
    t1 > t0 || throw(
        ArgumentError(
            "ContinuousESN requires `tspan[2] > tspan[1]`, got tspan = ($t0, $t1)."
        )
    )

    Δt = (t1 - t0) / n_samples
    input_ts = collect(range(t0, t1 - Δt; length = n_samples))
    sample_ts = collect(range(t0 + Δt, t1; length = n_samples))

    input_interp = _make_input_fn(data, input_ts)
    solve_p = _build_solve_params(nothing, ps.reservoir, input_interp)

    # `u0` element type follows `ps.reservoir.input_matrix` so the solver
    # state, the parameter pack, and the input signal share a numeric
    # type. The user controls eltype through the `init_*` initialisers.
    u0 = zeros(eltype(ps.reservoir.input_matrix), cell.out_dims)
    jac_prototype = known(cell.use_jac_prototype) ?
        _reservoir_jac_prototype(cell.equations, ps.reservoir.reservoir_matrix) :
        nothing
    ode_fn = ODEFunction{true, FullSpecialize}(cell.equations; jac_prototype = jac_prototype)
    prob = ODEProblem{true, FullSpecialize}(ode_fn, u0, cell.tspan, solve_p)

    sol = solve(
        prob, cell.args...;
        saveat = sample_ts,
        save_everystep = false,
        dense = false,
        cell.kwargs...
    )

    raw_states = _sample(TerminalStateSampling(), sol)
    modified_states, st_mods = _apply_modifiers_continuous(
        rc.state_modifiers, raw_states, ps.state_modifiers, st.state_modifiers
    )

    newst = (
        reservoir = st.reservoir,
        state_modifiers = st_mods,
        readout = st.readout,
    )
    return modified_states, newst
end

function _predict(
        cell::ContinuousESNCell,
        rc::AbstractReservoirComputer,
        steps::Integer,
        ps::NamedTuple,
        st::NamedTuple;
        initialdata::AbstractVector
    )
    steps ≥ 1 || throw(ArgumentError("steps must be ≥ 1, got $steps"))

    t0, t1 = cell.tspan
    t1 > t0 || throw(
        ArgumentError(
            "Autoregressive predict requires `tspan[2] > tspan[1]`, got " *
                "tspan = ($t0, $t1)."
        )
    )
    ts = collect(range(t0, t1; length = steps + 1))
    window_starts = @view ts[1:(end - 1)]
    window_ends = @view ts[2:end]

    current_state = zeros(eltype(ps.reservoir.input_matrix), cell.out_dims)
    current_input = initialdata

    st_mods = st.state_modifiers
    st_ro = st.readout

    jac_prototype = known(cell.use_jac_prototype) ?
        _reservoir_jac_prototype(cell.equations, ps.reservoir.reservoir_matrix) :
        nothing
    ode_fn = ODEFunction{true, FullSpecialize}(cell.equations; jac_prototype = jac_prototype)

    input_fn = ConstantInputWindow(current_input)
    solve_p = _build_solve_params(nothing, ps.reservoir, input_fn)
    sub_prob = ODEProblem{true, FullSpecialize}(
        ode_fn, current_state, (window_starts[1], window_ends[1]), solve_p
    )
    integrator = init(
        sub_prob, cell.args...;
        save_everystep = false, dense = false,
        save_start = false, save_end = true,
        cell.kwargs...,
    )

    local outputs
    for (step_idx, (t_lo, t_hi)) in enumerate(zip(window_starts, window_ends))
        input_fn.u_vec = convert(typeof(input_fn.u_vec), current_input)
        reinit!(
            integrator, current_state;
            t0 = t_lo,
            tf = t_hi,
            erase_sol = true,
            reset_dt = true,
        )
        solve!(integrator)
        current_state = integrator.u

        if !isempty(rc.state_modifiers)
            state_after_mods, st_mods = ReservoirComputing._apply_seq(
                rc.state_modifiers, current_state, ps.state_modifiers, st_mods
            )
        else
            state_after_mods = current_state
        end

        current_output, st_ro = apply(rc.readout, state_after_mods, ps.readout, st_ro)
        if step_idx == 1
            outputs = similar(current_output, length(current_output), steps)
        end
        outputs[:, step_idx] .= current_output
        current_input = current_output
    end

    newst = (
        reservoir = st.reservoir,
        state_modifiers = st_mods,
        readout = st_ro,
    )
    return outputs, newst
end

function ReservoirComputing.LSM(
        in_dims::Integer, res_dims::Integer, out_dims::Integer,
        tspan, args...;
        neuron = LIFNeuron(),
        encoder = CurrentInjection(),
        feature_map = ExponentialSpikeFilter(),
        use_bias::Bool = false,
        init_bias = zeros32,
        init_reservoir = dale_sparse,
        init_input = scaled_rand,
        init_state = zeros32,
        state_modifiers = (),
        readout_activation = identity,
        kwargs...
    )
    in_dims > 0 || throw(ArgumentError("in_dims must be positive, got $in_dims"))
    res_dims > 0 || throw(ArgumentError("res_dims must be positive, got $res_dims"))
    out_dims > 0 || throw(ArgumentError("out_dims must be positive, got $out_dims"))
    ReservoirComputing.__check_lsm_tspan(tspan)
    ReservoirComputing._check_protected_kwargs(kwargs)
    haskey(kwargs, :callback) && throw(
        ArgumentError("LSM owns the solve callback; drop `callback` from kwargs.")
    )

    cell = LSMCell(
        neuron, encoder, feature_map, in_dims, res_dims,
        init_bias, init_reservoir, init_input, init_state,
        static(use_bias), tspan, args, kwargs
    )
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
        Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    readout = LinearReadout(
        __feature_dim(feature_map, res_dims) => out_dims, readout_activation
    )
    return LSM(cell, mods, readout)
end

function __lsm_rhs!(du, u, p, t)
    n_units = p.n_units
    neuron = p.neuron
    T = eltype(u)
    tau_m = T(neuron.tau_m)
    tau_syn = T(neuron.tau_syn)
    r_m = T(neuron.r_m)
    v_rest = T(neuron.v_rest)
    mul!(p.I_ext, p.input_matrix, p.input(t))
    haskey(p, :bias) && (p.I_ext .+= p.bias)
    @inbounds for unit in 1:n_units
        du[n_units + unit] = -u[n_units + unit] / tau_syn
        du[unit] = t < p.ref_until[unit] ? zero(T) :
            (-(u[unit] - v_rest) + r_m * (u[n_units + unit] + p.I_ext[unit])) / tau_m
    end
    return nothing
end

function __lsm_fire!(integrator, unit::Integer)
    p = integrator.p
    (unit < 1 || unit > p.n_units || integrator.t <= p.ref_until[unit]) && return nothing
    n_units = p.n_units
    neuron = p.neuron
    T = eltype(integrator.u)
    t = integrator.t
    integrator.u[unit] = T(neuron.v_reset)
    p.ref_until[unit] = t + T(neuron.tau_ref)
    W = p.reservoir_matrix
    @inbounds for post in 1:n_units
        integrator.u[n_units + post] += W[post, unit]
    end
    push!(p.spike_t, t)
    push!(p.spike_i, Int(unit))
    return nothing
end

function __lsm_affect!(integrator, event_idx)
    if event_idx isa Integer
        __lsm_fire!(integrator, event_idx)
    else
        for unit in event_idx
            __lsm_fire!(integrator, unit)
        end
    end
    return nothing
end

function __lsm_condition!(out, u, t, integrator)
    n_units = integrator.p.n_units
    v_th = eltype(u)(integrator.p.neuron.v_th)
    @inbounds for unit in 1:n_units
        out[unit] = u[unit] - v_th
    end
    return nothing
end

__lsm_spike_cb(n_units::Int) = VectorContinuousCallback(
    __lsm_condition!, __lsm_affect!, n_units; save_positions = (false, false)
)

function __lsm_features!(
        features, ::SpikeCountFeatures, spike_t, spike_i, sample_ts, sol, n_units
    )
    fill!(features, 0)
    event = 1
    @inbounds for sample in eachindex(sample_ts)
        t_hi = sample_ts[sample]
        while event <= length(spike_t) && spike_t[event] <= t_hi
            features[spike_i[event], sample] += 1
            event += 1
        end
    end
    return features
end

function __advance_exp_filter!(
        filter_state::AbstractVector{T}, t_last::T, spike_t, spike_i, t_hi, filter_tau
    ) where {T}
    inv_tau = inv(T(filter_tau))
    @inbounds for event in eachindex(spike_t)
        t_spike = T(spike_t[event])
        isfinite(t_last) && (filter_state .*= exp((t_last - t_spike) * inv_tau))
        filter_state[spike_i[event]] += one(T)
        t_last = t_spike
    end
    t_sample = T(t_hi)
    if isfinite(t_last)
        filter_state .*= exp((t_last - t_sample) * inv_tau)
        t_last = t_sample
    end
    return t_last
end

function __lsm_features!(
        features::AbstractMatrix{T}, fmap::ExponentialSpikeFilter,
        spike_t, spike_i, sample_ts, sol, n_units
    ) where {T}
    filter_state = zeros(T, size(features, 1))
    t_last = typemin(T)
    event = 1
    inv_tau = inv(T(fmap.filter_tau))
    @inbounds for sample in eachindex(sample_ts)
        t_sample = T(sample_ts[sample])
        while event <= length(spike_t) && spike_t[event] <= t_sample
            t_spike = T(spike_t[event])
            isfinite(t_last) && (filter_state .*= exp((t_last - t_spike) * inv_tau))
            filter_state[spike_i[event]] += one(T)
            t_last = t_spike
            event += 1
        end
        if isfinite(t_last)
            filter_state .*= exp((t_last - t_sample) * inv_tau)
            t_last = t_sample
        end
        features[:, sample] .= filter_state
    end
    return features
end

function __lsm_features!(
        features, ::MembraneVoltageFeature, spike_t, spike_i, sample_ts, sol, n_units
    )
    @inbounds for sample in eachindex(sol.u)
        copyto!(view(features, :, sample), view(sol.u[sample], 1:n_units))
    end
    return features
end

function __poisson_events(
        rng::AbstractRNG, data::AbstractMatrix{T}, input_ts, t1, rate_scale
    ) where {T}
    in_dims, n_samples = size(data)
    times = T[]
    channels = Int[]
    scale = T(rate_scale)
    for sample in 1:n_samples
        t0 = T(input_ts[sample])
        t_hi = sample < n_samples ? T(input_ts[sample + 1]) : T(t1)
        for channel in 1:in_dims
            rate = scale * max(data[channel, sample], zero(T))
            rate <= 0 && continue
            t = t0
            while true
                t += T(randexp(rng)) / rate
                t >= t_hi && break
                push!(times, t)
                push!(channels, channel)
            end
        end
    end
    order = sortperm(times)
    return times[order], channels[order]
end

function __poisson_callback!(times::Vector{T}, channels::Vector{Int}, synaptic_weight) where {T}
    next_event = Ref(1)
    cond = let times = times, next_event = next_event
        function (u, t, integrator)
            return next_event[] <= length(times) && t >= times[next_event[]]
        end
    end
    aff! = let times = times, channels = channels, next_event = next_event,
            synaptic_weight = synaptic_weight
        function (integrator)
            p = integrator.p
            n_units = p.n_units
            weight = eltype(integrator.u)(synaptic_weight)
            while next_event[] <= length(times) && integrator.t >= times[next_event[]]
                channel = channels[next_event[]]
                @inbounds for unit in 1:n_units
                    integrator.u[n_units + unit] += weight * p.input_matrix[unit, channel]
                end
                next_event[] += 1
            end
            return nothing
        end
    end
    return DiscreteCallback(cond, aff!; save_positions = (false, false))
end

function __lsm_pack(cell::LSMCell, ps_res, input_fn, n_units::Int, ::Type{T}) where {T}
    base = (
        n_units = n_units,
        neuron = cell.neuron,
        input_matrix = ps_res.input_matrix,
        reservoir_matrix = ps_res.reservoir_matrix,
        input = input_fn,
        I_ext = zeros(T, n_units),
        ref_until = fill(typemin(T), n_units),
        spike_t = T[],
        spike_i = Int[],
    )
    return haskey(ps_res, :bias) ? merge(base, (bias = ps_res.bias,)) : base
end

function __lsm_u0(cell::LSMCell, st_res, n_units::Int, ::Type{T}) where {T}
    rng = replicate(st_res.rng)
    u0 = zeros(T, 2 * n_units)
    copyto!(view(u0, 1:n_units), vec(cell.init_state(rng, n_units, 1)))
    return u0, merge(st_res, (rng = rng,))
end

function __lsm_callbacks(cell::LSMCell, n_units::Int, data, input_ts, t1, st_enc)
    spike_cb = __lsm_spike_cb(n_units)
    if cell.encoder isa CurrentInjection
        return CallbackSet(spike_cb), _make_input_fn(data, input_ts), st_enc, nothing
    elseif cell.encoder isa PoissonRateEncoder
        enc = cell.encoder
        rng = copy(st_enc.rng)
        times, channels = __poisson_events(
            rng, data, input_ts, t1, enc.rate_scale
        )
        zoh = _make_input_fn(zeros(eltype(data), size(data)), input_ts)
        pcb = __poisson_callback!(times, channels, enc.synaptic_weight)
        return CallbackSet(spike_cb, pcb), zoh, (rng = rng,), times
    end
    return throw(ArgumentError("unsupported encoder $(typeof(cell.encoder))"))
end

function _collectstates(
        cell::LSMCell,
        rc::AbstractReservoirComputer,
        data::AbstractMatrix,
        ps::NamedTuple,
        st::NamedTuple
    )
    n_samples = size(data, 2)
    n_samples ≥ 2 || throw(
        ArgumentError("LSM collectstates needs at least 2 input columns; got $n_samples.")
    )
    t0, t1 = cell.tspan
    t1 > t0 || throw(
        ArgumentError("LSM requires `tspan[2] > tspan[1]`, got tspan = ($t0, $t1).")
    )

    n_units = cell.out_dims
    T = eltype(ps.reservoir.input_matrix)
    Δt = (t1 - t0) / n_samples
    input_ts = collect(range(t0, t1 - Δt; length = n_samples))
    sample_ts = collect(range(t0 + Δt, t1; length = n_samples))

    cbset, input_fn, st_enc_new, poisson_times = __lsm_callbacks(
        cell, n_units, data, input_ts, t1, st.reservoir.encoder
    )
    solve_p = __lsm_pack(cell, ps.reservoir, input_fn, n_units, T)
    u0, st_res_new = __lsm_u0(cell, st.reservoir, n_units, T)

    tstops = poisson_times === nothing ? sample_ts : sort!(vcat(sample_ts, poisson_times))
    ode_fn = ODEFunction{true, FullSpecialize}(__lsm_rhs!)
    prob = ODEProblem{true, FullSpecialize}(ode_fn, u0, cell.tspan, solve_p)
    sol = solve(
        prob, cell.args...;
        callback = cbset,
        tstops = tstops,
        saveat = sample_ts,
        save_everystep = false,
        dense = false,
        cell.kwargs...
    )
    successful_retcode(sol) || throw(
        ErrorException("LSM solve failed with retcode $(sol.retcode)")
    )

    features = Matrix{T}(undef, __feature_dim(cell.feature_map, n_units), n_samples)
    __lsm_features!(
        features, cell.feature_map, solve_p.spike_t, solve_p.spike_i,
        sample_ts, sol, n_units
    )

    modified_states, st_mods = _apply_modifiers_continuous(
        rc.state_modifiers, features, ps.state_modifiers, st.state_modifiers
    )
    newst = (
        reservoir = merge(st_res_new, (encoder = st_enc_new,)),
        state_modifiers = st_mods,
        readout = st.readout,
    )
    return modified_states, newst
end

function _predict(
        cell::LSMCell,
        rc::AbstractReservoirComputer,
        steps::Integer,
        ps::NamedTuple,
        st::NamedTuple;
        initialdata::AbstractVector
    )
    __supports_ar(cell.feature_map) || throw(
        ArgumentError(
            "$(typeof(cell.feature_map)) does not support autoregressive predict"
        )
    )
    cell.encoder isa PoissonRateEncoder && throw(
        ArgumentError("PoissonRateEncoder does not support autoregressive predict")
    )
    steps ≥ 1 || throw(ArgumentError("steps must be ≥ 1, got $steps"))
    t0, t1 = cell.tspan
    t1 > t0 || throw(
        ArgumentError(
            "Autoregressive predict requires `tspan[2] > tspan[1]`, got " *
                "tspan = ($t0, $t1)."
        )
    )

    n_units = cell.out_dims
    T = eltype(ps.reservoir.input_matrix)
    ts = collect(range(t0, t1; length = steps + 1))
    window_starts = @view ts[1:(end - 1)]
    window_ends = @view ts[2:end]

    input_fn = ConstantInputWindow(convert(Vector{T}, initialdata))
    solve_p = __lsm_pack(cell, ps.reservoir, input_fn, n_units, T)
    current_state, st_res_new = __lsm_u0(cell, st.reservoir, n_units, T)
    current_input = convert(Vector{T}, initialdata)
    st_mods = st.state_modifiers
    st_ro = st.readout
    features_col = zeros(T, __feature_dim(cell.feature_map, n_units))
    filter_state = zeros(T, n_units)
    filter_t = typemin(T)
    filter_tau = cell.feature_map isa ExponentialSpikeFilter ?
        T(cell.feature_map.filter_tau) : zero(T)

    ode_fn = ODEFunction{true, FullSpecialize}(__lsm_rhs!)
    sub_prob = ODEProblem{true, FullSpecialize}(
        ode_fn, current_state, (window_starts[1], window_ends[1]), solve_p
    )
    integrator = init(
        sub_prob, cell.args...;
        callback = __lsm_spike_cb(n_units),
        save_everystep = false, dense = false,
        save_start = false, save_end = true,
        cell.kwargs...,
    )

    local outputs
    for (step_idx, (t_lo, t_hi)) in enumerate(zip(window_starts, window_ends))
        input_fn.u_vec = convert(typeof(input_fn.u_vec), current_input)
        empty!(solve_p.spike_t)
        empty!(solve_p.spike_i)
        reinit!(
            integrator, current_state;
            t0 = t_lo, tf = t_hi, erase_sol = true, reset_dt = true,
        )
        solve!(integrator)
        successful_retcode(integrator.sol) || throw(
            ErrorException("LSM AR step failed with retcode $(integrator.sol.retcode)")
        )
        current_state = integrator.u

        if cell.feature_map isa MembraneVoltageFeature
            copyto!(features_col, view(current_state, 1:n_units))
        else
            filter_t = __advance_exp_filter!(
                filter_state, filter_t, solve_p.spike_t, solve_p.spike_i,
                t_hi, filter_tau
            )
            copyto!(features_col, filter_state)
        end

        if !isempty(rc.state_modifiers)
            state_after_mods, st_mods = ReservoirComputing._apply_seq(
                rc.state_modifiers, features_col, ps.state_modifiers, st_mods
            )
        else
            state_after_mods = features_col
        end
        current_output, st_ro = apply(rc.readout, state_after_mods, ps.readout, st_ro)
        if step_idx == 1
            outputs = similar(current_output, length(current_output), steps)
        end
        outputs[:, step_idx] .= current_output
        current_input = current_output
    end

    newst = (
        reservoir = st_res_new,
        state_modifiers = st_mods,
        readout = st_ro,
    )
    return outputs, newst
end

end # module
