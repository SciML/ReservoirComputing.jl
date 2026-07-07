module RCODEReservoirExt

using DataInterpolations: ConstantInterpolation
using LinearAlgebra: mul!
using LuxCore: apply
using Random: AbstractRNG
using SciMLBase: ODEProblem, remake, solve, NullParameters

using ReservoirComputing: ReservoirComputing,
    AbstractReservoirComputer,
    AbstractSampler,
    AbstractSciMLProblemReservoir,
    ContinuousESN,
    ContinuousESNCell,
    LinearReadout,
    TerminalStateSampling,
    _wrap_layers,
    collectstates,
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
        rc.states_modifiers, raw_states, ps.states_modifiers, st.states_modifiers
    )

    newst = (
        reservoir = st.reservoir,
        states_modifiers = st_mods,
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

    st_mods = st.states_modifiers
    st_ro = st.readout

    # `outputs` is allocated *after* the first readout call so its element
    # type and row count come from `apply(rc.readout, …)` rather than
    # `initialdata`. Otherwise a readout returning a different eltype
    # (e.g. Float64 vs the Float32 input) would force a silent
    # conversion at the column assignment.
    local outputs
    for (step_idx, (t_lo, t_hi)) in enumerate(zip(window_starts, window_ends))
        input_fn = _make_const_input_fn(current_input, t_lo, t_hi)
        solve_p = _build_solve_params(res.prob.p, ps.reservoir, input_fn)
        sub_prob = remake(
            res.prob;
            tspan = (t_lo, t_hi),
            p = solve_p,
            u0 = current_state
        )
        sol = solve(
            sub_prob, res.args...;
            saveat = [t_hi],
            save_everystep = false,
            dense = false,
            res.kwargs...
        )
        current_state = sol.u[end]

        if !isempty(rc.states_modifiers)
            state_after_mods, st_mods = ReservoirComputing._apply_seq(
                rc.states_modifiers, current_state, ps.states_modifiers, st_mods
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
        states_modifiers = st_mods,
        readout = st_ro,
    )
    return outputs, newst
end

function ReservoirComputing.ContinuousESN(
        in_dims::Integer, res_dims::Integer, out_dims::Integer,
        activation, tspan, args...;
        use_bias::Bool = false,
        init_bias = ReservoirComputing.zeros32,
        init_reservoir = rand_sparse,
        init_input = scaled_rand,
        init_state = ReservoirComputing.randn32,
        equations = ReservoirComputing._continuous_esn_rhs!,
        state_modifiers = (),
        readout_activation = identity,
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
        ReservoirComputing.static(use_bias),
        equations, tspan, args, kwargs
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
    prob = ODEProblem(cell.equations, u0, cell.tspan, solve_p)

    sol = solve(
        prob, cell.args...;
        saveat = sample_ts,
        save_everystep = false,
        dense = false,
        cell.kwargs...
    )

    raw_states = _sample(TerminalStateSampling(), sol)
    modified_states, st_mods = _apply_modifiers_continuous(
        rc.states_modifiers, raw_states, ps.states_modifiers, st.states_modifiers
    )

    newst = (
        reservoir = st.reservoir,
        states_modifiers = st_mods,
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

    st_mods = st.states_modifiers
    st_ro = st.readout

    local outputs
    for (step_idx, (t_lo, t_hi)) in enumerate(zip(window_starts, window_ends))
        input_fn = _make_const_input_fn(current_input, t_lo, t_hi)
        solve_p = _build_solve_params(nothing, ps.reservoir, input_fn)
        sub_prob = ODEProblem(cell.equations, current_state, (t_lo, t_hi), solve_p)
        sol = solve(
            sub_prob, cell.args...;
            saveat = [t_hi],
            save_everystep = false,
            dense = false,
            cell.kwargs...
        )
        current_state = sol.u[end]

        if !isempty(rc.states_modifiers)
            state_after_mods, st_mods = ReservoirComputing._apply_seq(
                rc.states_modifiers, current_state, ps.states_modifiers, st_mods
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
        states_modifiers = st_mods,
        readout = st_ro,
    )
    return outputs, newst
end

end # module
