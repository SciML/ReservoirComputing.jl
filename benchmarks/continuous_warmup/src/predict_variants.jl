# Experimental AR predict variants — mirror extension logic without
# changing package API. Used only for investigation.

using ReservoirComputing  # reexports LuxCore.apply
using SciMLBase
using OrdinaryDiffEq

function _require_ext()
    ext = Base.get_extension(ReservoirComputing, :RCODEReservoirExt)
    ext === nothing && error(
        "RCODEReservoirExt not loaded. Import SciMLBase, DataInterpolations, " *
            "and an ODE solver package first."
    )
    return ext
end

"""
    terminal_state_from_collect(rc, data, ps, st)

Teacher-forced `collectstates` then last state column (raw continuous
state after modifiers are applied — same matrix the readout sees).

Note: for continuous models this is the **sampled** reservoir feature
after modifiers, not necessarily the raw ODE `u`. For AR seeding we need
the raw ODE state. Prefer [`raw_terminal_ode_state`](@ref) when seeding `u0`.
"""
function terminal_state_from_collect(rc, data::AbstractMatrix, ps, st)
    states, new_st = collectstates(rc, data, ps, st)
    return states[:, end], new_st
end

"""
    raw_terminal_ode_state(rc, data, ps, st)

Re-run the continuous collect path and return the last **unmodified**
ODE state (before state_modifiers). This is the correct seed for `u0`.
"""
function raw_terminal_ode_state(rc, data::AbstractMatrix, ps, st)
    ext = _require_ext()
    res = rc.reservoir

    if res isa ContinuousESNCell
        return _raw_terminal_continuous_esn(ext, res, rc, data, ps, st)
    elseif res isa AbstractSciMLProblemReservoir
        return _raw_terminal_sciml(ext, res, rc, data, ps, st)
    else
        # Discrete: collectstates last column is the recurrent carry
        # after modifiers — for ESN without modifiers this is fine.
        states, _ = collectstates(rc, data, ps, st)
        return copy(states[:, end])
    end
end

function _raw_terminal_continuous_esn(ext, cell::ContinuousESNCell, rc, data, ps, st)
    n_samples = size(data, 2)
    # Unit window width (Δt = 1), independent of the model’s train/predict
    # `tspan`. Using `cell.tspan` with a short warmup of length K would stretch
    # those K samples across the full train interval and change the ODE pacing.
    t0, t1 = 0.0, Float64(n_samples)
    Δt = (t1 - t0) / n_samples  # == 1
    input_ts = collect(range(t0, t1 - Δt; length = n_samples))
    sample_ts = collect(range(t0 + Δt, t1; length = n_samples))

    input_interp = ext._make_input_fn(data, input_ts)
    solve_p = ext._build_solve_params(nothing, ps.reservoir, input_interp)
    u0 = zeros(eltype(ps.reservoir.input_matrix), cell.out_dims)
    prob = ODEProblem(cell.equations, u0, (t0, t1), solve_p)
    sol = solve(
        prob, cell.args...;
        saveat = sample_ts,
        save_everystep = false,
        dense = false,
        cell.kwargs...,
    )
    return copy(sol.u[end])
end

function _raw_terminal_sciml(ext, res, rc, data, ps, st)
    n_samples = size(data, 2)
    t0, t1 = 0.0, Float64(n_samples)
    Δt = (t1 - t0) / n_samples
    input_ts = collect(range(t0, t1 - Δt; length = n_samples))
    sample_ts = collect(range(t0 + Δt, t1; length = n_samples))

    input_interp = ext._make_input_fn(data, input_ts)
    solve_p = ext._build_solve_params(res.prob.p, ps.reservoir, input_interp)
    remade = remake(res.prob; tspan = (t0, t1), p = solve_p)
    sol = solve(
        remade, res.args...;
        saveat = sample_ts,
        save_everystep = false,
        dense = false,
        res.kwargs...,
    )
    return copy(sol.u[end])
end

"""
    predict_ar_seeded(rc, steps, ps, st; initialdata, initial_state)

Autoregressive rollout with an explicit reservoir seed `initial_state`.
Mirrors `RCODEReservoirExt` continuous AR `predict` for both
`ContinuousESNCell` and generic `SciMLProblemReservoir`.
"""
function predict_ar_seeded(
        rc,
        steps::Integer,
        ps,
        st;
        initialdata::AbstractVector,
        initial_state::AbstractVector,
    )
    steps ≥ 1 || throw(ArgumentError("steps must be ≥ 1, got $steps"))
    ext = _require_ext()
    res = rc.reservoir

    if res isa ContinuousESNCell
        return _ar_continuous_esn(ext, res, rc, steps, ps, st; initialdata, initial_state)
    elseif res isa AbstractSciMLProblemReservoir
        return _ar_sciml(ext, res, rc, steps, ps, st; initialdata, initial_state)
    else
        throw(ArgumentError("predict_ar_seeded only for continuous reservoirs, got $(typeof(res))"))
    end
end

function _ar_continuous_esn(
        ext, cell, rc, steps, ps, st;
        initialdata, initial_state,
    )
    t0, t1 = cell.tspan
    ts = collect(range(t0, t1; length = steps + 1))
    window_starts = @view ts[1:(end - 1)]
    window_ends = @view ts[2:end]

    current_state = copy(initial_state)
    current_input = initialdata
    st_mods = st.states_modifiers
    st_ro = st.readout

    local outputs
    for (step_idx, (t_lo, t_hi)) in enumerate(zip(window_starts, window_ends))
        input_fn = ext._make_const_input_fn(current_input, t_lo, t_hi)
        solve_p = ext._build_solve_params(nothing, ps.reservoir, input_fn)
        sub_prob = ODEProblem(cell.equations, current_state, (t_lo, t_hi), solve_p)
        sol = solve(
            sub_prob, cell.args...;
            saveat = [t_hi],
            save_everystep = false,
            dense = false,
            cell.kwargs...,
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

function _ar_sciml(
        ext, res, rc, steps, ps, st;
        initialdata, initial_state,
    )
    t0, t1 = res.tspan
    ts = collect(range(t0, t1; length = steps + 1))
    window_starts = @view ts[1:(end - 1)]
    window_ends = @view ts[2:end]

    current_state = copy(initial_state)
    current_input = initialdata
    st_mods = st.states_modifiers
    st_ro = st.readout

    local outputs
    for (step_idx, (t_lo, t_hi)) in enumerate(zip(window_starts, window_ends))
        input_fn = ext._make_const_input_fn(current_input, t_lo, t_hi)
        solve_p = ext._build_solve_params(res.prob.p, ps.reservoir, input_fn)
        sub_prob = remake(
            res.prob;
            tspan = (t_lo, t_hi),
            p = solve_p,
            u0 = current_state,
        )
        sol = solve(
            sub_prob, res.args...;
            saveat = [t_hi],
            save_everystep = false,
            dense = false,
            res.kwargs...,
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

"""
    predict_ar_warmup(rc, steps, ps, st; initialdata, warmup_data)

Teacher-force on `warmup_data`, seed AR from raw terminal ODE state,
first AR input = `initialdata` (default: last column of warmup or
first test input — caller passes explicitly).
"""
function predict_ar_warmup(
        rc,
        steps::Integer,
        ps,
        st;
        initialdata::AbstractVector,
        warmup_data::AbstractMatrix,
    )
    u0 = raw_terminal_ode_state(rc, warmup_data, ps, st)
    return predict_ar_seeded(
        rc, steps, ps, st;
        initialdata = initialdata,
        initial_state = u0,
    )
end

"""Package AR predict — cold start as implemented on master."""
function predict_ar_cold(rc, steps, ps, st; initialdata)
    return predict(rc, steps, ps, st; initialdata = initialdata)
end

"""
Inspect whether `st` after continuous train/collect carries a hidden
state usable as `u0`. Returns a NamedTuple of diagnostics.
"""
function inspect_st_reservoir(st)
    r = st.reservoir
    return (
        type = string(typeof(r)),
        is_namedtuple = r isa NamedTuple,
        keys = r isa NamedTuple ? collect(keys(r)) : nothing,
        has_carry = r isa NamedTuple && haskey(r, :carry),
        has_cell = r isa NamedTuple && haskey(r, :cell),
        # ContinuousESNCell initialstates is typically (rng = …) only
        summary = sprint(show, r),
    )
end
