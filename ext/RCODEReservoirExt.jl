module RCODEReservoirExt

using DataInterpolations: ConstantInterpolation
using LuxCore: apply
# `OrdinaryDiffEq` is loaded as a weakdep trigger so concrete solver types
# (e.g. `Tsit5`) the user puts in `res.args[1]` are usable at solve time.
# We don't reference any of its names directly — `solve` and `remake` come
# from `SciMLBase`, and the solver algorithm dispatch is selected by the
# concrete object the user passed in.
using OrdinaryDiffEq: OrdinaryDiffEq
using SciMLBase: remake, solve, NullParameters

using ReservoirComputing: ReservoirComputing,
    AbstractReservoirComputer,
    AbstractSampler,
    AbstractSciMLProblemReservoir,
    TerminalStateSampling,
    collectstates
import ReservoirComputing: _collectstates, _predict

# ---------------------------------------------------------------------------
# Parameter assembly
#
# At solve time the extension must hand the ODE three things:
#   (1) the interpolated input signal `u(t)` exposed as `p.input`,
#   (2) the user's static parameters (if any) from `prob.p`,
#   (3) any Lux-managed reservoir parameters from `ps.reservoir`.
#
# `_to_namedtuple` normalises `prob.p` so that nothing / `NullParameters` / a
# user `NamedTuple` all collapse into a single `NamedTuple` we can merge into.
# Anything else is rejected with a clear error — wrapping unknown payloads
# silently would hide bugs in user-defined ODEs.
# ---------------------------------------------------------------------------

_to_namedtuple(p::NamedTuple) = p
_to_namedtuple(::NullParameters) = NamedTuple()
_to_namedtuple(::Nothing) = NamedTuple()
function _to_namedtuple(p)
    return throw(
        ArgumentError(
            "SciMLProblemReservoir requires `prob.p` to be a NamedTuple, " *
                "`nothing`, or `SciMLBase.NullParameters()`, got $(typeof(p)). " *
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

# ---------------------------------------------------------------------------
# Input signal construction
#
# `collectstates` sees a discrete `data::AbstractMatrix` and reconstructs the
# continuous-time input via linear interpolation between input columns. The
# grid mirrors the `saveat` grid so an input column and its corresponding
# state sample share the same time stamp.
#
# `_make_const_input_fn` is the closed-loop counterpart used inside the
# autoregressive `predict`: between two reservoir-output events the input is
# the previous output, held constant.
# ---------------------------------------------------------------------------

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
    n = length(ts)
    t < ts[1] && return view(interp.data, :, 1)
    t ≥ ts[end] && return view(interp.data, :, n)
    k = searchsortedlast(ts, t)
    return view(interp.data, :, clamp(k, 1, n))
end

function _make_input_fn(data::AbstractMatrix, ts::AbstractVector)
    return ZeroOrderHoldInterp(data, ts)
end

function _make_const_input_fn(u_vec::AbstractVector, t_lo, t_hi)
    # `cache_parameters=true` is fine for vector u (autoregressive predict
    # always holds the previous readout output constant over one sub-interval).
    return ConstantInterpolation([u_vec, u_vec], [t_lo, t_hi]; cache_parameters = true)
end

# ---------------------------------------------------------------------------
# Samplers
#
# A sampler maps a continuous trajectory into the discrete state matrix the
# readout sees. `TerminalStateSampling` reads the solution exactly at the
# user-visible time grid (the same one we pass through `saveat`), so the
# result is just the columnar view of `sol.u`.
# ---------------------------------------------------------------------------

function _sample(::TerminalStateSampling, sol)
    return reduce(hcat, sol.u)
end

# ---------------------------------------------------------------------------
# State-modifier composition
#
# The discrete fallback threads `states_modifiers` per reservoir step (see
# `_partial_apply` in `reservoircomputer.jl`). For the continuous path we
# evolve the trajectory first and then apply modifiers column-by-column to
# the sampled matrix. This keeps the per-sample semantics identical to the
# discrete code without contaminating the ODE right-hand side.
# ---------------------------------------------------------------------------

function _apply_modifiers_continuous(
        modifiers::Tuple, states_matrix::AbstractMatrix, ps_mods, st_mods
    )
    isempty(modifiers) && return states_matrix, st_mods
    T = size(states_matrix, 2)
    col1, new_st = ReservoirComputing._apply_seq(
        modifiers, view(states_matrix, :, 1), ps_mods, st_mods
    )
    # `similar(col1, ...)` — not `similar(states_matrix, ...)` — so the
    # output matrix takes the modifier output's eltype. If a modifier
    # promotes/demotes (e.g. Float32 → Float64), we want that to surface,
    # not be silently truncated back to the reservoir state's eltype.
    out = similar(col1, length(col1), T)
    out[:, 1] .= col1
    for t in 2:T
        col, new_st = ReservoirComputing._apply_seq(
            modifiers, view(states_matrix, :, t), ps_mods, new_st
        )
        out[:, t] .= col
    end
    return out, new_st
end

# ---------------------------------------------------------------------------
# Continuous `_collectstates`
#
# Pipeline:
#   1. Split `res.tspan` into `T_steps` equal-width windows.
#   2. Place input column `k` at the *start* of window `k` (time
#      `t0 + (k-1)Δt`) and request a sample at the *end* of window `k`
#      (time `t0 + kΔt`). This alignment matches the discrete reservoir
#      semantics — `states[:, k]` is the state after processing input `k`
#      — and is what makes the Euler-equivalence test land without an
#      off-by-one shift.
#   3. `remake` the user's problem with the locked `tspan` and the merged
#      parameter pack (interpolated input injected as `p.input`).
#   4. `solve(...; saveat = sample_ts, save_everystep=false, dense=false)`.
#      `res.kwargs` come last so user kwargs win on collision — the
#      constructor already rejects the three protected keys, so they
#      cannot collide in practice.
#   5. Push the trajectory through the sampler → raw state matrix.
#   6. Apply state modifiers → final state matrix matching the discrete
#      `(state_dims, T)` shape expected by the readout.
# ---------------------------------------------------------------------------

function _collectstates(
        res::AbstractSciMLProblemReservoir,
        rc::AbstractReservoirComputer,
        data::AbstractMatrix,
        ps::NamedTuple,
        st::NamedTuple
    )
    T_steps = size(data, 2)
    T_steps ≥ 2 || throw(
        ArgumentError(
            "SciMLProblemReservoir collectstates needs at least 2 input " *
                "columns to define a time grid; got $T_steps."
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

    Δt = (t1 - t0) / T_steps
    input_ts = collect(range(t0, t1 - Δt; length = T_steps))
    sample_ts = collect(range(t0 + Δt, t1; length = T_steps))

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

# ---------------------------------------------------------------------------
# Teacher-forced `predict`
#
# Solve once over the whole tspan, then apply the readout column-by-column.
# Cheaper than the autoregressive path because the ODE never has to be
# restarted between samples.
# ---------------------------------------------------------------------------

function _predict(
        ::AbstractSciMLProblemReservoir,
        rc::AbstractReservoirComputer,
        data::AbstractMatrix,
        ps::NamedTuple,
        st::NamedTuple
    )
    states, new_st = collectstates(rc, data, ps, st)
    T = size(states, 2)
    st_ro = new_st.readout
    y1, st_ro = apply(rc.readout, view(states, :, 1), ps.readout, st_ro)
    Y = similar(y1, size(y1, 1), T)
    Y[:, 1] .= y1
    for t in 2:T
        yt, st_ro = apply(rc.readout, view(states, :, t), ps.readout, st_ro)
        Y[:, t] .= yt
    end
    return Y, merge(new_st, (readout = st_ro,))
end

# ---------------------------------------------------------------------------
# Autoregressive `predict`
#
# Split `tspan` into `steps` equal sub-intervals. For each sub-interval the
# input is the previous readout output, held constant via a
# `ConstantInterpolation`. After each sub-solve we:
#   - sample the terminal state,
#   - apply state modifiers (per-sample, consistent with the discrete loop),
#   - apply the readout,
#   - feed the output back as the next input.
#
# The initial reservoir state is `res.prob.u0`; users who want to continue
# from a previously computed trajectory should `remake(prob; u0 = …)` before
# constructing the reservoir.
# ---------------------------------------------------------------------------

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

    # Preserve `u0`'s original type — `collect` would degrade `SVector` /
    # `ComponentArray` / scalar states into a plain `Vector` and either
    # error (no `collect(::Number)` method) or silently flatten the
    # user's chosen representation. We only ever read `x_current`, never
    # mutate it in place, so a direct reference is safe.
    x_current = res.prob.u0
    current_input = initialdata

    st_mods = st.states_modifiers
    st_ro = st.readout

    # `output` is allocated *after* the first readout call so its element
    # type and row count come from `apply(rc.readout, …)` rather than
    # `initialdata`. Otherwise a readout returning a different eltype
    # (e.g. Float64 vs the Float32 input) would force a silent
    # conversion at the column assignment.
    local output
    for k in 1:steps
        input_fn = _make_const_input_fn(current_input, ts[k], ts[k + 1])
        solve_p = _build_solve_params(res.prob.p, ps.reservoir, input_fn)
        sub_prob = remake(
            res.prob;
            tspan = (ts[k], ts[k + 1]),
            p = solve_p,
            u0 = x_current
        )
        sol = solve(
            sub_prob, res.args...;
            saveat = [ts[k + 1]],
            save_everystep = false,
            dense = false,
            res.kwargs...
        )
        x_current = sol.u[end]

        if !isempty(rc.states_modifiers)
            x_after_mods, st_mods = ReservoirComputing._apply_seq(
                rc.states_modifiers, x_current, ps.states_modifiers, st_mods
            )
        else
            x_after_mods = x_current
        end

        y, st_ro = apply(rc.readout, x_after_mods, ps.readout, st_ro)
        if k == 1
            output = similar(y, length(y), steps)
        end
        output[:, k] .= y
        current_input = y
    end

    newst = (
        reservoir = st.reservoir,
        states_modifiers = st_mods,
        readout = st_ro,
    )
    return output, newst
end

end # module
