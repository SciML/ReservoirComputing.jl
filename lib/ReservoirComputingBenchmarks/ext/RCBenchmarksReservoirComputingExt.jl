module RCBenchmarksReservoirComputingExt

using Random: AbstractRNG, default_rng, randn
using ReservoirComputing: AbstractReservoirComputer, collectstates
using ReservoirComputingBenchmarks: ReservoirComputingBenchmarks,
    memory_capacity, nonlinear_memory,
    nonlinear_transformation, sin_approximation,
    narma, ipc,
    kernel_rank, generalization_rank

# ── Helpers ─────────────────────────────────────────────────────────────────

@inline function _uniform_input(rng::AbstractRNG, T::Integer, lo::Real, hi::Real)
    @assert T >= 2 "Signal length T must be >= 2, got $T"
    @assert hi > lo "Require hi > lo, got lo=$lo, hi=$hi"
    return rand(rng, T) .* (hi - lo) .+ lo
end

@inline function _resolve_input(
        rng::AbstractRNG, T::Integer, input::Union{Nothing, AbstractVector}
    )
    u = input === nothing ? _uniform_input(rng, T, -1.0, 1.0) : input
    @assert length(u) >= 2 "input must have at least 2 samples, got $(length(u))"
    return u
end

# Best-effort check that the model accepts scalar (in_dims == 1) inputs.
# Walks `rc.reservoir.cell.in_dims` when available; no-op otherwise so that
# unconventional model layouts still fall through to `collectstates`'s own
# shape error.
@inline function _check_scalar_input(rc::AbstractReservoirComputer)
    res = getfield(rc, :reservoir)
    cell = hasproperty(res, :cell) ? getproperty(res, :cell) : nothing
    cell === nothing && return nothing
    if hasproperty(cell, :in_dims)
        in_dims = getproperty(cell, :in_dims)
        in_dims == 1 || throw(
            ArgumentError(
                "ReservoirComputingBenchmarks model dispatch requires a reservoir with " *
                    "scalar input (in_dims == 1), got in_dims == $in_dims. Provide a model " *
                    "constructed with `in_dims = 1`, or call the array-based method directly " *
                    "with your own (n_features, T) state matrix.",
            ),
        )
    end
    return nothing
end

@inline function _collect_scalar_states(
        rc::AbstractReservoirComputer, u::AbstractVector, ps, st
    )
    _check_scalar_input(rc)
    data = reshape(u, 1, length(u))
    states, _ = collectstates(rc, data, ps, st)
    return states
end

# Drive the reservoir with `u`, return the **final** state vector. Each call
# uses the user-provided initial `st` (with carry=nothing) so successive runs
# are independent. Skips the per-call scalar-input check (assumed already
# verified once by the caller).
@inline function _final_state(
        rc::AbstractReservoirComputer, u::AbstractVector, ps, st
    )
    data = reshape(u, 1, length(u))
    states, _ = collectstates(rc, data, ps, st)
    return states[:, end]
end

# ── Memory Capacity ─────────────────────────────────────────────────────────

@doc raw"""
    memory_capacity(rc::AbstractReservoirComputer, ps, st;
                    T=3000, rng=Random.default_rng(),
                    input=nothing, kwargs...)

Linear memory capacity of a reservoir computing model. Generates a uniform
``[-1, 1]`` input, drives the model via [`collectstates`](@ref), and dispatches
to the array-based [`memory_capacity`](@ref).

Remaining keyword arguments (`max_delay`, `train_ratio`, `reg`) are forwarded.
The reservoir model must accept scalar inputs (`in_dims == 1`).
"""
function ReservoirComputingBenchmarks.memory_capacity(
        rc::AbstractReservoirComputer, ps, st;
        T::Integer = 3000,
        rng::AbstractRNG = default_rng(),
        input::Union{Nothing, AbstractVector} = nothing,
        kwargs...,
    )
    u = _resolve_input(rng, T, input)
    states = _collect_scalar_states(rc, u, ps, st)
    return memory_capacity(u, states; kwargs...)
end

# ── Nonlinear Memory ────────────────────────────────────────────────────────

@doc raw"""
    nonlinear_memory(rc::AbstractReservoirComputer, ps, st;
                     T=3000, rng=Random.default_rng(),
                     input=nothing, kwargs...)

Nonlinear memory capacity of a reservoir computing model. Generates a uniform
``[-1, 1]`` input, drives the model via [`collectstates`](@ref), and dispatches
to the array-based [`nonlinear_memory`](@ref).

Remaining keyword arguments (`f`, `max_delay`, `train_ratio`, `reg`) are
forwarded. The reservoir model must accept scalar inputs (`in_dims == 1`).
"""
function ReservoirComputingBenchmarks.nonlinear_memory(
        rc::AbstractReservoirComputer, ps, st;
        T::Integer = 3000,
        rng::AbstractRNG = default_rng(),
        input::Union{Nothing, AbstractVector} = nothing,
        kwargs...,
    )
    u = _resolve_input(rng, T, input)
    states = _collect_scalar_states(rc, u, ps, st)
    return nonlinear_memory(u, states; kwargs...)
end

# ── Nonlinear Transformation ────────────────────────────────────────────────

@doc raw"""
    nonlinear_transformation(rc::AbstractReservoirComputer, ps, st;
                             T=3000, rng=Random.default_rng(),
                             input=nothing, kwargs...)

Memoryless nonlinear transformation benchmark on a reservoir computing model.
Generates a uniform ``[-1, 1]`` input, drives the model via
[`collectstates`](@ref), and dispatches to the array-based
[`nonlinear_transformation`](@ref).

Remaining keyword arguments (`f`, `train_ratio`, `reg`, `metric`, `washout`)
are forwarded. The reservoir model must accept scalar inputs (`in_dims == 1`).
"""
function ReservoirComputingBenchmarks.nonlinear_transformation(
        rc::AbstractReservoirComputer, ps, st;
        T::Integer = 3000,
        rng::AbstractRNG = default_rng(),
        input::Union{Nothing, AbstractVector} = nothing,
        kwargs...,
    )
    u = _resolve_input(rng, T, input)
    states = _collect_scalar_states(rc, u, ps, st)
    return nonlinear_transformation(u, states; kwargs...)
end

# ── Sin Approximation ───────────────────────────────────────────────────────

@doc raw"""
    sin_approximation(rc::AbstractReservoirComputer, ps, st;
                      T=3000, rng=Random.default_rng(),
                      input=nothing, kwargs...)

Sin-approximation benchmark on a reservoir computing model. Generates a
uniform ``[-1, 1]`` input, drives the model via [`collectstates`](@ref), and
dispatches to the array-based [`sin_approximation`](@ref).

Remaining keyword arguments (`freq`, `train_ratio`, `reg`, `metric`,
`washout`) are forwarded. The reservoir model must accept scalar inputs
(`in_dims == 1`).
"""
function ReservoirComputingBenchmarks.sin_approximation(
        rc::AbstractReservoirComputer, ps, st;
        T::Integer = 3000,
        rng::AbstractRNG = default_rng(),
        input::Union{Nothing, AbstractVector} = nothing,
        kwargs...,
    )
    u = _resolve_input(rng, T, input)
    states = _collect_scalar_states(rc, u, ps, st)
    return sin_approximation(u, states; kwargs...)
end

# ── NARMA ───────────────────────────────────────────────────────────────────

@doc raw"""
    narma(rc::AbstractReservoirComputer, ps, st;
          T=3000, rng=Random.default_rng(),
          input=nothing, kwargs...)

Evaluate a reservoir computing model on the NARMA-N task. Generates a uniform
``[-1, 1]`` input, drives the model via [`collectstates`](@ref), and dispatches
to the array-based [`narma`](@ref).

Remaining keyword arguments (`order`, `metric`, `train_ratio`, `reg`,
`washout`, NARMA coefficients, ...) are forwarded. The reservoir model must
accept scalar inputs (`in_dims == 1`).
"""
function ReservoirComputingBenchmarks.narma(
        rc::AbstractReservoirComputer, ps, st;
        T::Integer = 3000,
        rng::AbstractRNG = default_rng(),
        input::Union{Nothing, AbstractVector} = nothing,
        kwargs...,
    )
    u = _resolve_input(rng, T, input)
    states = _collect_scalar_states(rc, u, ps, st)
    return narma(u, states; kwargs...)
end

# ── IPC ─────────────────────────────────────────────────────────────────────

@doc raw"""
    ipc(rc::AbstractReservoirComputer, ps, st;
        T=3000, rng=Random.default_rng(),
        input=nothing, kwargs...)

Information Processing Capacity of a reservoir computing model. Generates a
uniform ``[-1, 1]`` input, drives the model via [`collectstates`](@ref), and
dispatches to the array-based [`ipc`](@ref).

Remaining keyword arguments (`max_delay`, `max_degree`, `max_total_degree`,
`cross_terms`, `train_ratio`, `reg`) are forwarded. The reservoir model must
accept scalar inputs (`in_dims == 1`).
"""
function ReservoirComputingBenchmarks.ipc(
        rc::AbstractReservoirComputer, ps, st;
        T::Integer = 3000,
        rng::AbstractRNG = default_rng(),
        input::Union{Nothing, AbstractVector} = nothing,
        kwargs...,
    )
    u = _resolve_input(rng, T, input)
    states = _collect_scalar_states(rc, u, ps, st)
    return ipc(u, states; kwargs...)
end

# ── Kernel Rank ─────────────────────────────────────────────────────────────

# Drive `n_streams` independent runs and stack the final states column-wise.
function _collect_final_states(
        rc::AbstractReservoirComputer, ps, st,
        streams::Function, n_streams::Integer
    )
    @assert n_streams >= 1 "n_streams must be >= 1, got $n_streams"
    _check_scalar_input(rc)
    final_states = nothing
    n_features = -1
    @inbounds for i in 1:n_streams
        u = streams(i)
        x = _final_state(rc, u, ps, st)
        if final_states === nothing
            n_features = length(x)
            final_states = Matrix{eltype(x)}(undef, n_features, n_streams)
        end
        @assert length(x) == n_features "Final state size changed across runs ($(length(x)) vs $n_features)"
        final_states[:, i] .= x
    end
    return final_states
end

@doc raw"""
    kernel_rank(rc::AbstractReservoirComputer, ps, st;
                n_streams=500, stream_length=100,
                rng=Random.default_rng(), threshold=0.01)

Drive the reservoir with `n_streams` independent uniform ``[-1, 1]`` input
streams of length `stream_length`, collect the final state of each run, and
return the numerical rank of the resulting `(n_features, n_streams)` matrix.

The user-provided initial `st` is reused for each run, ensuring that every
stream starts from the same fresh carry (`nothing`), so the runs are
independent.

Remaining keyword arguments (`threshold`) are forwarded to the array-based
[`kernel_rank`](@ref). The reservoir model must accept scalar inputs
(`in_dims == 1`).
"""
function ReservoirComputingBenchmarks.kernel_rank(
        rc::AbstractReservoirComputer, ps, st;
        n_streams::Integer = 500,
        stream_length::Integer = 100,
        rng::AbstractRNG = default_rng(),
        threshold::Real = 0.01,
    )
    @assert stream_length >= 2 "stream_length must be >= 2, got $stream_length"
    streams = _ -> _uniform_input(rng, stream_length, -1.0, 1.0)
    M = _collect_final_states(rc, ps, st, streams, n_streams)
    return kernel_rank(M; threshold = threshold)
end

# ── Generalization Rank ─────────────────────────────────────────────────────

@doc raw"""
    generalization_rank(rc::AbstractReservoirComputer, ps, st;
                        n_streams=500, stream_length=100,
                        perturbation=0.01, base_input=nothing,
                        rng=Random.default_rng(), threshold=0.01)

Drive the reservoir with `n_streams` slightly perturbed copies of a common
base input stream of length `stream_length`, collect the final state of each
run, and return the numerical rank of the resulting matrix.

A *lower* generalization rank means the reservoir collapses similar inputs to
similar states (good generalization). The base stream is sampled uniformly in
``[-1, 1]`` unless supplied via `base_input`; perturbations are i.i.d. Gaussian
noise of standard deviation `perturbation`.

Remaining keyword arguments (`threshold`) are forwarded to the array-based
[`generalization_rank`](@ref). The reservoir model must accept scalar inputs
(`in_dims == 1`).
"""
function ReservoirComputingBenchmarks.generalization_rank(
        rc::AbstractReservoirComputer, ps, st;
        n_streams::Integer = 500,
        stream_length::Integer = 100,
        perturbation::Real = 0.01,
        base_input::Union{Nothing, AbstractVector} = nothing,
        rng::AbstractRNG = default_rng(),
        threshold::Real = 0.01,
    )
    @assert stream_length >= 2 "stream_length must be >= 2, got $stream_length"
    @assert perturbation >= 0 "perturbation must be >= 0, got $perturbation"
    base = base_input === nothing ?
        _uniform_input(rng, stream_length, -1.0, 1.0) : base_input
    @assert length(base) == stream_length (
        "base_input length ($(length(base))) must equal stream_length ($stream_length)"
    )
    streams = _ -> base .+ perturbation .* randn(rng, stream_length)
    M = _collect_final_states(rc, ps, st, streams, n_streams)
    return generalization_rank(M; threshold = threshold)
end

end # module
