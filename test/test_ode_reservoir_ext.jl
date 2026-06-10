using Test
using Random
using LinearAlgebra
using ReservoirComputing
using OrdinaryDiffEq
using SciMLBase
using DataInterpolations

# ---------------------------------------------------------------------------
# Helpers — small ESN-shaped ODE used across several tests.
# ---------------------------------------------------------------------------

function esn_rhs!(dx, x, p, t)
    u = p.input(t)
    return dx .= .-x .+ tanh.(p.Wr * x .+ p.Win * u .+ p.b)
end

function build_esn_problem(rng, in_dim, res_dim, tspan)
    Wr = 0.2 .* randn(rng, res_dim, res_dim)
    Win = 0.5 .* randn(rng, res_dim, in_dim)
    bias = 0.1 .* randn(rng, res_dim)
    initial_state = zeros(res_dim)
    params = (Wr = Wr, Win = Win, b = bias)
    return ODEProblem(esn_rhs!, initial_state, tspan, params),
        Wr, Win, bias, initial_state
end

# ---------------------------------------------------------------------------
# 1. Linear ODE — analytic match
#
# Trivial ODE dx/dt = -x + u, u constant, x(0) = 0 has the closed form
# x(t) = u (1 - exp(-t)). Verify that the continuous helper recovers the
# analytic curve to within a tight solver tolerance. With the corrected
# saveat alignment, the first sample is at `t = Δt`, not `t = 0`.
# ---------------------------------------------------------------------------

@testset "Linear ODE analytic match" begin
    function lin_rhs!(dx, x, p, t)
        u_val = p.input(t)
        return dx .= .-x .+ u_val
    end

    T_steps = 10
    tspan = (0.0, 1.0)
    Δt = (tspan[2] - tspan[1]) / T_steps
    sample_ts = collect(range(tspan[1] + Δt, tspan[2]; length = T_steps))
    u_const = 1.0
    data = fill(u_const, 1, T_steps)
    initial_state = [0.0]

    prob = ODEProblem(lin_rhs!, initial_state, tspan, (;))
    res = SciMLProblemReservoir(
        prob, TerminalStateSampling(), tspan, Tsit5();
        reltol = 1.0e-10, abstol = 1.0e-12
    )
    rc = ReservoirComputer(res, LinearReadout(1 => 1))
    ps, st = setup(MersenneTwister(0), rc)

    states, _ = collectstates(rc, data, ps, st)
    analytic = u_const .* (1 .- exp.(.-sample_ts))

    @test size(states) == (1, T_steps)
    @test states[1, :] ≈ analytic atol = 1.0e-6
end

# ---------------------------------------------------------------------------
# 2. Euler equivalence
#
# Solving dx/dt = -x + tanh(Wr x + Win u(t) + b) with explicit Euler at
# step size dt = 1 collapses algebraically to the discrete reservoir update
# x_{k+1} = tanh(Wr x_k + Win u_{k+1} + b). With the corrected alignment
# (inputs at window starts, samples at window ends), there is no off-by-one
# between continuous and discrete trajectories.
# ---------------------------------------------------------------------------

@testset "Euler equivalence with discrete reservoir update" begin
    rng = MersenneTwister(7)
    in_dim, res_dim, T_steps = 2, 6, 12

    prob, Wr, Win, bias, initial_state = build_esn_problem(
        rng, in_dim, res_dim,
        (0.0, Float64(T_steps))
    )
    data = randn(rng, in_dim, T_steps)

    res = SciMLProblemReservoir(
        prob, TerminalStateSampling(),
        (0.0, Float64(T_steps)), Euler();
        dt = 1.0
    )
    rc = ReservoirComputer(res, LinearReadout(res_dim => 1))
    ps, st = setup(MersenneTwister(0), rc)

    cont_states, _ = collectstates(rc, data, ps, st)

    disc_states = zeros(res_dim, T_steps)
    state = copy(initial_state)
    for (step_idx, input_col) in enumerate(eachcol(data))
        state = tanh.(Wr * state + Win * input_col + bias)
        disc_states[:, step_idx] = state
    end

    @test size(cont_states) == (res_dim, T_steps)
    @test cont_states ≈ disc_states atol = 1.0e-10
end

# ---------------------------------------------------------------------------
# 3. Sampler shape contract
#
# `TerminalStateSampling` must produce a `(state_dim, T_input)` matrix
# regardless of solver. Guards against accidental transposition or extra
# rows/columns sneaking in via `reduce(hcat, sol.u)`.
# ---------------------------------------------------------------------------

@testset "TerminalStateSampling output shape" begin
    rng = MersenneTwister(11)
    in_dim, res_dim, T_steps = 3, 8, 20
    tspan = (0.0, 2.0)
    prob, _, _, _, _ = build_esn_problem(rng, in_dim, res_dim, tspan)
    data = randn(rng, in_dim, T_steps)
    res = SciMLProblemReservoir(
        prob, TerminalStateSampling(), tspan, Tsit5();
        reltol = 1.0e-8, abstol = 1.0e-10
    )
    rc = ReservoirComputer(res, LinearReadout(res_dim => 1))
    ps, st = setup(MersenneTwister(0), rc)

    states, _ = collectstates(rc, data, ps, st)
    @test size(states) == (res_dim, T_steps)
    @test all(isfinite, states)
end

# ---------------------------------------------------------------------------
# 4. Teacher-forced predict
#
# `predict(rc, data, ps, st)` runs one bulk ODE solve and applies the
# readout column-by-column. Output dims must match the readout's
# `out_dims`, and the result must be deterministic.
# ---------------------------------------------------------------------------

@testset "Teacher-forced predict" begin
    rng = MersenneTwister(23)
    in_dim, res_dim, out_dim, T_steps = 2, 6, 4, 15
    tspan = (0.0, 3.0)
    prob, _, _, _, _ = build_esn_problem(rng, in_dim, res_dim, tspan)
    data = randn(rng, in_dim, T_steps)
    res = SciMLProblemReservoir(
        prob, TerminalStateSampling(), tspan, Tsit5();
        reltol = 1.0e-8, abstol = 1.0e-10
    )
    rc = ReservoirComputer(res, LinearReadout(res_dim => out_dim))
    ps, st = setup(MersenneTwister(0), rc)

    preds1, _ = predict(rc, data, ps, st)
    preds2, _ = predict(rc, data, ps, st)
    @test size(preds1) == (out_dim, T_steps)
    @test all(isfinite, preds1)
    @test preds1 ≈ preds2
end

# ---------------------------------------------------------------------------
# 5. Autoregressive predict
#
# `predict(rc, steps, ps, st; initialdata)` runs `steps` sub-solves,
# feeding the previous readout output back as the constant input on the
# next sub-interval. Shape and determinism both checked here.
# ---------------------------------------------------------------------------

@testset "Autoregressive predict" begin
    rng = MersenneTwister(31)
    res_dim, dim, steps = 6, 3, 5
    tspan = (0.0, 1.0)
    prob, _, _, _, _ = build_esn_problem(rng, dim, res_dim, tspan)
    res = SciMLProblemReservoir(
        prob, TerminalStateSampling(), tspan, Tsit5();
        reltol = 1.0e-8, abstol = 1.0e-10
    )
    rc = ReservoirComputer(res, LinearReadout(res_dim => dim))
    ps, st = setup(MersenneTwister(0), rc)

    initialdata = randn(dim)
    preds1, _ = predict(rc, steps, ps, st; initialdata = initialdata)
    preds2, _ = predict(rc, steps, ps, st; initialdata = initialdata)
    @test size(preds1) == (dim, steps)
    @test all(isfinite, preds1)
    @test preds1 ≈ preds2
end

# ---------------------------------------------------------------------------
# 6. State modifiers compose with the continuous path
#
# `states_modifiers` must compose with the continuous reservoir the same
# way they do with the discrete one — apply per saved sample, threading
# the modifier state across columns. NLAT2 doubles even-indexed columns
# of its input, so the modified state must differ in those columns from
# the raw one.
# ---------------------------------------------------------------------------

@testset "State modifiers on continuous path" begin
    rng = MersenneTwister(41)
    in_dim, res_dim, T_steps = 2, 6, 8
    tspan = (0.0, 1.0)
    prob, _, _, _, _ = build_esn_problem(rng, in_dim, res_dim, tspan)
    data = randn(rng, in_dim, T_steps)
    res = SciMLProblemReservoir(
        prob, TerminalStateSampling(), tspan, Tsit5();
        reltol = 1.0e-8, abstol = 1.0e-10
    )

    rc_plain = ReservoirComputer(res, LinearReadout(res_dim => 1))
    rc_mod = ReservoirComputer(res, (NLAT2(),), LinearReadout(res_dim => 1))

    ps_plain, st_plain = setup(MersenneTwister(0), rc_plain)
    ps_mod, st_mod = setup(MersenneTwister(0), rc_mod)

    states_plain, _ = collectstates(rc_plain, data, ps_plain, st_plain)
    states_mod, _ = collectstates(rc_mod, data, ps_mod, st_mod)

    @test size(states_mod) == size(states_plain)
    @test all(isfinite, states_mod)
    # NLAT2 mutates even-indexed rows in-place — at least some entries must
    # differ from the unmodified state matrix.
    @test states_mod != states_plain
end

# ---------------------------------------------------------------------------
# 7. Boundary inputs
#
# Smallest valid sizes: `T_steps = 2` for collectstates, `steps = 1` for
# autoregressive predict. Larger guards (≥2 and ≥1) reject anything
# smaller with an ArgumentError. Make sure both paths land cleanly at
# their lower bounds and that the guards actually fire one step lower.
# ---------------------------------------------------------------------------

@testset "Boundary sizes" begin
    rng = MersenneTwister(53)
    in_dim, res_dim = 2, 4
    tspan = (0.0, 1.0)
    prob, _, _, _, _ = build_esn_problem(rng, in_dim, res_dim, tspan)
    res = SciMLProblemReservoir(
        prob, TerminalStateSampling(), tspan, Tsit5();
        reltol = 1.0e-8, abstol = 1.0e-10
    )
    rc = ReservoirComputer(res, LinearReadout(res_dim => in_dim))
    ps, st = setup(MersenneTwister(0), rc)

    # collectstates: T_steps = 2 works
    data2 = randn(rng, in_dim, 2)
    states2, _ = collectstates(rc, data2, ps, st)
    @test size(states2) == (res_dim, 2)

    # collectstates: T_steps < 2 errors
    data1 = randn(rng, in_dim, 1)
    @test_throws ArgumentError collectstates(rc, data1, ps, st)
    data0 = Matrix{Float64}(undef, in_dim, 0)
    @test_throws ArgumentError collectstates(rc, data0, ps, st)

    # autoregressive predict: steps = 1 works
    preds1, _ = predict(rc, 1, ps, st; initialdata = randn(in_dim))
    @test size(preds1) == (in_dim, 1)

    # autoregressive predict: steps < 1 errors
    @test_throws ArgumentError predict(rc, 0, ps, st; initialdata = randn(in_dim))
end

# ---------------------------------------------------------------------------
# 8. Construction-time validation of protected kwargs
#
# `SciMLProblemReservoir` rejects `saveat`, `save_everystep`, and `dense`
# in `kwargs` at construction.
# ---------------------------------------------------------------------------

@testset "Protected solve kwargs rejected at construction" begin
    placeholder = (placeholder = true,)
    sampler = TerminalStateSampling()
    tspan = (0.0, 1.0)
    for badkw in (:saveat, :save_everystep, :dense)
        @test_throws ArgumentError SciMLProblemReservoir(
            placeholder, sampler, tspan, Tsit5(); (badkw => true,)...
        )
    end
end

# ---------------------------------------------------------------------------
# 9. `tspan` must be a strictly positive interval
#
# Degenerate `tspan = (c, c)` (or backward) would divide by zero when the
# extension synthesises the input grid step. Validate at solve time.
# ---------------------------------------------------------------------------

@testset "tspan strictly positive" begin
    rng = MersenneTwister(67)
    in_dim, res_dim = 2, 4
    prob, _, _, _, _ = build_esn_problem(rng, in_dim, res_dim, (0.0, 1.0))
    data = randn(rng, in_dim, 4)

    # Equal endpoints
    res_eq = SciMLProblemReservoir(
        prob, TerminalStateSampling(), (1.0, 1.0), Tsit5()
    )
    rc_eq = ReservoirComputer(res_eq, LinearReadout(res_dim => 1))
    ps, st = setup(MersenneTwister(0), rc_eq)
    @test_throws ArgumentError collectstates(rc_eq, data, ps, st)
    @test_throws ArgumentError predict(rc_eq, 3, ps, st; initialdata = randn(in_dim))

    # Backward interval
    res_back = SciMLProblemReservoir(
        prob, TerminalStateSampling(), (1.0, 0.0), Tsit5()
    )
    rc_back = ReservoirComputer(res_back, LinearReadout(res_dim => 1))
    @test_throws ArgumentError collectstates(rc_back, data, ps, st)
end

# ---------------------------------------------------------------------------
# 10. Reserved `:input` key collision
#
# `:input` is the name the extension injects into the solve params so the
# user's RHS can read `p.input(t)`. A `prob.p` already carrying `:input`
# would be silently shadowed — error loudly instead.
# ---------------------------------------------------------------------------

@testset "Reserved `:input` key collision errors" begin
    rng = MersenneTwister(79)
    res_dim = 4

    function rhs_bad!(dx, x, p, t)
        return dx .= .-x
    end

    prob_bad = ODEProblem(
        rhs_bad!, zeros(res_dim), (0.0, 1.0),
        (input = "already taken",)
    )
    res = SciMLProblemReservoir(
        prob_bad, TerminalStateSampling(), (0.0, 1.0), Tsit5()
    )
    rc = ReservoirComputer(res, LinearReadout(res_dim => 1))
    ps, st = setup(MersenneTwister(0), rc)
    data = randn(rng, 1, 4)
    @test_throws ArgumentError collectstates(rc, data, ps, st)
end

# ---------------------------------------------------------------------------
# 11. `prob.p` accepted forms: NamedTuple / nothing / NullParameters
#
# `_to_namedtuple` advertises three valid inputs and rejects anything
# else. Exercise all three success paths end-to-end so a future refactor
# can't quietly break two of them.
# ---------------------------------------------------------------------------

@testset "prob.p accepts NamedTuple / nothing / NullParameters" begin
    rng = MersenneTwister(83)
    in_dim, res_dim, T_steps = 1, 4, 6
    tspan = (0.0, 1.0)
    data = randn(rng, in_dim, T_steps)

    # `prob.p` is read inside the RHS, so to truly exercise all three we use
    # an RHS that does not touch `p` apart from `p.input(t)`. A `let` block
    # captures `Win` as a local binding inside the closure — avoids the
    # type-instability hazard of a `global` and keeps the symbol out of the
    # surrounding module scope.
    rhs_noparams! = let Win = 0.5 .* randn(rng, res_dim, in_dim)
        (dx, x, p, t) -> (dx .= .-x .+ Win * p.input(t))
    end

    for p_value in ((;), nothing, SciMLBase.NullParameters())
        prob = ODEProblem(rhs_noparams!, zeros(res_dim), tspan, p_value)
        res = SciMLProblemReservoir(
            prob, TerminalStateSampling(), tspan, Tsit5();
            reltol = 1.0e-8, abstol = 1.0e-10
        )
        rc = ReservoirComputer(res, LinearReadout(res_dim => 1))
        ps, st = setup(MersenneTwister(0), rc)
        states, _ = collectstates(rc, data, ps, st)
        @test size(states) == (res_dim, T_steps)
        @test all(isfinite, states)
    end
end

# ---------------------------------------------------------------------------
# 12. Non-NamedTuple `prob.p` rejected with a clear error
# ---------------------------------------------------------------------------

@testset "Non-NamedTuple prob.p errors clearly" begin
    function rhs!(dx, x, p, t)
        return dx .= .-x
    end
    prob = ODEProblem(rhs!, [0.0], (0.0, 1.0), [1.0, 2.0])  # Vector params
    res = SciMLProblemReservoir(
        prob, TerminalStateSampling(), (0.0, 1.0), Tsit5();
        reltol = 1.0e-6
    )
    rc = ReservoirComputer(res, LinearReadout(1 => 1))
    ps, st = setup(MersenneTwister(0), rc)
    data = ones(1, 4)
    @test_throws ArgumentError collectstates(rc, data, ps, st)
end
