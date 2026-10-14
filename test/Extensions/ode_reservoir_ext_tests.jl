# ODE reservoir extension contracts.
begin
    using Test
    using Random
    using LinearAlgebra
    using ReservoirComputing
    using LuxCore: setup
    using OrdinaryDiffEq
    # Euler lives in OrdinaryDiffEqLowOrderRK under OrdinaryDiffEq v7+
    if !isdefined(@__MODULE__, :Euler)
        using OrdinaryDiffEqLowOrderRK: Euler
    end
    using SciMLBase
    using DataInterpolations

    # ---------------------------------------------------------------------------
    # Helpers — small ESN-shaped ODE used across several tests.
    # ---------------------------------------------------------------------------

    function esn_rhs!(dx, x, p, t)
        input_t = p.input(t)
        return dx .= .-x .+ tanh.(p.Wr * x .+ p.Win * input_t .+ p.b)
    end

    function build_esn_problem(rng, in_dim, res_dim, tspan)
        Wr = 0.2 .* randn(rng, res_dim, res_dim)
        Win = 0.5 .* randn(rng, res_dim, in_dim)
        bias = 0.1 .* randn(rng, res_dim)
        initial_state = zeros(res_dim)
        params = (Wr = Wr, Win = Win, b = bias)
        return ODEProblem(esn_rhs!, initial_state, tspan, params),
            Wr,
            Win,
            bias,
            initial_state
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
            input_t = p.input(t)
            return dx .= .-x .+ input_t
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
            prob,
            TerminalStateSampling(),
            tspan,
            Tsit5();
            reltol = 1.0e-10,
            abstol = 1.0e-12,
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

        prob, Wr, Win, bias, initial_state =
            build_esn_problem(rng, in_dim, res_dim, (0.0, Float64(T_steps)))
        data = randn(rng, in_dim, T_steps)

        res = SciMLProblemReservoir(
            prob,
            TerminalStateSampling(),
            (0.0, Float64(T_steps)),
            Euler();
            dt = 1.0,
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
            prob,
            TerminalStateSampling(),
            tspan,
            Tsit5();
            reltol = 1.0e-8,
            abstol = 1.0e-10,
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
            prob,
            TerminalStateSampling(),
            tspan,
            Tsit5();
            reltol = 1.0e-8,
            abstol = 1.0e-10,
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
            prob,
            TerminalStateSampling(),
            tspan,
            Tsit5();
            reltol = 1.0e-8,
            abstol = 1.0e-10,
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

    @testset "Autoregressive predict uses carry" begin
        rng = MersenneTwister(37)
        res_dim, dim, T_steps, steps = 8, 2, 16, 4
        tspan = (0.0, 2.0)
        prob, _, _, _, _ = build_esn_problem(rng, dim, res_dim, tspan)
        prob = remake(prob; u0 = 0.3 .* randn(rng, res_dim))
        res = SciMLProblemReservoir(
            prob, TerminalStateSampling(), tspan, Tsit5();
            reltol = 1.0e-8, abstol = 1.0e-10,
        )
        rc = ReservoirComputer(res, LinearReadout(res_dim => dim))
        ps, st0 = setup(MersenneTwister(0), rc)
        data = randn(rng, dim, T_steps)
        init = data[:, end]

        states, st1 = collectstates(rc, data, ps, st0)
        @test first(st1.reservoir.carry) ≈ states[:, end]

        cold, _ = predict(rc, steps, ps, st0; initialdata = init)
        warm, _ = predict(rc, steps, ps, st1; initialdata = init)
        @test cold ≉ warm

        st_forced = merge(st0, (reservoir = (; carry = (copy(prob.u0),)),))
        from_u0, _ = predict(rc, steps, ps, st_forced; initialdata = init)
        @test from_u0 ≈ cold
    end

    # ---------------------------------------------------------------------------
    # 6. State modifiers compose with the continuous path
    #
    # `state_modifiers` must compose with the continuous reservoir the same
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
            prob,
            TerminalStateSampling(),
            tspan,
            Tsit5();
            reltol = 1.0e-8,
            abstol = 1.0e-10,
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
            prob,
            TerminalStateSampling(),
            tspan,
            Tsit5();
            reltol = 1.0e-8,
            abstol = 1.0e-10,
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
                placeholder,
                sampler,
                tspan,
                Tsit5();
                (badkw => true,)...,
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
        res_eq = SciMLProblemReservoir(prob, TerminalStateSampling(), (1.0, 1.0), Tsit5())
        rc_eq = ReservoirComputer(res_eq, LinearReadout(res_dim => 1))
        ps, st = setup(MersenneTwister(0), rc_eq)
        @test_throws ArgumentError collectstates(rc_eq, data, ps, st)
        @test_throws ArgumentError predict(rc_eq, 3, ps, st; initialdata = randn(in_dim))

        # Backward interval
        res_back = SciMLProblemReservoir(prob, TerminalStateSampling(), (1.0, 0.0), Tsit5())
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

        prob_bad =
            ODEProblem(rhs_bad!, zeros(res_dim), (0.0, 1.0), (input = "already taken",))
        res = SciMLProblemReservoir(prob_bad, TerminalStateSampling(), (0.0, 1.0), Tsit5())
        rc = ReservoirComputer(res, LinearReadout(res_dim => 1))
        ps, st = setup(MersenneTwister(0), rc)
        data = randn(rng, 1, 4)
        @test_throws ArgumentError collectstates(rc, data, ps, st)
    end

    # ---------------------------------------------------------------------------
    # 11. `prob.p` accepted forms: NamedTuple / nothing / NullParameters
    #
    # `__to_namedtuple` advertises three valid inputs and rejects anything
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
                prob,
                TerminalStateSampling(),
                tspan,
                Tsit5();
                reltol = 1.0e-8,
                abstol = 1.0e-10,
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
            prob,
            TerminalStateSampling(),
            (0.0, 1.0),
            Tsit5();
            reltol = 1.0e-6,
        )
        rc = ReservoirComputer(res, LinearReadout(1 => 1))
        ps, st = setup(MersenneTwister(0), rc)
        data = ones(1, 4)
        @test_throws ArgumentError collectstates(rc, data, ps, st)
    end

    @testset "input_dt makes the time step independent of the horizon" begin
        rng = MersenneTwister(21)
        in_dim, res_dim, T_train, steps = 2, 6, 12, 7
        prob, = build_esn_problem(rng, in_dim, res_dim, (0.0, 1.0))
        kw = (; reltol = 1.0e-10, abstol = 1.0e-12)
        make(tspan; kwargs...) = ReservoirComputer(
            SciMLProblemReservoir(
                remake(prob; tspan), TerminalStateSampling(), tspan, Tsit5();
                kw..., kwargs...
            ),
            LinearReadout(res_dim => in_dim),
        )
        data = randn(rng, in_dim, T_train)
        Δt = 0.5

        rc_default = make((0.0, T_train * Δt))
        rc_dt = make((0.0, 1.0); input_dt = Δt)
        ps, st = setup(MersenneTwister(0), rc_default)

        s_default, _ = collectstates(rc_default, data, ps, st)
        s_dt, _ = collectstates(rc_dt, data, ps, st)
        @test s_dt ≈ s_default atol = 1.0e-8

        rc_ref = make((0.0, steps * Δt))
        x0 = data[:, end]
        out_ref, _ = predict(rc_ref, steps, ps, st; initialdata = x0)
        out_dt, _ = predict(rc_dt, steps, ps, st; initialdata = x0)
        out_default, _ = predict(rc_default, steps, ps, st; initialdata = x0)
        @test out_dt ≈ out_ref atol = 1.0e-8
        @test !isapprox(out_default, out_ref; atol = 1.0e-3)

        rc_old = make((0.0, 1.0))
        s_old, _ = collectstates(rc_old, data, ps, st)
        @test collectstates(make((0.0, 1.0); input_dt = nothing), data, ps, st)[1] == s_old
    end

    @testset "input_dt: user dt still reaches solve" begin
        rng = MersenneTwister(22)
        prob, = build_esn_problem(rng, 2, 4, (0.0, 3.0))
        data = randn(rng, 2, 3)
        make(dt) = ReservoirComputer(
            SciMLProblemReservoir(
                prob, TerminalStateSampling(), (0.0, 3.0), Euler();
                input_dt = 1.0, dt = dt
            ),
            LinearReadout(4 => 2),
        )
        rc_a, rc_b = make(0.5), make(0.1)
        ps, st = setup(MersenneTwister(0), rc_a)
        @test :dt in keys(rc_a.reservoir.kwargs)
        @test !(:input_dt in keys(rc_a.reservoir.kwargs))
        sa, _ = collectstates(rc_a, data, ps, st)
        sb, _ = collectstates(rc_b, data, ps, st)
        @test !isapprox(sa, sb; atol = 1.0e-4)
    end

    struct PlainContinuousReservoir <: AbstractSciMLProblemReservoir
        prob
        sampler
        tspan
        args
        kwargs
    end

    struct DtContinuousReservoir <: AbstractSciMLProblemReservoir
        prob
        sampler
        tspan
        args
        kwargs
        input_dt
    end

    @testset "custom AbstractSciMLProblemReservoir subtypes and input_dt" begin
        rng = MersenneTwister(31)
        in_dim, res_dim, T_train, steps = 2, 5, 9, 6
        tspan = (0.0, Float64(T_train))
        prob, = build_esn_problem(rng, in_dim, res_dim, tspan)
        args = (Tsit5(),)
        kwargs = (; reltol = 1.0e-10, abstol = 1.0e-12)
        sampler = TerminalStateSampling()
        data = randn(rng, in_dim, T_train)
        x0 = data[:, end]
        readout = LinearReadout(res_dim => in_dim)

        builtin = ReservoirComputer(
            SciMLProblemReservoir(prob, sampler, tspan, args...; kwargs...), readout
        )
        plain = ReservoirComputer(
            PlainContinuousReservoir(prob, sampler, tspan, args, kwargs), readout
        )
        ps, st = setup(MersenneTwister(0), builtin)
        s_builtin, _ = collectstates(builtin, data, ps, st)
        s_plain, _ = collectstates(plain, data, ps, st)
        @test s_plain == s_builtin
        out_builtin, _ = predict(builtin, steps, ps, st; initialdata = x0)
        out_plain, _ = predict(plain, steps, ps, st; initialdata = x0)
        @test out_plain == out_builtin

        with_dt = ReservoirComputer(
            DtContinuousReservoir(prob, sampler, (0.0, 1.0), args, kwargs, 1.0), readout
        )
        out_dt, _ = predict(with_dt, steps, ps, st; initialdata = x0)
        ref = ReservoirComputer(
            SciMLProblemReservoir(
                prob, sampler, (0.0, Float64(steps)), args...; kwargs...
            ),
            readout,
        )
        out_ref, _ = predict(ref, steps, ps, st; initialdata = x0)
        @test out_dt == out_ref
        @test !isapprox(out_plain, out_ref; atol = 1.0e-3)

        no_dt = ReservoirComputer(
            DtContinuousReservoir(prob, sampler, tspan, args, kwargs, nothing), readout
        )
        @test collectstates(no_dt, data, ps, st)[1] == s_builtin
    end

    @testset "input_dt grids: precision and integer input_dt" begin
        rng = MersenneTwister(41)
        prob, = build_esn_problem(rng, 1, 3, (0.0, 1.0))
        ext = Base.get_extension(ReservoirComputing, :RCODEReservoirExt)
        n = 3000
        res = SciMLProblemReservoir(
            prob, TerminalStateSampling(), (0.0, 1.0), Euler();
            input_dt = Float16(0.01), dt = 0.005
        )
        ts = ext.__input_grid(
            res.tspan, res.input_dt, n
        )
        @test allunique(ts[2])
        @test all(>(0), diff(ts[2]))
        win = ext.__window_grid(
            res.tspan, res.input_dt, n
        )
        @test length(win) == n + 1
        @test all(>(0), diff(win))

        data = randn(rng, 1, 5)
        make(dt_in, tspan) = ReservoirComputer(
            SciMLProblemReservoir(
                prob, TerminalStateSampling(), tspan, Tsit5(); input_dt = dt_in
            ),
            LinearReadout(3 => 1),
        )
        rc_int, rc_float = make(1, (0, 5)), make(1.0, (0.0, 5.0))
        ps, st = setup(MersenneTwister(0), rc_float)
        @test collectstates(rc_int, data, ps, st)[1] == collectstates(rc_float, data, ps, st)[1]
        x0 = data[:, end]
        @test predict(rc_int, 4, ps, st; initialdata = x0)[1] ==
            predict(rc_float, 4, ps, st; initialdata = x0)[1]
    end
end
