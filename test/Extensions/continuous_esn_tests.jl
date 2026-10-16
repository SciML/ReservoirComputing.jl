# Continuous ESN extension contracts.
begin
    using Test
    using Random
    using LinearAlgebra
    using Statistics
    using ReservoirComputing
    using LuxCore: setup
    using OrdinaryDiffEq
    # Euler lives in OrdinaryDiffEqLowOrderRK under OrdinaryDiffEq v7+
    if !isdefined(@__MODULE__, :Euler)
        using OrdinaryDiffEqLowOrderRK: Euler
    end
    using SciMLBase
    using DataInterpolations

    @testset "ContinuousESN: construction + parameter shapes" begin
        rng = MersenneTwister(0)
        in_dim, res_dim, out_dim = 3, 50, 2
        esn = ContinuousESN(in_dim, res_dim, out_dim, (0.0, 5.0), Tsit5())

        @test esn isa ContinuousESN
        @test propertynames(esn) == (:reservoir, :state_modifiers, :readout)
        @test esn.reservoir isa ContinuousESNCell
        @test esn.readout isa LinearReadout
        @test isempty(esn.state_modifiers)

        ps, st = setup(rng, esn)
        @test size(ps.reservoir.input_matrix) == (res_dim, in_dim)
        @test size(ps.reservoir.reservoir_matrix) == (res_dim, res_dim)
        @test !haskey(ps.reservoir, :bias)
        @test all(isfinite, ps.reservoir.input_matrix)
        @test all(isfinite, ps.reservoir.reservoir_matrix)
        @test size(ps.readout.weight) == (out_dim, res_dim)
    end

    @testset "ContinuousESN: construction validation" begin
        @test_throws ArgumentError ContinuousESN(0, 5, 2, (0.0, 1.0), Tsit5())
        @test_throws ArgumentError ContinuousESN(3, 0, 2, (0.0, 1.0), Tsit5())
        @test_throws ArgumentError ContinuousESN(3, 5, 0, (0.0, 1.0), Tsit5())
        @test_throws ArgumentError ContinuousESN(3, 5, 2, (0.0, 0.0), Tsit5())
        @test_throws ArgumentError ContinuousESN(3, 5, 2, (1.0, 0.0), Tsit5())
        @test_throws ArgumentError ContinuousESN(3, 5, 2, (0.0, Inf), Tsit5())
        @test_throws ArgumentError ContinuousESN(3, 5, 2, (-Inf, 1.0), Tsit5())
        @test_throws ArgumentError ContinuousESN(3, 5, 2, (0.0, NaN), Tsit5())
        @test_throws ArgumentError ContinuousESN(3, 5, 2, (1.0,), Tsit5())
        @test_throws ArgumentError ContinuousESN(3, 5, 2, (0.0, 1.0, 2.0), Tsit5())
        for badkw in (:saveat, :save_everystep, :dense)
            @test_throws ArgumentError ContinuousESN(
                3,
                5,
                2,
                (0.0, 1.0),
                Tsit5();
                (badkw => true,)...,
            )
        end
    end

    @testset "ContinuousESN: bias toggle" begin
        rng = MersenneTwister(1)
        in_dim, res_dim, out_dim = 2, 10, 1
        esn_b =
            ContinuousESN(in_dim, res_dim, out_dim, (0.0, 1.0), Tsit5(); use_bias = true)
        ps, st = setup(rng, esn_b)
        @test haskey(ps.reservoir, :bias)
        @test length(ps.reservoir.bias) == res_dim
        @test all(==(0), ps.reservoir.bias)
    end

    @testset "ContinuousESN: bias path under Tsit5" begin
        rng = MersenneTwister(101)
        in_dim, res_dim, out_dim, T_steps = 2, 20, 1, 12
        nonzero_bias(rng, d...) = 0.5f0 .* randn(rng, Float32, d...)
        esn_with_bias = ContinuousESN(
            in_dim,
            res_dim,
            out_dim,
            (0.0, 2.0),
            Tsit5();
            use_bias = true,
            init_bias = nonzero_bias,
            reltol = 1.0e-8,
            abstol = 1.0e-10,
        )
        esn_no_bias = ContinuousESN(
            in_dim,
            res_dim,
            out_dim,
            (0.0, 2.0),
            Tsit5();
            use_bias = false,
            reltol = 1.0e-8,
            abstol = 1.0e-10,
        )
        ps_b, st_b = setup(MersenneTwister(0), esn_with_bias)
        ps_n, st_n = setup(MersenneTwister(0), esn_no_bias)
        data = randn(Float32, in_dim, T_steps)

        s_b, _ = collectstates(esn_with_bias, data, ps_b, st_b)
        s_n, _ = collectstates(esn_no_bias, data, ps_n, st_n)
        @test all(isfinite, s_b)
        @test s_b != s_n
    end

    @testset "ContinuousESN: forward (collectstates)" begin
        rng = MersenneTwister(7)
        in_dim, res_dim, out_dim, T_steps = 2, 16, 1, 25
        esn = ContinuousESN(
            in_dim,
            res_dim,
            out_dim,
            (0.0, 3.0),
            Tsit5();
            reltol = 1.0e-8,
            abstol = 1.0e-10,
        )
        ps, st = setup(rng, esn)
        data = randn(Float32, in_dim, T_steps)

        s1, _ = collectstates(esn, data, ps, st)
        s2, _ = collectstates(esn, data, ps, st)
        @test size(s1) == (res_dim, T_steps)
        @test all(isfinite, s1)
        @test s1 ≈ s2
    end

    @testset "ContinuousESN: Euler equivalence with discrete leaky ESN" begin
        dense_init_f64(rng, d...) = rand_sparse(rng, Float64, d...; sparsity = 0.5)
        init_input_f64(rng, d...) = scaled_rand(rng, Float64, d...)
        init_bias_f64(rng, d...) = zeros(Float64, d...)
        for α in (1.0, 0.5)
            rng = MersenneTwister(13)
            in_dim, res_dim, out_dim, T_steps = 2, 16, 1, 12
            tspan = (0.0, T_steps * α)

            esn = ContinuousESN(
                in_dim,
                res_dim,
                out_dim,
                tspan,
                Euler();
                use_bias = true,
                dt = α,
                init_reservoir = dense_init_f64,
                init_input = init_input_f64,
                init_bias = init_bias_f64,
            )
            ps, st = setup(MersenneTwister(0), esn)

            data = randn(rng, Float64, in_dim, T_steps)
            cont_states, _ = collectstates(esn, data, ps, st)

            W_r = ps.reservoir.reservoir_matrix
            W_in = ps.reservoir.input_matrix
            b = ps.reservoir.bias
            disc_states = zeros(Float64, res_dim, T_steps)
            x = zeros(Float64, res_dim)
            for (k, u_col) in enumerate(eachcol(data))
                x = (1 - α) .* x .+ α .* tanh.(W_r * x .+ W_in * u_col .+ b)
                disc_states[:, k] = x
            end
            @test cont_states ≈ disc_states atol = 1.0e-10
        end
    end

    @testset "ContinuousESN: teacher-forced predict" begin
        rng = MersenneTwister(21)
        in_dim, res_dim, out_dim, T_steps = 2, 12, 3, 10
        esn = ContinuousESN(
            in_dim,
            res_dim,
            out_dim,
            (0.0, 2.0),
            Tsit5();
            reltol = 1.0e-8,
            abstol = 1.0e-10,
        )
        ps, st = setup(rng, esn)

        data = randn(Float32, in_dim, T_steps)
        tf1, _ = predict(esn, data, ps, st)
        tf2, _ = predict(esn, data, ps, st)
        @test size(tf1) == (out_dim, T_steps)
        @test all(isfinite, tf1)
        @test tf1 ≈ tf2
    end

    @testset "ContinuousESN: autoregressive predict" begin
        rng = MersenneTwister(22)
        # Autoregressive rollout feeds outputs back as inputs, so in_dim == out_dim.
        dim, res_dim, steps = 3, 12, 5
        esn = ContinuousESN(
            dim,
            res_dim,
            dim,
            (0.0, 1.0),
            Tsit5();
            reltol = 1.0e-8,
            abstol = 1.0e-10,
        )
        ps, st = setup(rng, esn)

        init = randn(Float32, dim)
        ar1, _ = predict(esn, steps, ps, st; initialdata = init)
        ar2, _ = predict(esn, steps, ps, st; initialdata = init)
        @test size(ar1) == (dim, steps)
        @test all(isfinite, ar1)
        @test ar1 ≈ ar2
    end

    @testset "ContinuousESN: state modifiers compose" begin
        rng = MersenneTwister(31)
        in_dim, res_dim, out_dim, T_steps = 2, 10, 1, 8
        esn_plain = ContinuousESN(
            in_dim,
            res_dim,
            out_dim,
            (0.0, 1.0),
            Tsit5();
            reltol = 1.0e-8,
            abstol = 1.0e-10,
        )
        esn_mod = ContinuousESN(
            in_dim,
            res_dim,
            out_dim,
            (0.0, 1.0),
            Tsit5();
            reltol = 1.0e-8,
            abstol = 1.0e-10,
            state_modifiers = (NLAT2(),),
        )
        esn_extended = ContinuousESN(
            in_dim,
            res_dim,
            out_dim,
            (0.0, 1.0),
            Tsit5();
            reltol = 1.0e-8,
            abstol = 1.0e-10,
            state_modifiers = (Extend(Collect()),),
        )

        ps_p, st_p = setup(MersenneTwister(0), esn_plain)
        ps_m, st_m = setup(MersenneTwister(0), esn_mod)
        ps_e, st_e = setup(MersenneTwister(0), esn_extended)
        data = randn(Float32, in_dim, T_steps)

        sp, _ = collectstates(esn_plain, data, ps_p, st_p)
        sm, _ = collectstates(esn_mod, data, ps_m, st_m)
        se, _ = collectstates(esn_extended, data, ps_e, st_e)
        @test size(sm) == size(sp)
        @test all(isfinite, sm)
        @test sm != sp
        @test se[1:in_dim, :] == data
        @test size(ps_e.readout.weight, 2) == size(se, 1)
    end

    @testset "ContinuousESN: custom init eltype propagates" begin
        rng = MersenneTwister(43)
        init_input_f64(rng, d...) = scaled_rand(rng, Float64, d...)
        init_res_f64(rng, d...) = rand_sparse(rng, Float64, d...)
        esn = ContinuousESN(
            2,
            32,
            1,
            (0.0, 1.0),
            Tsit5();
            init_input = init_input_f64,
            init_reservoir = init_res_f64,
        )
        ps, _ = setup(rng, esn)
        @test eltype(ps.reservoir.input_matrix) == Float64
        @test eltype(ps.reservoir.reservoir_matrix) == Float64
    end

    @testset "ContinuousESN: resetcarry! with init_carry" begin
        n, T_steps = 12, 8
        esn = ContinuousESN(2, n, 1, (0.0, 1.0), Tsit5(); init_state = zeros32)
        ps, st0 = setup(MersenneTwister(2), esn)
        data = randn(MersenneTwister(1), Float32, 2, T_steps)
        const03 = (rng, d...) -> fill(0.3f0, d...)
        run = st -> first(collectstates(esn, data, ps, deepcopy(st)))
        base = run(st0)
        _, st1 = collectstates(esn, data, ps, deepcopy(st0))
        manual = merge(st0, (reservoir = (carry = (fill(0.3f0, n),),),))
        for st in (st0, st1)
            st_zero = resetcarry!(MersenneTwister(3), esn, deepcopy(st); init_carry = zeros32)
            @test first(st_zero.reservoir.carry) isa Vector
            @test run(st_zero) == base
            st_const = resetcarry!(MersenneTwister(3), esn, deepcopy(st); init_carry = const03)
            @test first(st_const.reservoir.carry) == fill(0.3f0, n)
            @test run(st_const) == run(manual)
            @test run(st_const) != base
            st_clear = resetcarry!(MersenneTwister(3), esn, deepcopy(st))
            @test run(st_clear) == base
        end
    end

    @testset "ContinuousESN: autoregressive predict uses carry" begin
        rng = MersenneTwister(99)
        dim, res_dim, T_steps, steps = 2, 16, 20, 5
        esn = ContinuousESN(
            dim, res_dim, dim, (0.0, 2.0), Tsit5();
            reltol = 1.0e-8, abstol = 1.0e-10,
        )
        esn_mod = ContinuousESN(
            dim, res_dim, dim, (0.0, 2.0), Tsit5();
            reltol = 1.0e-8, abstol = 1.0e-10,
            state_modifiers = (NLAT2(),),
        )
        ps, st0 = setup(rng, esn)
        data = randn(Float32, dim, T_steps)
        init = data[:, end]

        states, st1 = collectstates(esn, data, ps, st0)
        @test first(st1.reservoir.carry) ≈ states[:, end]

        cold, _ = predict(esn, steps, ps, st0; initialdata = init)
        warm, st2 = predict(esn, steps, ps, st1; initialdata = init)
        @test cold ≉ warm
        @test first(st2.reservoir.carry) ≉ first(st1.reservoir.carry)

        st_clear = resetcarry!(MersenneTwister(0), esn, st1)
        @test get(st_clear.reservoir, :carry, nothing) === nothing
        cold2, _ = predict(esn, steps, ps, st_clear; initialdata = init)
        @test cold2 ≈ cold

        s_cont, _ = collectstates(esn, data, ps, st1)
        @test s_cont ≉ states

        ps_m, st_m = setup(MersenneTwister(0), esn_mod)
        sm, stm = collectstates(esn_mod, data, ps_m, st_m)
        @test first(stm.reservoir.carry) ≉ sm[:, end]
        @test length(first(stm.reservoir.carry)) == res_dim
    end

    @testset "ContinuousESN: activation and init_state are used" begin
        dim, res_dim = 2, 20
        data = randn(MersenneTwister(1), Float32, dim, 30)
        function sin_rhs!(dx, x, p, t)
            mul!(dx, p.reservoir_matrix, x)
            mul!(dx, p.input_matrix, p.input(t), true, true)
            @. dx = -x + sin(dx)
            return nothing
        end
        solve_kw = (; reltol = 1.0e-8, abstol = 1.0e-10)
        esn_tanh = ContinuousESN(dim, res_dim, dim, (0.0, 3.0), Tsit5(); solve_kw...)
        esn_sin = ContinuousESN(dim, res_dim, dim, sin, (0.0, 3.0), Tsit5(); solve_kw...)
        esn_ref = ContinuousESN(
            dim, res_dim, dim, (0.0, 3.0), Tsit5(); equations = sin_rhs!, solve_kw...
        )
        ps, st = setup(MersenneTwister(7), esn_tanh)
        s_tanh, _ = collectstates(esn_tanh, data, ps, st)
        s_sin, _ = collectstates(esn_sin, data, ps, st)
        s_ref, _ = collectstates(esn_ref, data, ps, st)
        @test s_sin ≈ s_ref
        @test !isapprox(s_sin, s_tanh; rtol = 1.0e-3)
        ps_act = merge(ps, (reservoir = merge(ps.reservoir, (activation = sin,)),))
        @test first(collectstates(esn_tanh, data, ps_act, st)) ≈ s_sin

        esn_half = ContinuousESN(
            dim, res_dim, dim, (0.0, 3.0), Tsit5();
            init_state = (rng, dims...) -> fill(0.5f0, dims...), solve_kw...
        )
        s_half, _ = collectstates(esn_half, data, ps, st)
        @test !isapprox(s_half[:, 1], s_tanh[:, 1]; rtol = 1.0e-3)
        @test eltype(s_half) == eltype(ps.reservoir.input_matrix)

        esn_rand = ContinuousESN(
            dim, res_dim, dim, (0.0, 3.0), Tsit5(); init_state = randn32, solve_kw...
        )
        s_rand, st_rand = collectstates(esn_rand, data, ps, st)
        @test s_rand == first(collectstates(esn_rand, data, ps, st))
        @test s_rand != s_tanh
        @test st_rand.reservoir.rng != st.reservoir.rng
    end

    @testset "ContinuousESN: collectstates is reproducible with default_rng setup" begin
        esn = ContinuousESN(
            3, 30, 1, (0.0, 1.0), Tsit5();
            init_state = (rng, d...) -> rand(rng, Float32, d...),
        )
        ps, st = setup(Random.default_rng(), esn)
        data = randn(MersenneTwister(1), Float32, 3, 8)
        s1, _ = collectstates(esn, data, ps, deepcopy(st))
        s2, _ = collectstates(esn, data, ps, deepcopy(st))
        @test s1 == s2
    end
end
