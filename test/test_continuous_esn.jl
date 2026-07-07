using Test
using Random
using LinearAlgebra
using Statistics
using ReservoirComputing
using OrdinaryDiffEq
using SciMLBase
using DataInterpolations

@testset "ContinuousESN: construction + parameter shapes" begin
    rng = MersenneTwister(0)
    in_dim, res_dim, out_dim = 3, 50, 2
    esn = ContinuousESN(in_dim, res_dim, out_dim, (0.0, 5.0), Tsit5())

    @test esn isa ContinuousESN
    @test propertynames(esn) == (:reservoir, :states_modifiers, :readout)
    @test esn.reservoir isa ContinuousESNCell
    @test esn.readout isa LinearReadout
    @test isempty(esn.states_modifiers)

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
            3, 5, 2, (0.0, 1.0), Tsit5(); (badkw => true,)...
        )
    end
end

@testset "ContinuousESN: bias toggle" begin
    rng = MersenneTwister(1)
    in_dim, res_dim, out_dim = 2, 10, 1
    esn_b = ContinuousESN(
        in_dim, res_dim, out_dim, (0.0, 1.0), Tsit5(); use_bias = true
    )
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
        in_dim, res_dim, out_dim, (0.0, 2.0), Tsit5();
        use_bias = true, init_bias = nonzero_bias,
        reltol = 1.0e-8, abstol = 1.0e-10
    )
    esn_no_bias = ContinuousESN(
        in_dim, res_dim, out_dim, (0.0, 2.0), Tsit5();
        use_bias = false, reltol = 1.0e-8, abstol = 1.0e-10
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
        in_dim, res_dim, out_dim, (0.0, 3.0), Tsit5();
        reltol = 1.0e-8, abstol = 1.0e-10
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
            in_dim, res_dim, out_dim, tspan, Euler();
            use_bias = true, dt = α,
            init_reservoir = dense_init_f64,
            init_input = init_input_f64,
            init_bias = init_bias_f64
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
        in_dim, res_dim, out_dim, (0.0, 2.0), Tsit5();
        reltol = 1.0e-8, abstol = 1.0e-10
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
        dim, res_dim, dim, (0.0, 1.0), Tsit5();
        reltol = 1.0e-8, abstol = 1.0e-10
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
        in_dim, res_dim, out_dim, (0.0, 1.0), Tsit5();
        reltol = 1.0e-8, abstol = 1.0e-10
    )
    esn_mod = ContinuousESN(
        in_dim, res_dim, out_dim, (0.0, 1.0), Tsit5();
        reltol = 1.0e-8, abstol = 1.0e-10,
        state_modifiers = (NLAT2(),)
    )

    ps_p, st_p = setup(MersenneTwister(0), esn_plain)
    ps_m, st_m = setup(MersenneTwister(0), esn_mod)
    data = randn(Float32, in_dim, T_steps)

    sp, _ = collectstates(esn_plain, data, ps_p, st_p)
    sm, _ = collectstates(esn_mod, data, ps_m, st_m)
    @test size(sm) == size(sp)
    @test all(isfinite, sm)
    @test sm != sp
end

@testset "ContinuousESN: custom init eltype propagates" begin
    rng = MersenneTwister(43)
    init_input_f64(rng, d...) = scaled_rand(rng, Float64, d...)
    init_res_f64(rng, d...) = rand_sparse(rng, Float64, d...)
    esn = ContinuousESN(
        2, 32, 1, (0.0, 1.0), Tsit5();
        init_input = init_input_f64, init_reservoir = init_res_f64
    )
    ps, _ = setup(rng, esn)
    @test eltype(ps.reservoir.input_matrix) == Float64
    @test eltype(ps.reservoir.reservoir_matrix) == Float64
end
