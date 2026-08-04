begin
    using Test
    using Random
    using LinearAlgebra
    using ReservoirComputing
    using LuxCore: setup
    using OrdinaryDiffEq
    using DataInterpolations

    @testset "dale_sparse: Dale signs + shape" begin
        rng = MersenneTwister(0)
        n, n_exc = 20, 16
        W = dale_sparse(
            rng, Float64, n, n;
            excitatory_fraction = 0.8, sparsity = 0.3, radius = 0.9
        )
        @test size(W) == (n, n)
        @test all(>=(0), W[:, 1:n_exc])
        @test all(<=(0), W[:, (n_exc + 1):n])
        @test isapprox(maximum(abs, eigvals(W)), 0.9; rtol = 1.0e-4)
    end

    @testset "LIF ISI + refractory" begin
        τ_m, R_m, V_rest, V_th, V_reset = 0.02, 1.0, 0.0, 1.0, 0.0
        I = 1.5
        T_isi = τ_m * log((R_m * I - V_rest) / (R_m * I - V_th))
        τ_ref = 0.005
        spikes = Float64[]
        last_fire = Ref(-Inf)
        cond! = (out, u, t, integrator) -> (out[1] = u[1] - V_th; nothing)
        aff! = function (integrator, idx)
            t = integrator.t
            t <= last_fire[] && return nothing
            push!(spikes, t)
            last_fire[] = t + τ_ref
            integrator.u[1] = V_reset
            return nothing
        end
        cb = VectorContinuousCallback(cond!, aff!, 1; save_positions = (false, false))
        rhs! = function (du, u, p, t)
            du[1] = t < last_fire[] ? 0.0 : (-(u[1] - V_rest) + R_m * I) / τ_m
            return nothing
        end
        sol = solve(
            ODEProblem(rhs!, [0.0], (0.0, 0.5); callback = cb), Tsit5();
            reltol = 1.0e-10, abstol = 1.0e-12, dense = false
        )
        @test successful_retcode(sol)
        @test length(spikes) >= 5
        isis = diff(spikes)
        @test all(isi -> isi >= τ_ref - 1.0e-8, isis)
        @test all(isi -> isapprox(isi, T_isi + τ_ref; rtol = 0.02), isis)
    end

    @testset "LSM: construction" begin
        rng = MersenneTwister(1)
        lsm = LSM(2, 30, 1, (0.0, 1.0), Tsit5(); reltol = 1.0e-6, abstol = 1.0e-8)
        @test lsm isa LSM
        @test lsm.reservoir isa LSMCell
        @test lsm.readout isa LinearReadout
        ps, st = setup(rng, lsm)
        @test size(ps.reservoir.input_matrix) == (30, 2)
        @test size(ps.reservoir.reservoir_matrix) == (30, 30)
        @test size(ps.readout.weight) == (1, 30)
        @test haskey(st.reservoir, :rng)
    end

    @testset "LSM: construction validation" begin
        @test_throws ArgumentError LSM(0, 5, 1, (0.0, 1.0), Tsit5())
        @test_throws ArgumentError LSM(2, 5, 1, (1.0, 0.0), Tsit5())
        @test_throws ArgumentError LSM(2, 5, 1, (0.0, 1.0), Tsit5(); saveat = 0.1)
        @test_throws ArgumentError LSM(2, 5, 1, (0.0, 1.0), Tsit5(); callback = nothing)
    end

    @testset "LSM: collectstates shape + determinism" begin
        rng = MersenneTwister(2)
        n_in, n_res, n_out, T_steps = 2, 25, 1, 20
        lsm = LSM(
            n_in, n_res, n_out, (0.0, Float64(T_steps)), Tsit5();
            neuron = LIFCell(; τ_m = 0.02, τ_ref = 0.002, τ_syn = 0.005),
            spike_readout = ExponentialFilterReadout(0.05),
            init_input = (rng, d...) -> scaled_rand(rng, Float64, d...; scaling = 0.5),
            init_reservoir = (rng, d...) -> dale_sparse(
                rng, Float64, d...; radius = 0.5, sparsity = 0.2
            ),
            init_state = (rng, d...) -> zeros(Float64, d...),
            reltol = 1.0e-6,
            abstol = 1.0e-8,
            dtmax = 5.0e-4,
        )
        ps, st = setup(rng, lsm)
        data = 0.3 .+ 0.2 .* rand(rng, Float64, n_in, T_steps)
        s1, st1 = collectstates(lsm, data, ps, st)
        s2, _ = collectstates(lsm, data, ps, st)
        @test size(s1) == (n_res, T_steps)
        @test all(isfinite, s1)
        @test s1 ≈ s2
    end

    @testset "LSM: SpikeCountReadout + AR rejection" begin
        rng = MersenneTwister(3)
        lsm = LSM(
            1, 15, 1, (0.0, 5.0), Tsit5();
            spike_readout = SpikeCountReadout(),
            init_input = (rng, d...) -> scaled_rand(rng, Float64, d...; scaling = 1.0),
            init_reservoir = (rng, d...) -> dale_sparse(rng, Float64, d...; radius = 0.3),
            init_state = (rng, d...) -> zeros(Float64, d...),
            dtmax = 5.0e-4,
        )
        ps, st = setup(rng, lsm)
        data = ones(Float64, 1, 10)
        s, _ = collectstates(lsm, data, ps, st)
        @test size(s) == (15, 10)
        @test all(x -> x >= 0, s)
        @test_throws ArgumentError predict(lsm, 5, ps, st; initialdata = [1.0])
    end

    @testset "LSM: FilteredVoltageReadout + teacher-forced predict" begin
        rng = MersenneTwister(4)
        lsm = LSM(
            2, 20, 2, (0.0, 4.0), Tsit5();
            spike_readout = FilteredVoltageReadout(),
            init_input = (rng, d...) -> scaled_rand(rng, Float64, d...; scaling = 0.2),
            init_reservoir = (rng, d...) -> dale_sparse(rng, Float64, d...; radius = 0.4),
            init_state = (rng, d...) -> zeros(Float64, d...),
            dtmax = 5.0e-4,
        )
        ps, st = setup(rng, lsm)
        data = randn(rng, Float64, 2, 12)
        states, st = collectstates(lsm, data, ps, st)
        @test size(states) == (20, 12)
        y, _ = predict(lsm, data, ps, st)
        @test size(y) == (2, 12)
        @test all(isfinite, y)
    end

    @testset "LSM: PoissonRateEncoder smoke" begin
        rng = MersenneTwister(5)
        lsm = LSM(
            1, 12, 1, (0.0, 2.0), Tsit5();
            encoder = PoissonRateEncoder(; scale = 20.0, weight = 0.5),
            spike_readout = SpikeCountReadout(),
            init_input = (rng, d...) -> abs.(scaled_rand(rng, Float64, d...; scaling = 1.0)),
            init_reservoir = (rng, d...) -> dale_sparse(rng, Float64, d...; radius = 0.3),
            init_state = (rng, d...) -> zeros(Float64, d...),
            dtmax = 5.0e-4,
        )
        ps, st = setup(rng, lsm)
        data = ones(Float64, 1, 8)
        s1, st1 = collectstates(lsm, data, ps, st)
        s2, _ = collectstates(lsm, data, ps, st)
        @test size(s1) == (12, 8)
        @test s1 ≈ s2
    end

    @testset "LSM: train + AR predict" begin
        rng = MersenneTwister(6)
        n_in, n_res, n_out, T_steps = 2, 25, 2, 30
        lsm = LSM(
            n_in, n_res, n_out, (0.0, Float64(T_steps)), Tsit5();
            spike_readout = ExponentialFilterReadout(0.05),
            init_input = (rng, d...) -> fill(1.0, d...),
            init_reservoir = (rng, d...) -> dale_sparse(
                rng, Float64, d...; radius = 0.3, sparsity = 0.2
            ),
            init_state = (rng, d...) -> zeros(Float64, d...),
            dtmax = 5.0e-4,
            maxiters = 1_000_000,
        )
        ps, st = setup(rng, lsm)
        data = 0.8 .* ones(Float64, n_in, T_steps)
        target = circshift(data, (0, -1))
        ps, st = train(
            lsm, data[:, 1:(end - 1)], target[:, 1:(end - 1)], ps, st;
            washout = 3, objective = RidgeRegression(1.0e-2)
        )
        @test size(ps.readout.weight) == (n_out, n_res)
        y_tf, _ = predict(lsm, data[:, 1:(end - 1)], ps, st)
        @test size(y_tf) == (n_out, T_steps - 1)
        y_ar, _ = predict(lsm, 5, ps, st; initialdata = data[:, end])
        @test size(y_ar) == (n_out, 5)
        @test all(isfinite, y_ar)
    end
end
