begin
    using Test
    using Random
    using LinearAlgebra
    using Statistics
    using ReservoirComputing
    using LuxCore: setup
    using OrdinaryDiffEq
    using DataInterpolations

    const f64_zeros = (rng, d...) -> zeros(Float64, d...)
    const f64_ones = (rng, d...) -> ones(Float64, d...)

    struct _ClosedNeuron <: AbstractSpikingNeuron end
    struct _ClosedEncoder <: AbstractInputEncoder end
    struct _ClosedFeature <: AbstractSpikeFeature end

    function _lsm_f64(
            in_dim, res_dim, out_dim, tspan, args...;
            neuron = LIFNeuron(),
            encoder = CurrentInjection(),
            feature_map = MembraneVoltageFeature(),
            init_input = f64_ones,
            init_reservoir = f64_zeros,
            init_state = f64_zeros,
            kwargs...,
        )
        return LSM(
            in_dim, res_dim, out_dim, tspan, args...;
            neuron, encoder, feature_map, init_input, init_reservoir, init_state,
            kwargs...,
        )
    end

    function _set_res!(ps; Win, W)
        return merge(
            ps,
            (reservoir = merge(ps.reservoir, (input_matrix = Win, reservoir_matrix = W)),),
        )
    end

    function _isi(tau_m, r_m, I, v_th = 1.0, v_rest = 0.0)
        return tau_m * log((r_m * I - v_rest) / (r_m * I - v_th))
    end

    @testset "dale_sparse: Dale structure" begin
        rng = MersenneTwister(0)
        n = 40
        W = dale_sparse(
            rng, Float64, n, n;
            excitatory_fraction = 0.8, sparsity = 0.25, radius = 0.85,
            ei_weight_ratio = 1.5,
        )
        n_exc = round(Int, 0.8 * n)
        @test size(W) == (n, n)
        @test all(>=(0), W[:, 1:n_exc])
        @test all(<=(0), W[:, (n_exc + 1):n])
        @test isapprox(maximum(abs, eigvals(W)), 0.85; rtol = 1.0e-4)
        @test any(!iszero, W)
    end

    @testset "LIF: analytical ISI through LSM" begin
        tau_m, r_m, I = 0.02, 1.0, 1.5
        T_isi = _isi(tau_m, r_m, I)
        T_end = 20 * T_isi
        n_win = 200
        lsm = _lsm_f64(
            1, 1, 1, (0.0, T_end), Tsit5();
            neuron = LIFNeuron(;
                tau_m = tau_m, r_m = r_m, v_rest = 0.0, v_th = 1.0,
                v_reset = 0.0, tau_ref = 0.0, tau_syn = 0.005,
            ),
            feature_map = SpikeCountFeatures(),
            reltol = 1.0e-10, abstol = 1.0e-12, dtmax = T_isi / 20,
        )
        ps, st = setup(MersenneTwister(0), lsm)
        ps = _set_res!(ps; Win = reshape([I], 1, 1), W = reshape([0.0], 1, 1))
        counts, _ = collectstates(lsm, ones(Float64, 1, n_win), ps, st)
        n_spikes = sum(counts)
        expected = 1 + floor(Int, (T_end - T_isi) / T_isi + 1.0e-12)
        @test abs(n_spikes - expected) <= 1
        @test isapprox(T_end / n_spikes, T_isi; rtol = 0.08)
    end

    @testset "LIF: refractory through LSM" begin
        tau_m, r_m, I, tau_ref = 0.02, 1.0, 1.5, 0.005
        T_isi = _isi(tau_m, r_m, I)
        T_period = T_isi + tau_ref
        T_end = 15 * T_period
        n_win = 300
        lsm = _lsm_f64(
            1, 1, 1, (0.0, T_end), Tsit5();
            neuron = LIFNeuron(;
                tau_m = tau_m, r_m = r_m, v_rest = 0.0, v_th = 1.0,
                v_reset = 0.0, tau_ref = tau_ref, tau_syn = 0.005,
            ),
            feature_map = SpikeCountFeatures(),
            reltol = 1.0e-10, abstol = 1.0e-12, dtmax = tau_ref / 4,
        )
        ps, st = setup(MersenneTwister(1), lsm)
        ps = _set_res!(ps; Win = reshape([I], 1, 1), W = reshape([0.0], 1, 1))
        counts, _ = collectstates(lsm, ones(Float64, 1, n_win), ps, st)
        n_spikes = sum(counts)
        expected = 1 + floor(Int, (T_end - T_isi) / T_period + 1.0e-12)
        @test abs(n_spikes - expected) <= 1
        @test isapprox(T_end / n_spikes, T_period; rtol = 0.08)
        @test T_end / n_spikes > T_isi * 1.1
    end

    @testset "LIF: subthreshold V matches closed form" begin
        tau_m, r_m, I = 0.02, 1.0, 0.5
        V_ss = r_m * I
        @test V_ss < 1.0
        T_end = 0.2
        n_win = 40
        lsm = _lsm_f64(
            1, 1, 1, (0.0, T_end), Tsit5();
            neuron = LIFNeuron(;
                tau_m = tau_m, r_m = r_m, v_rest = 0.0, v_th = 1.0,
                v_reset = 0.0, tau_ref = 0.002, tau_syn = 0.005,
            ),
            feature_map = MembraneVoltageFeature(),
            reltol = 1.0e-10, abstol = 1.0e-12,
        )
        ps, st = setup(MersenneTwister(2), lsm)
        ps = _set_res!(ps; Win = reshape([I], 1, 1), W = reshape([0.0], 1, 1))
        V, _ = collectstates(lsm, ones(Float64, 1, n_win), ps, st)
        t_sample = collect(range(T_end / n_win, T_end; length = n_win))
        V_exact = V_ss .* (1 .- exp.(-t_sample ./ tau_m))
        @test maximum(abs, vec(V) .- V_exact) < 1.0e-5
    end

    @testset "LIF: voltage stays below threshold after reset" begin
        tau_m, r_m, I = 0.02, 1.0, 2.0
        T_isi = _isi(tau_m, r_m, I)
        T_end = 5 * T_isi
        n_win = 500
        lsm = _lsm_f64(
            1, 1, 1, (0.0, T_end), Tsit5();
            neuron = LIFNeuron(;
                tau_m = tau_m, r_m = r_m, v_rest = 0.0, v_th = 1.0,
                v_reset = 0.0, tau_ref = 0.0, tau_syn = 0.005,
            ),
            feature_map = MembraneVoltageFeature(),
            reltol = 1.0e-10, abstol = 1.0e-12, dtmax = T_isi / 50,
        )
        ps, st = setup(MersenneTwister(3), lsm)
        ps = _set_res!(ps; Win = reshape([I], 1, 1), W = reshape([0.0], 1, 1))
        V, _ = collectstates(lsm, ones(Float64, 1, n_win), ps, st)
        @test all(v -> v < 1.0 + 1.0e-6, V)
        @test minimum(V) < 0.25
        @test maximum(V) > 0.5
    end

    @testset "recurrent synapse: postsynaptic depolarization" begin
        tau_m, r_m, I, w, tau_syn = 0.02, 1.0, 2.0, 0.8, 0.01
        T_isi = _isi(tau_m, r_m, I)
        T_end = T_isi + 0.5 * tau_syn
        n_win = 25
        lsm = _lsm_f64(
            1, 2, 1, (0.0, T_end), Tsit5();
            neuron = LIFNeuron(;
                tau_m = tau_m, r_m = r_m, v_rest = 0.0, v_th = 1.0,
                v_reset = 0.0, tau_ref = 0.002, tau_syn = tau_syn,
            ),
            feature_map = MembraneVoltageFeature(),
            reltol = 1.0e-10, abstol = 1.0e-12, dtmax = 1.0e-4,
        )
        ps, st = setup(MersenneTwister(4), lsm)
        Win = reshape([I, 0.0], 2, 1)
        W = [0.0 0.0; w 0.0]
        ps = _set_res!(ps; Win = Win, W = W)
        V, _ = collectstates(lsm, ones(Float64, 1, n_win), ps, st)
        @test V[2, end] > 0.05
        @test V[2, end] > V[2, 1]
        @test any(>(0.5), V[1, :])
    end

    @testset "SpikeCountFeatures: rate and AR guard" begin
        tau_m, r_m, I = 0.02, 1.0, 2.0
        T_isi = _isi(tau_m, r_m, I)
        Δt = 3 * T_isi
        n_win = 8
        T_end = n_win * Δt
        lsm = _lsm_f64(
            1, 1, 1, (0.0, T_end), Tsit5();
            neuron = LIFNeuron(;
                tau_m = tau_m, r_m = r_m, v_rest = 0.0, v_th = 1.0,
                v_reset = 0.0, tau_ref = 0.0, tau_syn = 0.005,
            ),
            feature_map = SpikeCountFeatures(),
            reltol = 1.0e-10, abstol = 1.0e-12, dtmax = T_isi / 20,
        )
        ps, st = setup(MersenneTwister(5), lsm)
        ps = _set_res!(ps; Win = reshape([I], 1, 1), W = reshape([0.0], 1, 1))
        C, _ = collectstates(lsm, ones(Float64, 1, n_win), ps, st)
        @test all(c -> 2 <= c <= 4, C)
        @test sum(C) >= 2 * n_win
        @test isapprox(sum(C) / T_end, 1 / T_isi; rtol = 0.05)
        @test_throws ArgumentError predict(lsm, 3, ps, st; initialdata = [1.0])
    end

    @testset "ExponentialSpikeFilter: tracks spikes and decays" begin
        tau_m, r_m, I, filter_tau = 0.02, 1.0, 2.0, 0.03
        T_isi = _isi(tau_m, r_m, I)
        n_on, n_off = 25, 25
        n_win = n_on + n_off
        Δt = T_isi
        T_end = n_win * Δt
        lsm = _lsm_f64(
            1, 1, 1, (0.0, T_end), Tsit5();
            neuron = LIFNeuron(;
                tau_m = tau_m, r_m = r_m, v_rest = 0.0, v_th = 1.0,
                v_reset = 0.0, tau_ref = 0.0, tau_syn = 0.005,
            ),
            feature_map = ExponentialSpikeFilter(; filter_tau = filter_tau),
            reltol = 1.0e-10, abstol = 1.0e-12, dtmax = T_isi / 25,
        )
        ps, st = setup(MersenneTwister(6), lsm)
        ps = _set_res!(ps; Win = reshape([1.0], 1, 1), W = reshape([0.0], 1, 1))
        data = hcat(fill(I, 1, n_on), zeros(1, n_off))
        S, _ = collectstates(lsm, data, ps, st)
        s = vec(S)
        @test maximum(s[1:n_on]) > 0.5
        @test mean(s[1:n_on]) > mean(s[(n_on + 1):end])
        @test s[end] < s[n_on]
        @test s[end] < 0.5 * s[n_on]

        ps_sub = _set_res!(ps; Win = reshape([0.3], 1, 1), W = reshape([0.0], 1, 1))
        S0, _ = collectstates(lsm, ones(Float64, 1, n_win), ps_sub, st)
        @test maximum(abs, S0) < 1.0e-8
    end

    @testset "ESP: two initial voltages converge under same input" begin
        n = 20
        T_steps = 80
        tspan = (0.0, Float64(T_steps))
        lsm = _lsm_f64(
            2, n, 2, tspan, Tsit5();
            neuron = LIFNeuron(;
                tau_m = 0.02, r_m = 1.0, v_rest = 0.0, v_th = 1.0,
                v_reset = 0.0, tau_ref = 0.002, tau_syn = 0.01,
            ),
            feature_map = ExponentialSpikeFilter(; filter_tau = 0.05),
            init_input = (rng, d...) -> scaled_rand(rng, Float64, d...; scaling = 0.8),
            init_reservoir = (rng, d...) -> dale_sparse(
                rng, Float64, d...; radius = 0.4, sparsity = 0.15,
            ),
            reltol = 1.0e-7, abstol = 1.0e-9, dtmax = 5.0e-4, maxiters = 2_000_000,
        )
        ps, st = setup(MersenneTwister(7), lsm)
        data = 0.6 .+ 0.3 .* rand(MersenneTwister(8), Float64, 2, T_steps)

        init_a = (rng, d...) -> fill(0.0, d...)
        init_b = (rng, d...) -> fill(0.4, d...)
        lsm_a = _lsm_f64(
            2, n, 2, tspan, Tsit5();
            neuron = lsm.reservoir.neuron,
            feature_map = lsm.reservoir.feature_map,
            init_input = f64_ones, init_reservoir = f64_zeros, init_state = init_a,
            reltol = 1.0e-7, abstol = 1.0e-9, dtmax = 5.0e-4, maxiters = 2_000_000,
        )
        lsm_b = _lsm_f64(
            2, n, 2, tspan, Tsit5();
            neuron = lsm.reservoir.neuron,
            feature_map = lsm.reservoir.feature_map,
            init_input = f64_ones, init_reservoir = f64_zeros, init_state = init_b,
            reltol = 1.0e-7, abstol = 1.0e-9, dtmax = 5.0e-4, maxiters = 2_000_000,
        )
        ps_a, st_a = setup(MersenneTwister(7), lsm_a)
        ps_b, st_b = setup(MersenneTwister(7), lsm_b)
        ps_a = merge(ps_a, (reservoir = ps.reservoir,))
        ps_b = merge(ps_b, (reservoir = ps.reservoir,))

        Fa, _ = collectstates(lsm_a, data, ps_a, st_a)
        Fb, _ = collectstates(lsm_b, data, ps_b, st_b)
        early = mean(abs, Fa[:, 1:5] .- Fb[:, 1:5])
        late = mean(abs, Fa[:, (end - 4):end] .- Fb[:, (end - 4):end])
        @test late <= early + 1.0e-3 || late < 0.15
        @test late < 0.25
        @test all(isfinite, Fa) && all(isfinite, Fb)
    end

    @testset "teacher-forced predict matches readout∘states" begin
        n_in, n_res, n_out, T_steps = 2, 30, 2, 40
        lsm = _lsm_f64(
            n_in, n_res, n_out, (0.0, Float64(T_steps)), Tsit5();
            neuron = LIFNeuron(; tau_m = 0.02, tau_ref = 0.002, tau_syn = 0.008),
            feature_map = ExponentialSpikeFilter(; filter_tau = 0.04),
            init_input = (rng, d...) -> scaled_rand(rng, Float64, d...; scaling = 1.0),
            init_reservoir = (rng, d...) -> dale_sparse(
                rng, Float64, d...; radius = 0.45, sparsity = 0.2,
            ),
            reltol = 1.0e-7, abstol = 1.0e-9, dtmax = 5.0e-4, maxiters = 2_000_000,
        )
        ps, st = setup(MersenneTwister(9), lsm)
        data = 0.5 .+ 0.4 .* rand(MersenneTwister(10), Float64, n_in, T_steps)
        states, st2 = collectstates(lsm, data, ps, st)
        @test sum(abs, states) > 0
        y_tf, _ = predict(lsm, data, ps, st)
        y_man = ps.readout.weight * states
        @test y_tf ≈ y_man rtol = 1.0e-5
    end

    @testset "train improves one-step forecast over zero readout" begin
        n_in, n_res, n_out, T_steps = 2, 60, 2, 120
        lsm = _lsm_f64(
            n_in, n_res, n_out, (0.0, Float64(T_steps)), Tsit5();
            neuron = LIFNeuron(; tau_m = 0.02, tau_ref = 0.002, tau_syn = 0.008),
            feature_map = ExponentialSpikeFilter(; filter_tau = 0.05),
            init_input = (rng, d...) -> abs.(randn(rng, Float64, d...)) .+ 0.5,
            init_reservoir = (rng, d...) -> dale_sparse(
                rng, Float64, d...; radius = 0.5, sparsity = 0.15,
            ),
            reltol = 1.0e-6, abstol = 1.0e-8, dtmax = 5.0e-4, maxiters = 3_000_000,
        )
        ps, st = setup(MersenneTwister(11), lsm)
        rng = MersenneTwister(12)
        u = abs.(randn(rng, Float64, n_in, T_steps)) .+ 0.8
        y = similar(u)
        y[:, 1] .= u[:, 1]
        for t in 2:T_steps
            y[:, t] .= 0.6 .* u[:, t] .+ 0.4 .* reverse(u[:, t - 1])
        end

        states, _ = collectstates(lsm, u, ps, st)
        @test mean(abs, states) > 0.05

        nrmse0 = norm(zeros(n_out, T_steps) .- y) / norm(y .- mean(y; dims = 2))
        ps_tr, st_tr = train(
            lsm, u, y, ps, st;
            washout = 20, objective = RidgeRegression(1.0e-3),
        )
        y_pred, _ = predict(lsm, u, ps_tr, st_tr)
        nrmse = norm(y_pred[:, 21:end] .- y[:, 21:end]) /
            norm(y[:, 21:end] .- mean(y[:, 21:end]; dims = 2))
        @test nrmse < nrmse0
        @test nrmse < 1.0
        @test all(isfinite, y_pred)
    end

    @testset "AR predict is deterministic and finite" begin
        dim, n_res, steps = 2, 40, 15
        lsm = _lsm_f64(
            dim, n_res, dim, (0.0, Float64(steps)), Tsit5();
            neuron = LIFNeuron(; tau_m = 0.02, tau_ref = 0.002, tau_syn = 0.008),
            feature_map = ExponentialSpikeFilter(; filter_tau = 0.04),
            init_input = (rng, d...) -> scaled_rand(rng, Float64, d...; scaling = 1.0),
            init_reservoir = (rng, d...) -> dale_sparse(
                rng, Float64, d...; radius = 0.4, sparsity = 0.2,
            ),
            reltol = 1.0e-7, abstol = 1.0e-9, dtmax = 5.0e-4, maxiters = 2_000_000,
        )
        ps, st = setup(MersenneTwister(13), lsm)
        u = 0.7 .* ones(Float64, dim, 40)
        y = circshift(u, (0, -1))
        lsm_tr = _lsm_f64(
            dim, n_res, dim, (0.0, 40.0), Tsit5();
            neuron = lsm.reservoir.neuron,
            feature_map = lsm.reservoir.feature_map,
            init_input = f64_ones, init_reservoir = f64_zeros, init_state = f64_zeros,
            reltol = 1.0e-7, abstol = 1.0e-9, dtmax = 5.0e-4, maxiters = 2_000_000,
        )
        ps_tr, st_tr = setup(MersenneTwister(13), lsm_tr)
        ps_tr = merge(ps_tr, (reservoir = ps.reservoir,))
        ps_tr, st_tr = train(
            lsm_tr, u, y, ps_tr, st_tr; washout = 5, objective = RidgeRegression(1.0e-2),
        )
        ps = merge(ps, (readout = ps_tr.readout,))
        init = [0.7, 0.7]
        a1, _ = predict(lsm, steps, ps, st; initialdata = init)
        a2, _ = predict(lsm, steps, ps, st; initialdata = init)
        @test size(a1) == (dim, steps)
        @test a1 ≈ a2
        @test all(isfinite, a1)
    end

    @testset "AR predict uses carry" begin
        n, T_steps, steps = 8, 12, 4
        lsm = _lsm_f64(
            1, n, 1, (0.0, 0.03), Tsit5();
            neuron = LIFNeuron(; tau_m = 0.02, tau_ref = 0.002, tau_syn = 0.008),
            feature_map = MembraneVoltageFeature(),
            reltol = 1.0e-7, abstol = 1.0e-9, dtmax = 5.0e-4,
        )
        ps, st0 = setup(MersenneTwister(3), lsm)
        data = ones(Float64, 1, T_steps)
        states, st1 = collectstates(lsm, data, ps, st0)
        u_end = first(st1.reservoir.carry)
        @test length(u_end) == 2n
        @test u_end[1:n] ≈ states[:, end]

        s_cont, _ = collectstates(lsm, data, ps, st1)
        @test s_cont ≉ states

        init = [1.0]
        cold, _ = predict(lsm, steps, ps, st0; initialdata = init)
        warm, _ = predict(lsm, steps, ps, st1; initialdata = init)
        @test cold ≉ warm
    end

    @testset "PoissonRateEncoder: event drive produces spikes" begin
        n_win = 40
        T_end = 2.0
        lsm = _lsm_f64(
            1, 8, 1, (0.0, T_end), Tsit5();
            encoder = PoissonRateEncoder(; rate_scale = 60.0, synaptic_weight = 3.0),
            neuron = LIFNeuron(;
                tau_m = 0.02, r_m = 1.0, v_rest = 0.0, v_th = 1.0,
                v_reset = 0.0, tau_ref = 0.002, tau_syn = 0.02,
            ),
            feature_map = SpikeCountFeatures(),
            init_input = (rng, d...) -> ones(Float64, d...),
            init_reservoir = f64_zeros,
            reltol = 1.0e-7, abstol = 1.0e-9, dtmax = 5.0e-4, maxiters = 2_000_000,
        )
        ps, st = setup(MersenneTwister(14), lsm)
        data = ones(Float64, 1, n_win)
        c1, _ = collectstates(lsm, data, ps, st)
        c2, _ = collectstates(lsm, data, ps, st)
        @test c1 ≈ c2
        @test sum(c1) > 0
        c_hi, _ = collectstates(lsm, 4 .* data, ps, st)
        @test sum(c_hi) >= sum(c1)
        @test_throws ArgumentError predict(lsm, 5, ps, st; initialdata = [1.0])
    end

    @testset "construction validation" begin
        @test_throws ArgumentError LSM(0, 5, 1, (0.0, 1.0), Tsit5())
        @test_throws ArgumentError LSM(2, 5, 1, (1.0, 0.0), Tsit5())
        @test_throws ArgumentError LSM(2, 5, 1, (0.0, 1.0), Tsit5(); saveat = 0.1)
        @test_throws ArgumentError LSM(2, 5, 1, (0.0, 1.0), Tsit5(); callback = nothing)
        @test_throws ArgumentError LIFNeuron(; tau_m = 0.0)
        @test_throws ArgumentError LIFNeuron(; tau_syn = -1.0)
        @test_throws ArgumentError LIFNeuron(; tau_ref = -0.1)
        @test_throws ArgumentError LIFNeuron(; r_m = 0.0)
        @test_throws ArgumentError ExponentialSpikeFilter(; filter_tau = 0.0)
        @test_throws ArgumentError PoissonRateEncoder(; rate_scale = -1.0)
        @test_throws ArgumentError dale_sparse(
            MersenneTwister(0), Float64, 8, 8; radius = -1.0,
        )
        @test_throws ArgumentError dale_sparse(
            MersenneTwister(0), Float64, 8, 8; ei_weight_ratio = 0.0,
        )
        @test_throws ArgumentError dale_sparse(
            MersenneTwister(0), Float64, 8, 8; excitatory_fraction = 1.0,
        )
        @test_throws ArgumentError LSM(
            2, 5, 1, (0.0, 1.0), Tsit5(); neuron = _ClosedNeuron(),
        )
        @test_throws ArgumentError LSM(
            2, 5, 1, (0.0, 1.0), Tsit5(); encoder = _ClosedEncoder(),
        )
        @test_throws ArgumentError LSM(
            2, 5, 1, (0.0, 1.0), Tsit5(); feature_map = _ClosedFeature(),
        )
        @test_throws ArgumentError LSMCell(
            2 => 5; tspan = (0.0, 1.0), neuron = _ClosedNeuron(),
        )
        @test !ReservoirComputing.__supports_ar(_ClosedFeature())
        @test ReservoirComputing.__supports_ar(ExponentialSpikeFilter())
        @test ReservoirComputing.__supports_ar(MembraneVoltageFeature())
        @test !ReservoirComputing.__supports_ar(SpikeCountFeatures())
        lsm = LSM(2, 12, 1, (0.0, 1.0), Tsit5())
        ps, st = setup(MersenneTwister(0), lsm)
        @test size(ps.reservoir.input_matrix) == (12, 2)
        @test size(ps.reservoir.reservoir_matrix) == (12, 12)
        @test lsm.reservoir isa LSMCell
        @test lsm.reservoir.neuron isa LIFNeuron
        @test lsm.reservoir.feature_map isa ExponentialSpikeFilter
    end

    @testset "init_state sets membrane IC" begin
        n = 4
        tau_m, V0, T_end, n_win = 0.02, 0.35, 0.05, 5
        lsm = _lsm_f64(
            1, n, 1, (0.0, T_end), Tsit5();
            neuron = LIFNeuron(;
                tau_m = tau_m, r_m = 1.0, v_rest = 0.0, v_th = 10.0,
                v_reset = 0.0, tau_ref = 0.002, tau_syn = 0.005,
            ),
            feature_map = MembraneVoltageFeature(),
            init_state = (rng, d...) -> fill(V0, d...),
            reltol = 1.0e-10, abstol = 1.0e-12,
        )
        ps, st = setup(MersenneTwister(21), lsm)
        ps = _set_res!(ps; Win = zeros(n, 1), W = zeros(n, n))
        V, _ = collectstates(lsm, zeros(Float64, 1, n_win), ps, st)
        t_sample = collect(range(T_end / n_win, T_end; length = n_win))
        V_exact = V0 .* exp.(-t_sample ./ tau_m)
        @test maximum(abs, V[1, :] .- V_exact) < 1.0e-6
        @test all(v -> isapprox(v, V[1, 1]; atol = 1.0e-8), V[:, 1])
    end

    @testset "state modifiers compose" begin
        n_in, n_res, T_steps = 2, 16, 20
        kwargs = (
            neuron = LIFNeuron(; tau_m = 0.02, tau_ref = 0.002, tau_syn = 0.008),
            feature_map = MembraneVoltageFeature(),
            init_input = (rng, d...) -> scaled_rand(rng, Float64, d...; scaling = 0.5),
            init_reservoir = (rng, d...) -> dale_sparse(
                rng, Float64, d...; radius = 0.35, sparsity = 0.2,
            ),
            init_state = f64_zeros,
            reltol = 1.0e-7, abstol = 1.0e-9, dtmax = 5.0e-4, maxiters = 1_000_000,
        )
        plain = LSM(n_in, n_res, 1, (0.0, Float64(T_steps)), Tsit5(); kwargs...)
        modded = LSM(
            n_in, n_res, 1, (0.0, Float64(T_steps)), Tsit5();
            state_modifiers = (NLAT2(),), kwargs...,
        )
        ps_p, st_p = setup(MersenneTwister(15), plain)
        ps_m, st_m = setup(MersenneTwister(15), modded)
        data = 0.5 .+ 0.3 .* rand(MersenneTwister(16), Float64, n_in, T_steps)
        sp, _ = collectstates(plain, data, ps_p, st_p)
        sm, _ = collectstates(modded, data, ps_m, st_m)
        @test size(sm, 2) == size(sp, 2)
        @test sm != sp
        @test all(isfinite, sm)
    end
end
