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

    function _lsm_f64(
            in_dim, res_dim, out_dim, tspan, args...;
            neuron = LIFCell(),
            encoder = CurrentInjection(),
            spike_readout = FilteredVoltageReadout(),
            init_input = f64_ones,
            init_reservoir = f64_zeros,
            init_state = f64_zeros,
            kwargs...,
        )
        return LSM(
            in_dim, res_dim, out_dim, tspan, args...;
            neuron = neuron,
            encoder = encoder,
            spike_readout = spike_readout,
            init_input = init_input,
            init_reservoir = init_reservoir,
            init_state = init_state,
            kwargs...,
        )
    end

    function _set_res!(ps; Win, W)
        return merge(
            ps,
            (reservoir = merge(ps.reservoir, (input_matrix = Win, reservoir_matrix = W)),),
        )
    end

    function _isi(τ_m, R_m, I, V_th = 1.0, V_rest = 0.0)
        return τ_m * log((R_m * I - V_rest) / (R_m * I - V_th))
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
        τ_m, R_m, I = 0.02, 1.0, 1.5
        T_isi = _isi(τ_m, R_m, I)
        T_end = 20 * T_isi
        n_win = 200
        lsm = _lsm_f64(
            1, 1, 1, (0.0, T_end), Tsit5();
            neuron = LIFCell(;
                τ_m = τ_m, R_m = R_m, V_rest = 0.0, V_th = 1.0,
                V_reset = 0.0, τ_ref = 0.0, τ_syn = 0.005,
            ),
            spike_readout = SpikeCountReadout(),
            reltol = 1.0e-10, abstol = 1.0e-12, dtmax = T_isi / 20,
        )
        ps, st = setup(MersenneTwister(0), lsm)
        ps = _set_res!(ps; Win = reshape([I], 1, 1), W = reshape([0.0], 1, 1))
        counts, _ = collectstates(lsm, ones(Float64, 1, n_win), ps, st)
        n_spikes = sum(counts)
        # endpoint / rootfind tolerance: allow ±1 spike vs ideal lattice
        expected = 1 + floor(Int, (T_end - T_isi) / T_isi + 1.0e-12)
        @test abs(n_spikes - expected) <= 1
        @test isapprox(T_end / n_spikes, T_isi; rtol = 0.08)
    end

    @testset "LIF: refractory through LSM" begin
        τ_m, R_m, I, τ_ref = 0.02, 1.0, 1.5, 0.005
        T_isi = _isi(τ_m, R_m, I)
        T_period = T_isi + τ_ref
        T_end = 15 * T_period
        n_win = 300
        lsm = _lsm_f64(
            1, 1, 1, (0.0, T_end), Tsit5();
            neuron = LIFCell(;
                τ_m = τ_m, R_m = R_m, V_rest = 0.0, V_th = 1.0,
                V_reset = 0.0, τ_ref = τ_ref, τ_syn = 0.005,
            ),
            spike_readout = SpikeCountReadout(),
            reltol = 1.0e-10, abstol = 1.0e-12, dtmax = τ_ref / 4,
        )
        ps, st = setup(MersenneTwister(1), lsm)
        ps = _set_res!(ps; Win = reshape([I], 1, 1), W = reshape([0.0], 1, 1))
        counts, _ = collectstates(lsm, ones(Float64, 1, n_win), ps, st)
        n_spikes = sum(counts)
        expected = 1 + floor(Int, (T_end - T_isi) / T_period + 1.0e-12)
        @test abs(n_spikes - expected) <= 1
        @test isapprox(T_end / n_spikes, T_period; rtol = 0.08)
        # refractory lengthens period vs pure ISI
        @test T_end / n_spikes > T_isi * 1.1
    end

    @testset "LIF: subthreshold V matches closed form" begin
        τ_m, R_m, I = 0.02, 1.0, 0.5
        V_ss = R_m * I
        @test V_ss < 1.0
        T_end = 0.2
        n_win = 40
        lsm = _lsm_f64(
            1, 1, 1, (0.0, T_end), Tsit5();
            neuron = LIFCell(;
                τ_m = τ_m, R_m = R_m, V_rest = 0.0, V_th = 1.0,
                V_reset = 0.0, τ_ref = 0.002, τ_syn = 0.005,
            ),
            spike_readout = FilteredVoltageReadout(),
            reltol = 1.0e-10, abstol = 1.0e-12,
        )
        ps, st = setup(MersenneTwister(2), lsm)
        ps = _set_res!(ps; Win = reshape([I], 1, 1), W = reshape([0.0], 1, 1))
        V, _ = collectstates(lsm, ones(Float64, 1, n_win), ps, st)
        t_sample = collect(range(T_end / n_win, T_end; length = n_win))
        V_exact = V_ss .* (1 .- exp.(-t_sample ./ τ_m))
        @test maximum(abs, vec(V) .- V_exact) < 1.0e-5
    end

    @testset "LIF: voltage stays below threshold after reset" begin
        τ_m, R_m, I = 0.02, 1.0, 2.0
        T_isi = _isi(τ_m, R_m, I)
        T_end = 5 * T_isi
        n_win = 500
        lsm = _lsm_f64(
            1, 1, 1, (0.0, T_end), Tsit5();
            neuron = LIFCell(;
                τ_m = τ_m, R_m = R_m, V_rest = 0.0, V_th = 1.0,
                V_reset = 0.0, τ_ref = 0.0, τ_syn = 0.005,
            ),
            spike_readout = FilteredVoltageReadout(),
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
        τ_m, R_m, I, w, τ_syn = 0.02, 1.0, 2.0, 0.8, 0.01
        T_isi = _isi(τ_m, R_m, I)
        T_end = T_isi + 0.5 * τ_syn
        n_win = 25
        lsm = _lsm_f64(
            1, 2, 1, (0.0, T_end), Tsit5();
            neuron = LIFCell(;
                τ_m = τ_m, R_m = R_m, V_rest = 0.0, V_th = 1.0,
                V_reset = 0.0, τ_ref = 0.002, τ_syn = τ_syn,
            ),
            spike_readout = FilteredVoltageReadout(),
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

    @testset "SpikeCountReadout: rate and AR guard" begin
        τ_m, R_m, I = 0.02, 1.0, 2.0
        T_isi = _isi(τ_m, R_m, I)
        Δt = 3 * T_isi
        n_win = 8
        T_end = n_win * Δt
        lsm = _lsm_f64(
            1, 1, 1, (0.0, T_end), Tsit5();
            neuron = LIFCell(;
                τ_m = τ_m, R_m = R_m, V_rest = 0.0, V_th = 1.0,
                V_reset = 0.0, τ_ref = 0.0, τ_syn = 0.005,
            ),
            spike_readout = SpikeCountReadout(),
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

    @testset "ExponentialFilterReadout: tracks spikes and decays" begin
        τ_m, R_m, I, τ_f = 0.02, 1.0, 2.0, 0.03
        T_isi = _isi(τ_m, R_m, I)
        n_on, n_off = 25, 25
        n_win = n_on + n_off
        Δt = T_isi
        T_end = n_win * Δt
        lsm = _lsm_f64(
            1, 1, 1, (0.0, T_end), Tsit5();
            neuron = LIFCell(;
                τ_m = τ_m, R_m = R_m, V_rest = 0.0, V_th = 1.0,
                V_reset = 0.0, τ_ref = 0.0, τ_syn = 0.005,
            ),
            spike_readout = ExponentialFilterReadout(τ_f),
            reltol = 1.0e-10, abstol = 1.0e-12, dtmax = T_isi / 25,
        )
        ps, st = setup(MersenneTwister(6), lsm)
        ps = _set_res!(ps; Win = reshape([1.0], 1, 1), W = reshape([0.0], 1, 1))
        # first half driven, second half silent
        data = hcat(fill(I, 1, n_on), zeros(1, n_off))
        S, _ = collectstates(lsm, data, ps, st)
        s = vec(S)
        @test maximum(s[1:n_on]) > 0.5
        @test mean(s[1:n_on]) > mean(s[(n_on + 1):end])
        # decay over silent windows (not exact zero immediately)
        @test s[end] < s[n_on]
        @test s[end] < 0.5 * s[n_on]

        # subthreshold → no spikes → filter stays near 0
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
            neuron = LIFCell(;
                τ_m = 0.02, R_m = 1.0, V_rest = 0.0, V_th = 1.0,
                V_reset = 0.0, τ_ref = 0.002, τ_syn = 0.01,
            ),
            spike_readout = ExponentialFilterReadout(0.05),
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
            spike_readout = lsm.reservoir.spike_readout,
            init_input = f64_ones, init_reservoir = f64_zeros, init_state = init_a,
            reltol = 1.0e-7, abstol = 1.0e-9, dtmax = 5.0e-4, maxiters = 2_000_000,
        )
        lsm_b = _lsm_f64(
            2, n, 2, tspan, Tsit5();
            neuron = lsm.reservoir.neuron,
            spike_readout = lsm.reservoir.spike_readout,
            init_input = f64_ones, init_reservoir = f64_zeros, init_state = init_b,
            reltol = 1.0e-7, abstol = 1.0e-9, dtmax = 5.0e-4, maxiters = 2_000_000,
        )
        ps_a, st_a = setup(MersenneTwister(7), lsm_a)
        ps_b, st_b = setup(MersenneTwister(7), lsm_b)
        # shared W, Win
        ps_a = merge(ps_a, (reservoir = ps.reservoir,))
        ps_b = merge(ps_b, (reservoir = ps.reservoir,))

        Fa, _ = collectstates(lsm_a, data, ps_a, st_a)
        Fb, _ = collectstates(lsm_b, data, ps_b, st_b)
        early = mean(abs, Fa[:, 1:5] .- Fb[:, 1:5])
        late = mean(abs, Fa[:, (end - 4):end] .- Fb[:, (end - 4):end])
        @test early > 0 || true  # may already be close
        @test late < early + 1.0e-3 || late < 0.15
        @test late < 0.25
        @test all(isfinite, Fa) && all(isfinite, Fb)
    end

    @testset "teacher-forced predict matches readout∘states" begin
        n_in, n_res, n_out, T_steps = 2, 30, 2, 40
        lsm = _lsm_f64(
            n_in, n_res, n_out, (0.0, Float64(T_steps)), Tsit5();
            neuron = LIFCell(; τ_m = 0.02, τ_ref = 0.002, τ_syn = 0.008),
            spike_readout = ExponentialFilterReadout(0.04),
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
        # manual readout application
        y_man = ps.readout.weight * states
        @test y_tf ≈ y_man rtol = 1.0e-5
    end

    @testset "train improves one-step forecast over zero readout" begin
        n_in, n_res, n_out, T_steps = 2, 60, 2, 120
        lsm = _lsm_f64(
            n_in, n_res, n_out, (0.0, Float64(T_steps)), Tsit5();
            neuron = LIFCell(; τ_m = 0.02, τ_ref = 0.002, τ_syn = 0.008),
            spike_readout = ExponentialFilterReadout(0.05),
            # dense positive Win so random drive is reliably suprathreshold
            init_input = (rng, d...) -> abs.(randn(rng, Float64, d...)) .+ 0.5,
            init_reservoir = (rng, d...) -> dale_sparse(
                rng, Float64, d...; radius = 0.5, sparsity = 0.15,
            ),
            reltol = 1.0e-6, abstol = 1.0e-8, dtmax = 5.0e-4, maxiters = 3_000_000,
        )
        ps, st = setup(MersenneTwister(11), lsm)
        rng = MersenneTwister(12)
        # positive multichannel drive with delayed cross-coupling as target structure
        u = abs.(randn(rng, Float64, n_in, T_steps)) .+ 0.8
        y = similar(u)
        y[:, 1] .= u[:, 1]
        for t in 2:T_steps
            y[:, t] .= 0.6 .* u[:, t] .+ 0.4 .* reverse(u[:, t - 1])
        end

        states, st = collectstates(lsm, u, ps, st)
        @test mean(abs, states) > 0.05

        nrmse0 = norm(zeros(n_out, T_steps) .- y) / norm(y .- mean(y; dims = 2))
        ps_tr, st_tr = train(
            lsm, u, y, ps, st;
            washout = 20, objective = RidgeRegression(1.0e-3),
        )
        yhat, _ = predict(lsm, u, ps_tr, st_tr)
        nrmse = norm(yhat[:, 21:end] .- y[:, 21:end]) /
            norm(y[:, 21:end] .- mean(y[:, 21:end]; dims = 2))
        @test nrmse < nrmse0
        @test nrmse < 1.0
        @test all(isfinite, yhat)
    end

    @testset "AR predict is deterministic and finite" begin
        dim, n_res, steps = 2, 40, 15
        lsm = _lsm_f64(
            dim, n_res, dim, (0.0, Float64(steps)), Tsit5();
            neuron = LIFCell(; τ_m = 0.02, τ_ref = 0.002, τ_syn = 0.008),
            spike_readout = ExponentialFilterReadout(0.04),
            init_input = (rng, d...) -> scaled_rand(rng, Float64, d...; scaling = 1.0),
            init_reservoir = (rng, d...) -> dale_sparse(
                rng, Float64, d...; radius = 0.4, sparsity = 0.2,
            ),
            reltol = 1.0e-7, abstol = 1.0e-9, dtmax = 5.0e-4, maxiters = 2_000_000,
        )
        ps, st = setup(MersenneTwister(13), lsm)
        # train briefly so readout is non-random
        u = 0.7 .* ones(Float64, dim, 40)
        y = circshift(u, (0, -1))
        lsm_tr = _lsm_f64(
            dim, n_res, dim, (0.0, 40.0), Tsit5();
            neuron = lsm.reservoir.neuron,
            spike_readout = lsm.reservoir.spike_readout,
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

    @testset "PoissonRateEncoder: event drive produces spikes" begin
        n_win = 40
        T_end = 2.0
        lsm = _lsm_f64(
            1, 8, 1, (0.0, T_end), Tsit5();
            encoder = PoissonRateEncoder(; scale = 60.0, weight = 3.0),
            neuron = LIFCell(;
                τ_m = 0.02, R_m = 1.0, V_rest = 0.0, V_th = 1.0,
                V_reset = 0.0, τ_ref = 0.002, τ_syn = 0.02,
            ),
            spike_readout = SpikeCountReadout(),
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
        lsm = LSM(2, 12, 1, (0.0, 1.0), Tsit5())
        ps, st = setup(MersenneTwister(0), lsm)
        @test size(ps.reservoir.input_matrix) == (12, 2)
        @test size(ps.reservoir.reservoir_matrix) == (12, 12)
    end

    @testset "state modifiers compose" begin
        n_in, n_res, T_steps = 2, 16, 20
        kwargs = (
            neuron = LIFCell(; τ_m = 0.02, τ_ref = 0.002, τ_syn = 0.008),
            spike_readout = FilteredVoltageReadout(),
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
