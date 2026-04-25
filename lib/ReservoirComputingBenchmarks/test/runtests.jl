using ReservoirComputingBenchmarks
using Test
using Random
using Statistics
using ReservoirComputing

@testset "ReservoirComputingBenchmarks" begin
    rng = Random.MersenneTwister(42)

    @testset "Memory Capacity" begin
        T = 2000
        n_features = 50
        input = rand(rng, T) .* 2 .- 1  # Uniform[-1, 1]

        # Build a simple linear reservoir (identity + shift) that should
        # have high memory capacity
        states = zeros(n_features, T)
        for t in 2:T
            states[1, t] = input[t - 1]
            for i in 2:n_features
                states[i, t] = states[i - 1, t - 1]
            end
        end

        result = memory_capacity(input, states; max_delay = 20, reg = 1.0e-6)

        @test haskey(result, :total)
        @test haskey(result, :delays)
        @test length(result.delays) == 20
        @test all(d -> 0 <= d <= 1.001, result.delays)
        @test result.total >= 0
        # A shift-register reservoir should have near-perfect MC for
        # delays up to n_features
        @test result.delays[1] > 0.9
        @test result.delays[5] > 0.8
        @test result.total > 10
    end

    @testset "NARMA Target Generation" begin
        T = 1000
        input = rand(rng, T) .* 0.5  # Uniform[0, 0.5]

        # NARMA-2
        y2 = generate_narma(input; order = 2, normalize = false)
        @test length(y2) == T
        @test y2[1] == 0.0
        @test y2[2] == 0.0
        @test y2[3] != 0.0  # First non-trivial value

        # NARMA-10
        y10 = generate_narma(input; order = 10, normalize = false)
        @test length(y10) == T
        @test all(y10[1:10] .== 0.0)
        @test y10[11] != 0.0

        # NARMA-30
        y30 = generate_narma(input; order = 30, normalize = false)
        @test length(y30) == T
        @test all(y30[1:30] .== 0.0)
        @test y30[31] != 0.0

        # Target should not diverge for standard inputs
        @test all(isfinite.(y10))
        @test all(isfinite.(y30))

        # Custom coefficients for non-standard order
        @test_throws AssertionError generate_narma(input; order = 5)
        y5 = generate_narma(input; order = 5, alpha = 0.3, beta = 0.05, gamma = 1.5, delta = 0.1, normalize = false)
        @test length(y5) == T

        # Order must be >= 2
        @test_throws AssertionError generate_narma(input; order = 1)
    end

    @testset "NARMA Evaluation" begin
        T = 1000
        n_features = 100
        input = rand(rng, T)

        # Use random states (won't be great, but should run)
        states = randn(rng, n_features, T)

        result = narma(input, states; order = 10, metric = nmse, reg = 1.0)

        @test haskey(result, :score)
        @test haskey(result, :target)
        @test isfinite(result.score)
        @test result.score >= 0
        @test length(result.target) == T

        # Test other metrics
        result_rnmse = narma(input, states; order = 10, metric = rnmse, reg = 1.0)
        @test isfinite(result_rnmse.score)

        result_mse = narma(input, states; order = 10, metric = mse, reg = 1.0)
        @test isfinite(result_mse.score)

        # Custom metric
        custom_metric(y_true, y_pred) = mean(abs.(y_true .- y_pred))
        result_custom = narma(input, states; order = 10, metric = custom_metric, reg = 1.0)
        @test isfinite(result_custom.score)
    end

    @testset "IPC" begin
        T = 1000
        n_features = 30
        input = rand(rng, T) .* 2 .- 1  # Uniform[-1, 1]

        # Simple linear reservoir
        states = zeros(n_features, T)
        for t in 2:T
            states[1, t] = input[t - 1]
            for i in 2:n_features
                states[i, t] = 0.95 * states[i, t - 1] + 0.05 * randn(rng)
            end
        end

        result = ipc(input, states; max_delay = 5, max_degree = 2, reg = 1.0e-4)

        @test haskey(result, :total)
        @test haskey(result, :linear)
        @test haskey(result, :nonlinear)
        @test haskey(result, :by_degree)
        @test haskey(result, :by_delay)
        @test haskey(result, :basis_capacities)
        @test haskey(result, :theoretical_max)

        @test result.total >= 0
        @test result.linear >= 0
        @test result.nonlinear >= 0
        @test result.theoretical_max == n_features
        @test abs(result.total - result.linear - result.nonlinear) < 1.0e-10

        # Degree-1 capacity should be present
        @test haskey(result.by_degree, 1)

        # Total should not exceed theoretical maximum (with some tolerance)
        @test result.total <= n_features + 1.0

        # Without cross terms
        result_no_cross = ipc(input, states; max_delay = 5, max_degree = 2, cross_terms = false, reg = 1.0e-4)
        @test length(result_no_cross.basis_capacities) < length(result.basis_capacities)
    end

    @testset "Nonlinear Transformation" begin
        T = 1000
        n_features = 80
        input = rand(rng, T) .* 2 .- 1

        # Reservoir whose first feature is u(t) and rest are nonlinear maps of it
        states = zeros(n_features, T)
        for t in 1:T
            states[1, t] = input[t]
            states[2, t] = input[t]^2
            states[3, t] = sin(π * input[t])
            for i in 4:n_features
                states[i, t] = tanh(0.7 * input[t] + 0.1 * (i - 3))
            end
        end

        # sin(π u) target — a feature exactly matches it, so the readout
        # should fit it well (small NMSE up to ridge regularisation bias).
        r = nonlinear_transformation(input, states; f = x -> sin(π * x), reg = 1.0e-6)
        @test haskey(r, :score)
        @test haskey(r, :target)
        @test length(r.target) == T
        @test isfinite(r.score)
        @test r.score < 1.0e-3

        # x^2 target — exactly available
        r2 = nonlinear_transformation(input, states; f = x -> x^2, reg = 1.0e-6)
        @test r2.score < 1.0e-3

        # custom metric
        r3 = nonlinear_transformation(input, states; f = sin, metric = mse, reg = 1.0e-4)
        @test isfinite(r3.score)

        # Validation
        @test_throws AssertionError nonlinear_transformation(input, randn(rng, n_features, T - 1))
        @test_throws AssertionError nonlinear_transformation(input, states; washout = T)
        @test_throws AssertionError nonlinear_transformation(input, states; washout = -1)
    end

    @testset "Nonlinear Memory" begin
        T = 2000
        n_features = 50
        input = rand(rng, T) .* 2 .- 1

        # Build a reservoir whose features are u(t-k) (linear shift register)
        states = zeros(n_features, T)
        for t in 2:T
            states[1, t] = input[t - 1]
            for i in 2:n_features
                states[i, t] = states[i - 1, t - 1]
            end
        end

        # Linear memory baseline (should reduce to memory_capacity for f=identity)
        r_lin = nonlinear_memory(input, states; f = identity, max_delay = 20, reg = 1.0e-6)
        @test r_lin.delays[1] > 0.9
        @test r_lin.total > 10

        # Squared nonlinear memory: a linear reservoir cannot recover x^2 perfectly
        r_sq = nonlinear_memory(input, states; f = x -> x^2, max_delay = 20, reg = 1.0e-6)
        @test length(r_sq.delays) == 20
        @test all(d -> 0 <= d <= 1.001, r_sq.delays)
        @test r_sq.total >= 0

        # Validation
        @test_throws AssertionError nonlinear_memory(input, states; max_delay = 0)
        @test_throws AssertionError nonlinear_memory(input, states; max_delay = T)
        @test_throws AssertionError nonlinear_memory(rand(50), randn(rng, 10, 100))
    end

    @testset "Sin Approximation" begin
        T = 800
        n_features = 30
        input = rand(rng, T) .* 2 .- 1

        # Reservoir already contains sin(π u(t)) — perfect fit possible
        states = zeros(n_features, T)
        for t in 1:T
            states[1, t] = sin(π * input[t])
            for i in 2:n_features
                states[i, t] = tanh(0.5 * input[t] + 0.05 * (i - 2))
            end
        end

        r = sin_approximation(input, states; freq = π, reg = 1.0e-6)
        @test haskey(r, :score)
        @test r.score < 1.0e-3

        # freq = 1 — non-trivial map; just check finite
        r2 = sin_approximation(input, states; freq = 1.0)
        @test isfinite(r2.score)
    end

    @testset "Kernel / Generalization Rank (array form)" begin
        # n_features × n full-rank random matrix
        M_full = randn(rng, 30, 50)
        @test kernel_rank(M_full) == 30
        @test generalization_rank(M_full) == 30

        # Rank-1 matrix: numerical rank == 1
        v = randn(rng, 30)
        M_rank1 = v * randn(rng, 50)'
        @test kernel_rank(M_rank1) == 1
        @test generalization_rank(M_rank1) == 1

        # All-zero matrix → rank 0
        @test kernel_rank(zeros(30, 50)) == 0
        @test generalization_rank(zeros(30, 50)) == 0

        # threshold extremes
        M_mixed = randn(rng, 5, 5)
        @test kernel_rank(M_mixed; threshold = 1.0e-12) >= kernel_rank(M_mixed; threshold = 0.5)

        # Validation
        @test_throws AssertionError kernel_rank(M_full; threshold = 0.0)
        @test_throws AssertionError kernel_rank(M_full; threshold = -0.1)
    end

    @testset "Legendre Polynomials" begin
        leg = ReservoirComputingBenchmarks._legendre

        # P_0(x) = 1
        @test leg(0, 0.5) ≈ 1.0
        @test leg(0, -0.3) ≈ 1.0

        # P_1(x) = x
        @test leg(1, 0.5) ≈ 0.5
        @test leg(1, -0.3) ≈ -0.3

        # P_2(x) = (3x² - 1) / 2
        @test leg(2, 0.5) ≈ (3 * 0.25 - 1) / 2
        @test leg(2, 0.0) ≈ -0.5

        # P_3(x) = (5x³ - 3x) / 2
        @test leg(3, 0.5) ≈ (5 * 0.125 - 1.5) / 2

        # Negative degree should error
        @test_throws AssertionError leg(-1, 0.5)

        # Orthonormality: ∫₋₁¹ P̃_n(x) P̃_m(x) dx ≈ δ_{nm}
        nleg = ReservoirComputingBenchmarks._normalized_legendre
        N_quad = 10000
        x_quad = range(-1, 1; length = N_quad)
        dx = 2.0 / (N_quad - 1)

        for n in 0:3
            vals_n = nleg.(n, x_quad)
            integral = sum(vals_n .^ 2) * dx
            @test isapprox(integral, 1.0; atol = 0.01)
        end
    end

    @testset "Utilities" begin
        # _normalize
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        xn = ReservoirComputingBenchmarks._normalize(x, 0.0, 0.5)
        @test xn[1] ≈ 0.0
        @test xn[end] ≈ 0.5
        @test all(v -> 0 <= v <= 0.5, xn)

        # Constant input
        xc = ReservoirComputingBenchmarks._normalize([3.0, 3.0, 3.0], 0.0, 0.5)
        @test all(xc .== 0.0)

        # _squared_correlation
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        @test ReservoirComputingBenchmarks._squared_correlation(a, a) ≈ 1.0
        @test ReservoirComputingBenchmarks._squared_correlation(a, -a) ≈ 1.0

        # _nmse: perfect prediction → 0
        @test ReservoirComputingBenchmarks._nmse(a, a) ≈ 0.0

        # _train_test_split: invalid ratios
        @test_throws AssertionError ReservoirComputingBenchmarks._train_test_split(100, 0.0)
        @test_throws AssertionError ReservoirComputingBenchmarks._train_test_split(100, 1.0)
        @test_throws AssertionError ReservoirComputingBenchmarks._train_test_split(100, -0.5)
        @test_throws AssertionError ReservoirComputingBenchmarks._train_test_split(100, 1.5)

        # _ridge_regression: negative reg
        @test_throws AssertionError ReservoirComputingBenchmarks._ridge_regression(
            randn(50, 5), randn(50); reg = -1.0
        )
    end

    @testset "Validation" begin
        input = rand(rng, 500) .* 2 .- 1
        states = randn(rng, 10, 500)

        # max_delay must be >= 1
        @test_throws AssertionError memory_capacity(input, states; max_delay = 0)

        # max_delay must be < T
        @test_throws AssertionError memory_capacity(input, states; max_delay = 500)

        # Mismatched dimensions
        @test_throws AssertionError memory_capacity(rand(100), randn(10, 200))

        # NARMA order < 2
        @test_throws AssertionError generate_narma(rand(100); order = 1)

        # Missing NARMA coefficients for non-standard order
        @test_throws AssertionError generate_narma(rand(100); order = 7)

        # Input shorter than order
        @test_throws AssertionError generate_narma(rand(5); order = 10)

        # Washout out of range
        @test_throws AssertionError narma(input, states; order = 10, washout = -1)
        @test_throws AssertionError narma(input, states; order = 10, washout = 500)

        # IPC: max_degree/max_delay must be >= 1
        @test_throws AssertionError ipc(input, states; max_delay = 0, max_degree = 2)
        @test_throws AssertionError ipc(input, states; max_delay = 5, max_degree = 0)
    end

    @testset "ReservoirComputing model dispatch" begin
        rng_model = Random.MersenneTwister(7)
        T = 600
        in_dims, res_dims, out_dims = 1, 40, 1
        model = ESN(in_dims, res_dims, out_dims, tanh)
        ps, st = setup(rng_model, model)

        @testset "memory_capacity(model, ps, st)" begin
            rng_input = Random.MersenneTwister(11)
            r = memory_capacity(
                model, ps, st;
                T = T, rng = rng_input, max_delay = 10, reg = 1.0e-4,
            )
            @test haskey(r, :total)
            @test haskey(r, :delays)
            @test length(r.delays) == 10
            @test isfinite(r.total)
            @test r.total >= 0
            @test all(d -> 0 <= d <= 1.001, r.delays)

            # Reproducibility: same rng + same model → same result
            r2 = memory_capacity(
                model, ps, st;
                T = T, rng = Random.MersenneTwister(11), max_delay = 10, reg = 1.0e-4,
            )
            @test r2.total ≈ r.total
            @test r2.delays ≈ r.delays
        end

        @testset "narma(model, ps, st)" begin
            rng_input = Random.MersenneTwister(13)
            r = narma(
                model, ps, st;
                T = T, rng = rng_input, order = 10, metric = nmse, reg = 1.0e-4,
            )
            @test haskey(r, :score)
            @test haskey(r, :target)
            @test isfinite(r.score)
            @test r.score >= 0
            @test length(r.target) == T

            # User-supplied input bypasses rng
            user_input = rand(Random.MersenneTwister(17), T) .* 2 .- 1
            r_user = narma(
                model, ps, st;
                input = user_input, order = 10, reg = 1.0e-4,
            )
            @test isfinite(r_user.score)
            @test length(r_user.target) == T
        end

        @testset "ipc(model, ps, st)" begin
            rng_input = Random.MersenneTwister(19)
            r = ipc(
                model, ps, st;
                T = T, rng = rng_input,
                max_delay = 4, max_degree = 2, reg = 1.0e-4,
            )
            @test haskey(r, :total)
            @test haskey(r, :linear)
            @test haskey(r, :nonlinear)
            @test haskey(r, :basis_capacities)
            @test r.theoretical_max == res_dims
            @test r.total >= 0
            @test abs(r.total - r.linear - r.nonlinear) < 1.0e-10
        end

        @testset "nonlinear_transformation(model, ps, st)" begin
            rng_input = Random.MersenneTwister(23)
            r = nonlinear_transformation(
                model, ps, st;
                T = T, rng = rng_input, f = x -> x^2, reg = 1.0e-4,
            )
            @test haskey(r, :score)
            @test haskey(r, :target)
            @test isfinite(r.score)
            @test length(r.target) == T
        end

        @testset "nonlinear_memory(model, ps, st)" begin
            rng_input = Random.MersenneTwister(29)
            r = nonlinear_memory(
                model, ps, st;
                T = T, rng = rng_input, f = x -> x^2, max_delay = 8, reg = 1.0e-4,
            )
            @test haskey(r, :total)
            @test haskey(r, :delays)
            @test length(r.delays) == 8
            @test all(d -> 0 <= d <= 1.001, r.delays)
            @test isfinite(r.total)
        end

        @testset "sin_approximation(model, ps, st)" begin
            rng_input = Random.MersenneTwister(31)
            r = sin_approximation(
                model, ps, st;
                T = T, rng = rng_input, freq = π, reg = 1.0e-4,
            )
            @test haskey(r, :score)
            @test haskey(r, :target)
            @test isfinite(r.score)
            @test length(r.target) == T
        end

        @testset "kernel_rank(model, ps, st)" begin
            rng_input = Random.MersenneTwister(37)
            kr = kernel_rank(
                model, ps, st;
                n_streams = 80, stream_length = 60,
                rng = rng_input, threshold = 0.01,
            )
            @test kr isa Integer
            @test 0 <= kr <= res_dims

            # Reproducibility
            kr2 = kernel_rank(
                model, ps, st;
                n_streams = 80, stream_length = 60,
                rng = Random.MersenneTwister(37), threshold = 0.01,
            )
            @test kr2 == kr
        end

        @testset "generalization_rank(model, ps, st)" begin
            rng_input = Random.MersenneTwister(41)
            gr = generalization_rank(
                model, ps, st;
                n_streams = 80, stream_length = 60,
                perturbation = 1.0e-4,
                rng = rng_input, threshold = 0.01,
            )
            @test gr isa Integer
            @test 0 <= gr <= res_dims

            # Deterministic invariant: with perturbation == 0, all streams are
            # the same `base_input`, so all final states are identical and the
            # final-state matrix has numerical rank exactly 1.
            gr_zero = generalization_rank(
                model, ps, st;
                n_streams = 50, stream_length = 60,
                perturbation = 0.0,
                rng = Random.MersenneTwister(43), threshold = 0.01,
            )
            @test gr_zero == 1
        end

        @testset "input length validation" begin
            @test_throws AssertionError memory_capacity(
                model, ps, st; T = 1, max_delay = 1,
            )
            short = [0.1, 0.2]
            @test_throws AssertionError memory_capacity(
                model, ps, st; input = [0.1], max_delay = 1,
            )
            # input keyword length controls dispatch length, not T
            @test_throws AssertionError narma(model, ps, st; input = [0.1])
        end

        @testset "scalar input (in_dims == 1) check" begin
            # in_dims = 3 → friendly ArgumentError, not an opaque shape error
            multi_model = ESN(3, 20, 1, tanh)
            multi_ps, multi_st = setup(Random.MersenneTwister(5), multi_model)
            @test_throws ArgumentError memory_capacity(
                multi_model, multi_ps, multi_st; T = 200, max_delay = 4,
            )
            @test_throws ArgumentError narma(
                multi_model, multi_ps, multi_st; T = 200,
            )
            @test_throws ArgumentError ipc(
                multi_model, multi_ps, multi_st; T = 200, max_delay = 2, max_degree = 2,
            )
            @test_throws ArgumentError nonlinear_transformation(
                multi_model, multi_ps, multi_st; T = 200,
            )
            @test_throws ArgumentError nonlinear_memory(
                multi_model, multi_ps, multi_st; T = 200, max_delay = 4,
            )
            @test_throws ArgumentError sin_approximation(
                multi_model, multi_ps, multi_st; T = 200,
            )
            @test_throws ArgumentError kernel_rank(
                multi_model, multi_ps, multi_st; n_streams = 10, stream_length = 30,
            )
            @test_throws ArgumentError generalization_rank(
                multi_model, multi_ps, multi_st;
                n_streams = 10, stream_length = 30,
            )
        end
    end

    @testset "Cross-model dispatch ($(nameof(M)))" for M in (ES2N, EuSN)
        rng_model = Random.MersenneTwister(101)
        T = 400
        in_dims, res_dims, out_dims = 1, 25, 1
        model = M(in_dims, res_dims, out_dims, tanh)
        ps, st = setup(rng_model, model)

        rng_input = Random.MersenneTwister(103)
        r_mc = memory_capacity(
            model, ps, st;
            T = T, rng = rng_input, max_delay = 5, reg = 1.0e-4,
        )
        @test isfinite(r_mc.total)
        @test length(r_mc.delays) == 5

        r_narma = narma(
            model, ps, st;
            T = T, rng = Random.MersenneTwister(105), order = 10, reg = 1.0e-4,
        )
        @test isfinite(r_narma.score)

        r_ipc = ipc(
            model, ps, st;
            T = T, rng = Random.MersenneTwister(107),
            max_delay = 3, max_degree = 2, reg = 1.0e-4,
        )
        @test r_ipc.theoretical_max == res_dims
        @test isfinite(r_ipc.total)

        r_nlt = nonlinear_transformation(
            model, ps, st;
            T = T, rng = Random.MersenneTwister(109),
            f = x -> x^2, reg = 1.0e-4,
        )
        @test isfinite(r_nlt.score)

        r_nlm = nonlinear_memory(
            model, ps, st;
            T = T, rng = Random.MersenneTwister(111),
            f = x -> x^2, max_delay = 4, reg = 1.0e-4,
        )
        @test length(r_nlm.delays) == 4

        kr = kernel_rank(
            model, ps, st;
            n_streams = 60, stream_length = 50,
            rng = Random.MersenneTwister(113),
        )
        @test 0 <= kr <= res_dims

        gr_zero = generalization_rank(
            model, ps, st;
            n_streams = 30, stream_length = 50,
            perturbation = 0.0,
            rng = Random.MersenneTwister(115),
        )
        @test gr_zero == 1
    end
end
