using ReservoirComputingBenchmarks
using Test
using Random

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
        @test all(0 .<= result.delays .<= 1.001)
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

        result = narma(input, states; order = 10, metric = :nmse, reg = 1.0)

        @test haskey(result, :score)
        @test haskey(result, :metric)
        @test haskey(result, :target)
        @test result.metric === :nmse
        @test isfinite(result.score)
        @test result.score >= 0
        @test length(result.target) == T

        # Test other metrics
        result_rnmse = narma(input, states; order = 10, metric = :rnmse, reg = 1.0)
        @test result_rnmse.metric === :rnmse
        @test isfinite(result_rnmse.score)

        result_mse = narma(input, states; order = 10, metric = :mse, reg = 1.0)
        @test result_mse.metric === :mse
        @test isfinite(result_mse.score)

        # Invalid metric
        @test_throws ArgumentError narma(input, states; order = 10, metric = :invalid)
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
        @test all(0 .<= xn .<= 0.5)

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

        # Invalid NARMA metric
        @test_throws ArgumentError narma(input, states; order = 10, metric = :bad)

        # Washout out of range
        @test_throws AssertionError narma(input, states; order = 10, washout = -1)
        @test_throws AssertionError narma(input, states; order = 10, washout = 500)

        # IPC: max_degree/max_delay must be >= 1
        @test_throws AssertionError ipc(input, states; max_delay = 0, max_degree = 2)
        @test_throws AssertionError ipc(input, states; max_delay = 5, max_degree = 0)
    end
end
