begin
    using ReservoirComputing, LinearAlgebra, Random, SparseArrays
    using ReservoirComputing: AbstractSignPattern

    const res_size = 16
    const in_size = 4
    const radius = 1.0
    const rng = Random.default_rng()

    struct AlternatingTestSigns <: AbstractSignPattern end

    function ReservoirComputing.__apply_signs!(
            rng::AbstractRNG, ::AlternatingTestSigns, weights
        )
        weights[1:2:end] .*= -1
        return weights
    end

    function check_radius(matrix, target_radius; tolerance = 1.0e-5)
        if matrix isa SparseArrays.SparseMatrixCSC
            matrix = Matrix(matrix)
        end
        eigenvalues = eigvals(matrix)
        spectral_radius = maximum(abs.(eigenvalues))
        return isapprox(spectral_radius, target_radius; atol = tolerance)
    end

    function one_per_column(::Type{T}, n, sparsity) where {T}
        W = ReservoirComputing.__sparse_init(Xoshiro(1), T, n, n; sparsity, std = one(T))
        return all(==(1), count(!iszero, W; dims = 1))
    end

    function reference_rand_sparse(rng, ::Type{T}, n, sparsity) where {T}
        W = sparse_init(rng, T, n, n; sparsity = T(1) - sparsity, std = T(1))
        return scale_radius!(W, T(1))
    end

    ft = [Float16, Float32, Float64]
    reservoir_inits = [
        band_init,
        block_diagonal,
        chaotic_init,
        cycle_jumps,
        delay_line,
        delayline_backward,
        double_cycle,
        forward_connection,
        low_connectivity,
        lower_triangular,
        pseudo_svd,
        rand_hyper,
        rand_sparse,
        selfloop_cycle,
        selfloop_delayline_backward,
        selfloop_backward_cycle,
        selfloop_forwardconnection,
        simple_cycle,
        toeplitz_init,
        true_doublecycle,
        permutation_init,
        diagonal_init,
    ]
    input_inits = [
        chebyshev_mapping,
        logistic_mapping,
        minimal_init,
        minimal_init(; signs = IrrationalDigitSigns()),
        modified_lm(; factor = 4),
        scaled_rand,
        weighted_init,
        weighted_minimal,
    ]

    @testset "Reservoir Initializers" begin
        @testset "Sizes and types: $init $T" for init in reservoir_inits, T in ft
            #sizes
            @test size(init(res_size, res_size)) == (res_size, res_size)
            @test size(init(rng, res_size, res_size)) == (res_size, res_size)
            #types
            @test eltype(init(T, res_size, res_size)) == T
            @test eltype(init(rng, T, res_size, res_size)) == T
            #closure
            cl = init(rng)
            @test eltype(cl(T, res_size, res_size)) == T
        end

        @testset "Check spectral radius" begin
            sp = rand_sparse(res_size, res_size)
            @test check_radius(sp, radius)
        end

        @testset "Small reservoirs keep a nonzero entry in every column" begin
            @testset "ESN with $n neurons" for n in 1:12
                ps, _ = setup(Xoshiro(1), ESN(2, n, 2))
                W = ps.reservoir.reservoir_matrix
                @test size(W) == (n, n)
                @test all(>=(1), count(!iszero, W; dims = 1))
                @test check_radius(W, radius; tolerance = 1.0e-4)
            end

            @testset "RMNESN" begin
                ps, _ = setup(Xoshiro(1), RMNESN(2, 4, 6, 2))
                @test size(ps.reservoir.linear_reservoir.reservoir_matrix) == (4, 4)
                @test size(ps.reservoir.nonlinear_reservoir.reservoir_matrix) == (6, 6)
            end

            @testset "dale_sparse" begin
                W = dale_sparse(Xoshiro(1), Float32, 5, 5)
                @test all(>=(1), count(!iszero, W; dims = 1))
                @test all(>=(0), W[:, 1:4])
                @test all(<=(0), W[:, 5])
                @test check_radius(W, radius; tolerance = 1.0e-4)

                @testset "n = $n" for n in 2:12
                    W = dale_sparse(Xoshiro(1), Float64, n, n)
                    @test all(>=(1), count(!iszero, W; dims = 1))
                    @test check_radius(W, radius)
                end
            end

            @testset "Extreme sparsity values" begin
                dense = rand_sparse(Xoshiro(1), Float32, 8, 8; sparsity = 1.0)
                @test all(==(8), count(!iszero, dense; dims = 1))
                single = rand_sparse(Xoshiro(1), Float32, 8, 8; sparsity = 0.0)
                @test all(==(1), count(!iszero, single; dims = 1))
                @test check_radius(single, radius; tolerance = 1.0e-4)
            end

            @testset "Rounding of the sparsity bound: $T" for T in (Float32, Float64)
                sizes = vcat(1:200, 1000)
                @test all(one_per_column(T, n, T(1)) for n in sizes)
                @test all(one_per_column(T, n, T((n - 1) / n)) for n in sizes)
                @test all(one_per_column(T, n, nextfloat(T((n - 1) / n))) for n in sizes)
            end
        end

        @testset "rand_sparse output is unchanged where it already worked" begin
            @testset "$T, n = $n, sparsity = $sparsity" for T in (Float16, Float32, Float64),
                    n in (5, 10, 11, 20, 50, 100),
                    sparsity in (0.2, 0.3, 0.5, 0.8)

                @test rand_sparse(Xoshiro(1), T, n, n; sparsity) ==
                    reference_rand_sparse(Xoshiro(1), T, n, sparsity)
            end

            @testset "Default sparsity, $T, n = $n" for T in (Float16, Float32, Float64),
                    n in (10, 11, 20, 50, 100)

                @test rand_sparse(Xoshiro(1), T, n, n) ==
                    reference_rand_sparse(Xoshiro(1), T, n, T(0.1))
            end

            @testset "Random number stream" begin
                @testset "Nothing capped" begin
                    rng = Xoshiro(1)
                    reference_rng = copy(rng)
                    W = rand_sparse(rng, Float32, 20, 20)
                    @test W == reference_rand_sparse(reference_rng, Float32, 20, 0.1f0)
                    @test rng == reference_rng
                    @test rng != Xoshiro(1)
                end

                @testset "Capped, output unchanged" begin
                    rng = Xoshiro(1)
                    reference_rng = copy(rng)
                    W = rand_sparse(rng, Float64, 5, 5; sparsity = 0.2)
                    @test all(==(1), count(!iszero, W; dims = 1))
                    @test W == reference_rand_sparse(reference_rng, Float64, 5, 0.2)
                    @test rng == reference_rng
                    @test rng != Xoshiro(1)
                end

                @testset "All-zero matrix rescued" begin
                    rng = Xoshiro(1)
                    reference_rng = copy(rng)
                    rand_sparse(rng, Float32, 5, 5)
                    sparse_init(
                        reference_rng, Float32, 5, 5; sparsity = 1.0f0 - 0.1f0, std = 1.0f0
                    )
                    @test rng == reference_rng
                    @test rng != Xoshiro(1)
                end
            end

            @testset "Recorded sparsity structure" begin
                W32 = rand_sparse(Xoshiro(1), Float32, 20, 20)
                @test count(!iszero, W32) == 40
                @test all(==(2), count(!iszero, W32; dims = 1))

                W64 = rand_sparse(Xoshiro(1), Float64, 50, 50; sparsity = 0.2)
                @test count(!iszero, W64) == 500
                @test all(==(10), count(!iszero, W64; dims = 1))
            end
        end

        @testset "scale_radius! rejects matrices it cannot rescale" begin
            @test_throws ArgumentError scale_radius!(zeros(4, 4), 0.9)
            @test_throws ArgumentError scale_radius!(zeros(Float32, 4, 4), 0.9f0)
            @test_throws "spectral radius 0" scale_radius!(zeros(4, 4), 0.9)
            @test_throws "nilpotent" delay_line(Xoshiro(1), Float32, 6, 6; radius = 0.9)
            @test_throws "nilpotent" forward_connection(
                Xoshiro(1), Float32, 6, 6; radius = 0.9
            )
            @test delay_line(Xoshiro(1), Float32, 6, 6) isa Matrix{Float32}

            matrix = [0.0 2.0; 1.0 0.0]
            @test scale_radius!(matrix, 0.5) === matrix
            @test check_radius(matrix, 0.5)

            @testset "Scaling factor overflows" begin
                tiny = reshape([1.0e-310], 1, 1)
                @test_throws ArgumentError scale_radius!(tiny, 1.0)
                @test tiny == reshape([1.0e-310], 1, 1)
                @test_throws ArgumentError simple_cycle(
                    Xoshiro(1), Float64, 5, 5; cycle_weight = 1.0e-310, radius = 1.0
                )
            end

            @testset "Scaled entries overflow" begin
                @test_throws ArgumentError scale_radius!([1.0 1.0e300; 0.0 1.0], 1.0e10)
                @test_throws "overflows" scale_radius!([1.0 1.0e300; 0.0 1.0], 1.0e10)
                @test_throws "overflows" scale_radius!(Float32[1 1.0f30; 0 1], 1.0f10)
            end
        end

        @testset "Rng-less component fallbacks" begin
            @testset "$component" for component in (
                    delay_line!, add_jumps!, backward_connection!,
                    simple_cycle!, reverse_simple_cycle!, self_loop!,
                )
                @test_throws MethodError component(zeros(3, 3), "w", 1)
                @test_throws MethodError component(Xoshiro(1), zeros(3, 3), "w", 1)
            end

            matrix = zeros(Float32, 5, 5)
            delay_line!(matrix, 5.0, 2)
            @test matrix[3, 1] == matrix[4, 2] == matrix[5, 3] == 5.0f0

            matrix = zeros(Float32, 5, 5)
            self_loop!(matrix, 1.0)
            @test diag(matrix) == fill(1.0f0, 5)
        end

        @testset "add_jumps! validates public arguments" begin
            @test_throws DimensionMismatch add_jumps!(
                Xoshiro(1), zeros(Float32, 3, 4), 0.1f0, 1
            )
            @test_throws ArgumentError add_jumps!(
                Xoshiro(1), zeros(Float32, 4, 4), 0.1f0, 1; start = 0
            )
            @test_throws ArgumentError add_jumps!(
                Xoshiro(1), zeros(Float32, 4, 4), 0.1f0, 0
            )
            @test_throws ArgumentError add_jumps!(
                Xoshiro(1), zeros(Float32, 4, 4), 0.1f0, 4
            )
        end

        @testset "Minimum complexity: $init" for init in [
                delay_line,
                delayline_backward,
                cycle_jumps,
                simple_cycle,
                true_doublecycle,
                double_cycle,
                selfloop_cycle,
                selfloop_delayline_backward,
                selfloop_backward_cycle,
                selfloop_forwardconnection,
                forward_connection,
                permutation_init,
            ]
            dl = init(res_size, res_size)
            @test sort(unique(dl)) == Float32.([0.0, 0.1])
        end
    end

    @testset "Input Initializers" begin
        @testset "Sizes and types: $init $T" for init in input_inits, T in ft
            #sizes
            @test size(init(res_size, in_size)) == (res_size, in_size)
            @test size(init(rng, res_size, in_size)) == (res_size, in_size)
            #types
            @test eltype(init(T, res_size, in_size)) == T
            @test eltype(init(rng, T, res_size, in_size)) == T
            #closure
            cl = init(rng)
            @test eltype(cl(T, res_size, in_size)) == T
        end

        @testset "Minimum complexity: $init" for init in [
                minimal_init(; signs = RandomSigns()),
                minimal_init(; signs = IrrationalDigitSigns()),
            ]
            dl = init(res_size, in_size)
            @test sort(unique(dl)) == Float32.([-0.1, 0.1])
        end

        @testset "Sign patterns" begin
            @test all(==(0.1f0), minimal_init(Xoshiro(0), Float32, 2, 3))

            unchanged = Float32[1, -2, 3, -4, 5, -6]
            no_pattern = simple_cycle!(
                Xoshiro(1), zeros(Float32, 6, 6), copy(unchanged)
            )
            @test no_pattern[CartesianIndex.(mod1.(2:7, 6), 1:6)] == unchanged

            regular = minimal_init(
                Xoshiro(2), Float32, 1, 8; signs = RegularSigns(2)
            )
            @test vec(regular) == Float32[0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1]

            tuple_regular = minimal_init(
                Xoshiro(2), Float32, 1, 8; signs = RegularSigns((2, 3))
            )
            @test vec(tuple_regular) ==
                Float32[0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1]
            @test RegularSigns([2, 3]).strides == (2, 3)

            custom = minimal_init(
                Xoshiro(2), Float32, 1, 6; signs = AlternatingTestSigns()
            )
            @test vec(custom) == Float32[-0.1, 0.1, -0.1, 0.1, -0.1, 0.1]

            new_random = minimal_init(
                Xoshiro(3), Float32, 4, 3; signs = RandomSigns(0.25)
            )
            old_random = @test_deprecated minimal_init(
                Xoshiro(3), Float32, 4, 3;
                sampling_type = :bernoulli_sample!, positive_prob = 0.25
            )
            @test new_random == old_random

            new_irrational = minimal_init(
                Xoshiro(4), Float32, 4, 3;
                signs = IrrationalDigitSigns(pi; start = 2)
            )
            old_irrational = @test_deprecated minimal_init(
                Xoshiro(4), Float32, 4, 3;
                sampling_type = :irrational_sample!, irrational = pi, start = 2
            )
            @test new_irrational == old_irrational

            legacy_delay = @test_deprecated delay_line!(
                Xoshiro(5), zeros(Float32, 4, 4), 0.1f0, 1;
                sampling_type = :regular_sample!, strides = (2, 1)
            )
            new_delay = delay_line!(
                Xoshiro(5), zeros(Float32, 4, 4), 0.1f0, 1;
                signs = RegularSigns((2, 1))
            )
            @test legacy_delay == new_delay

            legacy_weighted = @test_deprecated weighted_minimal(
                Xoshiro(6), Float32, 6, 2;
                sampling_type = :bernoulli_sample!, positive_prob = 0.75
            )
            new_weighted = weighted_minimal(
                Xoshiro(6), Float32, 6, 2; signs = RandomSigns(0.75)
            )
            @test legacy_weighted == new_weighted

            legacy_nested = @test_deprecated cycle_jumps(
                Xoshiro(7), Float32, 6, 6;
                cycle_kwargs = (; sampling_type = :regular_sample!, strides = (2, 1))
            )
            new_nested = cycle_jumps(
                Xoshiro(7), Float32, 6, 6;
                cycle_kwargs = (; signs = RegularSigns((2, 1)))
            )
            @test legacy_nested == new_nested

            @test_throws ArgumentError RandomSigns(-0.1)
            @test_throws ArgumentError RandomSigns(1.1)
            @test_throws ArgumentError RegularSigns(0)
            @test_throws ArgumentError RegularSigns(())
            @test_throws ArgumentError RegularSigns((2, -1))
            @test_throws ArgumentError IrrationalDigitSigns(pi; start = 0)
            @test_throws ArgumentError minimal_init(
                2, 2; signs = RandomSigns(), sampling_type = :no_sample
            )
            @test_throws ArgumentError minimal_init(2, 2; positive_prob = 0.5)
        end

        @testset "Informed initializer" begin
            informed_res_size = 20
            informed_in_size = 5
            model_in_size = 2
            gamma = 0.6
            scaling = 0.25

            @testset "Sizes and types: $T" for T in ft
                matrix = informed_init(
                    Xoshiro(11), T, informed_res_size, informed_in_size;
                    model_in_size, gamma, scaling
                )
                @test size(matrix) == (informed_res_size, informed_in_size)
                @test eltype(matrix) == T

                init = informed_init(Xoshiro(11), T; model_in_size, gamma, scaling)
                @test init(informed_res_size, informed_in_size) == matrix
            end

            matrix = informed_init(
                Xoshiro(22), Float64, informed_res_size, informed_in_size;
                model_in_size, gamma, scaling
            )
            repeated = informed_init(
                Xoshiro(22), Float64, informed_res_size, informed_in_size;
                model_in_size, gamma, scaling
            )
            state_size = informed_in_size - model_in_size
            state_rows = findall(!iszero, vec(sum(abs, matrix[:, 1:state_size]; dims = 2)))
            model_rows = findall(
                !iszero,
                vec(sum(abs, matrix[:, (state_size + 1):end]; dims = 2))
            )

            @test matrix == repeated
            @test length(state_rows) == floor(Int, informed_res_size * gamma)
            @test length(model_rows) == floor(Int, informed_res_size * (1 - gamma))
            @test isempty(intersect(state_rows, model_rows))
            @test all(count(!iszero, row) == 1 for row in eachrow(matrix))
            @test all(abs(weight) <= scaling for weight in matrix)

            @test_throws DimensionMismatch informed_init(
                8, 3; model_in_size = 3
            )
            @test_throws DimensionMismatch informed_init(
                8, 3; model_in_size = 4
            )
        end
    end

end
