using ReservoirComputing, LinearAlgebra, Random, SparseArrays

const res_size = 16
const in_size = 4
const radius = 1.0
const rng = Random.default_rng()

function check_radius(matrix, target_radius; tolerance = 1.0e-5)
    if matrix isa SparseArrays.SparseMatrixCSC
        matrix = Matrix(matrix)
    end
    eigenvalues = eigvals(matrix)
    spectral_radius = maximum(abs.(eigenvalues))
    return isapprox(spectral_radius, target_radius; atol = tolerance)
end

ft = [Float16, Float32, Float64]
reservoir_inits = [
    block_diagonal,
    chaotic_init,
    cycle_jumps,
    delay_line,
    delayline_backward,
    double_cycle,
    forward_connection,
    low_connectivity,
    pseudo_svd,
    rand_hyper,
    rand_sparse,
    selfloop_cycle,
    selfloop_delayline_backward,
    selfloop_backward_cycle,
    selfloop_forwardconnection,
    simple_cycle,
    true_doublecycle,
    permutation_init,
    diagonal_init,
]
input_inits = [
    chebyshev_mapping,
    logistic_mapping,
    minimal_init,
    minimal_init(; sampling_type = :irrational_sample!),
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

# TODO: @MartinuzziFrancesco Missing tests for informed_init
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
            minimal_init,
            minimal_init(; sampling_type = :irrational_sample!),
        ]
        dl = init(res_size, in_size)
        @test sort(unique(dl)) == Float32.([-0.1, 0.1])
    end
end

@testset "Lower Triangular Topology Tests" begin
    
    # 1. Test Dense Return with Exact Sparsity
    target_sparsity_1 = 0.8
    reservoir_dense = lower_triangular(rng, Float16, res_size, res_size; sparsity=target_sparsity_1, return_sparse=false)
    
    @test reservoir_dense isa Matrix{Float16}
    @test size(reservoir_dense) == (res_size, res_size)
    @test istril(reservoir_dense) == true 
    
    expected_non_zeros_1 = round(Int, res_size * res_size * (1.0 - target_sparsity_1))
    @test count(!iszero, reservoir_dense) == expected_non_zeros_1
    
    # 2. Test Sparse Return with a different Exact Sparsity
    target_sparsity_2 = 0.85
    reservoir_sparse = lower_triangular(rng, Float32, res_size, res_size; sparsity=target_sparsity_2, return_sparse=true)
    
    @test reservoir_sparse isa SparseMatrixCSC{Float32, Int}
    @test size(reservoir_sparse) == (res_size, res_size)
    @test istril(reservoir_sparse) == true
    
    expected_non_zeros_2 = round(Int, res_size * res_size * (1.0 - target_sparsity_2))
    @test count(!iszero, reservoir_sparse) == expected_non_zeros_2
    
    # 3. Test Scale Radius
    reservoir_scaled = lower_triangular(rng, Float32, res_size, res_size; radius=2.5)
    spectral_radius = maximum(abs.(eigvals(reservoir_scaled)))
    @test spectral_radius ≈ 2.5 atol=1e-5
end