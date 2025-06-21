using ReservoirComputing, LinearAlgebra, Random, SparseArrays

const res_size = 16
const in_size = 4
const radius = 1.0
const sparsity = 0.1
const weight = 0.2
const jump_size = 3
const rng = Random.default_rng()

function check_radius(matrix, target_radius; tolerance = 1e-5)
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
    delay_line_backward,
    double_cycle,
    forward_connection,
    low_connectivity,
    pseudo_svd,
    rand_sparse,
    selfloop_cycle,
    selfloop_delayline_backward,
    selfloop_feedback_cycle,
    selfloop_forward_connection,
    simple_cycle,
    true_double_cycle
]
input_inits = [
    chebyshev_mapping,
    logistic_mapping,
    minimal_init,
    minimal_init(; sampling_type = :irrational_sample!),
    modified_lm(; factor = 4),
    scaled_rand,
    weighted_init,
    weighted_minimal
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
        delay_line_backward,
        cycle_jumps,
        simple_cycle
    ]
        dl = init(res_size, res_size)
        if init === delay_line_backward
            @test unique(dl) == Float32.([0.0, 0.1, 0.2])
        else
            @test unique(dl) == Float32.([0.0, 0.1])
        end
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
        minimal_init(; sampling_type = :irrational_sample!)
    ]
        dl = init(res_size, in_size)
        @test sort(unique(dl)) == Float32.([-0.1, 0.1])
    end
end
