using ReservoirComputing
using LinearAlgebra
using Random

const res_size = 30
const in_size = 3
const radius = 1.0
const sparsity = 0.1
const weight = 0.2
const jump_size = 3
const rng = Random.default_rng()

function check_radius(matrix, target_radius; tolerance = 1e-5)
    eigenvalues = eigvals(matrix)
    spectral_radius = maximum(abs.(eigenvalues))
    return isapprox(spectral_radius, target_radius, atol = tolerance)
end

ft = [Float16, Float32, Float64]
reservoir_inits = [
    rand_sparse,
    delay_line,
    delay_line_backward,
    cycle_jumps,
    simple_cycle,
    pseudo_svd
]
input_inits = [
    scaled_rand,
    weighted_init,
    sparse_init,
    minimal_init,
    minimal_init(; sampling_type = :irrational)
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
        minimal_init(; sampling_type = :irrational)
    ]
        dl = init(res_size, in_size)
        @test sort(unique(dl)) == Float32.([-0.1, 0.1])
    end
end
