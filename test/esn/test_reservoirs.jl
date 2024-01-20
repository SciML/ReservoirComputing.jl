using ReservoirComputing
using LinearAlgebra
using Random
include("../utils.jl")

const res_size = 20
const radius = 1.0
const sparsity = 0.1
const weight = 0.2
const jump_size = 3
const rng = Random.default_rng()

dtypes = [Float16, Float32, Float64]
reservoir_inits = [rand_sparse, delay_line]

@testset "Sizes and types" begin
    for init in reservoir_inits
        for dt in dtypes
            #sizes
            @test size(init(res_size, res_size)) == (res_size, res_size)
            @test size(init(rng, res_size, res_size)) == (res_size, res_size)
            #types
            @test eltype(init(dt, res_size, res_size)) == dt
            @test eltype(init(rng, dt, res_size, res_size)) == dt
            #closure
            cl = init(rng)
            @test cl(dt, res_size, res_size) isa AbstractArray{dt}
        end
    end
end

@testset "rand_sparse" begin
    sp = rand_sparse(res_size, res_size)
    @test check_radius(sp, radius)
end

@testset "delay_line" begin
    dl = delay_line(res_size, res_size)
    @test unique(dl) == Float32.([0.0, 0.1])
end

