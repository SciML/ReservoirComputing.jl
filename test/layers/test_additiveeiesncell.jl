using Test
using Random
using LinearAlgebra
using ReservoirComputing

const _W_I = (rng, m, n) -> Matrix{Float32}(I, m, n)
const _W_Z = (rng, m, n) -> zeros(Float32, m, n)
const _Z32 = (rng, dims...) -> zeros(Float32, dims...)

@testset "AdditiveEIESNCell: constructor & show" begin
    cell = AdditiveEIESNCell(3 => 5; exc_recurrence_scale = 0.8, input_scale = 0.5)
    io = IOBuffer()
    show(io, cell)
    shown = String(take!(io))

    @test occursin("AdditiveEIESNCell(3 => 5", shown)
    @test occursin("exc_recurrence_scale=0.8", shown)
    @test occursin("input_scale=0.5", shown)
end

@testset "AdditiveEIESNCell: initialparameters shapes" begin
    rng = MersenneTwister(1)
    cell = AdditiveEIESNCell(
        3 => 4;
        init_input = _W_I,
        init_reservoir = _W_I
    )
    ps = initialparameters(rng, cell)

    @test haskey(ps, :input_matrix)
    @test haskey(ps, :reservoir_matrix)
    @test size(ps.input_matrix) == (4, 3)
    @test size(ps.reservoir_matrix) == (4, 4)
end

@testset "AdditiveEIESNCell: initialstates carries RNG" begin
    rng = MersenneTwister(2)
    cell = AdditiveEIESNCell(2 => 2)
    st = initialstates(rng, cell)

    @test haskey(st, :rng)
end

@testset "AdditiveEIESNCell: forward single step (vector)" begin
    cell = AdditiveEIESNCell(
        3 => 3,
        identity;
        init_input = _W_I,
        init_reservoir = _W_Z,
        init_state = _Z32
    )
    ps = initialparameters(MersenneTwister(0), cell)
    x = Float32[1, 2, 3]
    h0 = zeros(Float32, 3)
    (y_tuple, _) = cell((x, (h0,)), ps, NamedTuple())
    y, (h1,) = y_tuple

    @test size(y) == (3,)
    @test size(h1) == (3,)
end

@testset "AdditiveEIESNCell: forward batch input" begin
    cell = AdditiveEIESNCell(
        3 => 3,
        identity;
        init_input = _W_I,
        init_reservoir = _W_Z,
        init_state = _Z32
    )
    ps = initialparameters(MersenneTwister(0), cell)
    X = Float32[1 2; 3 4; 5 6]
    H0 = zeros(Float32, 3, 2)
    (Y_tuple, _) = cell((X, (H0,)), ps, NamedTuple())
    Y, _ = Y_tuple

    @test size(Y) == (3, 2)
end

@testset "AdditiveEIESNCell: outer call initializes hidden state" begin
    rng = MersenneTwister(123)
    cell = AdditiveEIESNCell(
        2 => 2,
        identity;
        init_input = _W_I,
        init_reservoir = _W_Z,
        init_state = _Z32
    )
    ps = initialparameters(rng, cell)
    st = initialstates(rng, cell)
    x = Float32[7, 9]
    (y_tuple, st2) = cell(x, ps, st)
    y, _ = y_tuple

    @test size(y) == (2, 1)
    @test haskey(st2, :rng)
end
