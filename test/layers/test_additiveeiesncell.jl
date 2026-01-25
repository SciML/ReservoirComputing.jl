using Test
using Random
using LinearAlgebra
using ReservoirComputing
using LuxCore

const _W_I = (rng, m, n) -> Matrix{Float32}(I, m, n)
const _W_Z = (rng, m, n) -> zeros(Float32, m, n)
const _B_1 = (rng, m) -> ones(Float32, m)
const _Z32 = (rng, dims...) -> zeros(Float32, dims...)

@testset "AdditiveEIESNCell: constructor & show" begin
    cell = AdditiveEIESNCell(3 => 5; exc_recurrence_scale = 0.8, input_activation = abs, use_bias = true)
    io = IOBuffer()
    show(io, cell)
    shown = String(take!(io))

    @test occursin("AdditiveEIESNCell(3 => 5", shown)
    @test occursin("exc_recurrence_scale=0.8", shown)
    @test occursin("input_activation=abs", shown)
    @test occursin("use_bias=true", shown)
end

@testset "AdditiveEIESNCell: tuple activation logic" begin
    cell1 = AdditiveEIESNCell(3 => 5, tanh)

    @test cell1.activation isa Tuple
    @test length(cell1.activation) == 2
    @test cell1.activation[1] == tanh
    @test cell1.activation[2] == tanh

    cell2 = AdditiveEIESNCell(3 => 5, (tanh, identity))

    @test cell2.activation[1] == tanh
    @test cell2.activation[2] == identity
end

@testset "AdditiveEIESNCell: parameters with/without bias" begin
    rng = MersenneTwister(1)
    cell_bias = AdditiveEIESNCell(
        3 => 4;
        init_input = _W_I,
        init_reservoir = _W_I,
        use_bias = true,
        init_bias = _B_1
    )
    ps = initialparameters(rng, cell_bias)

    @test haskey(ps, :input_matrix)
    @test haskey(ps, :reservoir_matrix)
    @test haskey(ps, :bias_ex)
    @test haskey(ps, :bias_inh)
    @test haskey(ps, :bias_in)
    @test size(ps.bias_ex) == (4,)

    cell_nobias = AdditiveEIESNCell(3 => 4; use_bias = false)
    ps_nb = initialparameters(rng, cell_nobias)

    @test !haskey(ps_nb, :bias_ex)
    @test !haskey(ps_nb, :bias_in)
end

@testset "AdditiveEIESNCell: initialstates carries RNG" begin
    rng = MersenneTwister(2)
    cell = AdditiveEIESNCell(2 => 2)
    st = initialstates(rng, cell)

    @test haskey(st, :rng)
end

@testset "AdditiveEIESNCell: forward deterministic (input activation)" begin

    cell = AdditiveEIESNCell(
        3 => 3,
        identity;
        input_activation = abs,
        init_input = _W_I,
        init_reservoir = _W_Z,
        init_state = _Z32,
        use_bias = false
    )
    ps = initialparameters(MersenneTwister(0), cell)
    x = Float32[-5, -5, -5]
    h0 = zeros(Float32, 3)
    (y_tuple, _) = cell((x, (h0,)), ps, NamedTuple())
    y, (h1,) = y_tuple

    @test all(y .== 5.0f0)
    @test size(y) == (3,)
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
