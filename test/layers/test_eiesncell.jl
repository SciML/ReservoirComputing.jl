using Test
using Random
using LinearAlgebra
using ReservoirComputing
using LuxCore

const _W_I = (rng, m, n) -> Matrix{Float32}(I, m, n)
const _W_Z = (rng, m, n) -> zeros(Float32, m, n)
const _Z32 = (rng, dims...) -> zeros(Float32, dims...)

@testset "EIESNCell: constructor & show" begin
    cell = EIESNCell(3 => 5; exc_recurrence_scale = 0.8)
    io = IOBuffer()
    show(io, cell)
    shown = String(take!(io))

    @test occursin("EIESNCell(3 => 5", shown)
    @test occursin("exc_recurrence_scale=0.8", shown)
    @test occursin("use_bias=true", shown)

    cell_tuple = EIESNCell(3 => 5, (tanh, identity))
    io2 = IOBuffer()
    show(io2, cell_tuple)
    shown2 = String(take!(io2))

    @test occursin("activation=(tanh, identity)", shown2)
end

@testset "EIESNCell: parameters & bias" begin
    rng = MersenneTwister(1)
    cell = EIESNCell(
        3 => 4;
        init_input = _W_I,
        init_reservoir = _W_I
    )
    ps = initialparameters(rng, cell)

    @test haskey(ps, :input_matrix)
    @test haskey(ps, :reservoir_matrix)
    @test haskey(ps, :bias_ex)
    @test haskey(ps, :bias_inh)
    @test size(ps.bias_ex) == (4,)

    cell_nb = EIESNCell(3 => 4; use_bias = false)
    ps_nb = initialparameters(rng, cell_nb)

    @test haskey(ps_nb, :input_matrix)
    @test !haskey(ps_nb, :bias_ex)
    @test !haskey(ps_nb, :bias_inh)
end

@testset "EIESNCell: initialstates carries RNG" begin
    rng = MersenneTwister(2)
    cell = EIESNCell(2 => 2)
    st = initialstates(rng, cell)

    @test haskey(st, :rng)
end

@testset "EIESNCell: forward single step (Tuple Activation)" begin
    cell = EIESNCell(
        3 => 3,
        (identity, abs);
        init_input = _W_I,
        init_reservoir = _W_Z,
        init_state = _Z32,
        use_bias = false
    )
    ps = initialparameters(MersenneTwister(0), cell)
    x = Float32[-1, -2, -3]
    h0 = zeros(Float32, 3)
    (y_tuple, _) = cell((x, (h0,)), ps, NamedTuple())
    y, (h1,) = y_tuple

    @test size(y) == (3,)
    @test all(!isnan, y)
end

@testset "EIESNCell: forward batch input" begin
    cell = EIESNCell(
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

@testset "EIESNCell: outer call initializes hidden state" begin
    rng = MersenneTwister(123)
    cell = EIESNCell(
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
