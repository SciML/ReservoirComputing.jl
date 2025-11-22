using Test
using Random
using LinearAlgebra
using ReservoirComputing
using Static

const _I32 = (m, n) -> Matrix{Float32}(I, m, n)
const _Z32 = m -> zeros(Float32, m)
const _O32 = (rng, m) -> zeros(Float32, m)
const _W_I = (rng, m, n) -> _I32(m, n)
const _W_ZZ = (rng, m, n) -> zeros(Float32, m, n)
function init_state3(rng::AbstractRNG, m::Integer, B::Integer)
    B == 1 ? zeros(Float32, m) : zeros(Float32, m, B)
end

@testset "ESNCell: constructor & show" begin
    esn = ESNCell(3 => 5; leak_coefficient = 0.3, use_bias = False())
    io = IOBuffer()
    show(io, esn)
    shown = String(take!(io))
    @test occursin("ESNCell(3 => 5", shown)
    @test occursin("leak_coefficient=0.3", shown)
    @test occursin("use_bias=false", shown)
end

@testset "ESNCell: initialparameters shapes & bias flag" begin
    rng = MersenneTwister(1)

    esn_nobias = ESNCell(3 => 4; use_bias = False(),
        init_input = _W_I, init_reservoir = _W_I, init_bias = _O32)

    ps_nb = initialparameters(rng, esn_nobias)
    @test haskey(ps_nb, :input_matrix)
    @test haskey(ps_nb, :reservoir_matrix)
    @test !haskey(ps_nb, :bias)
    @test size(ps_nb.input_matrix) == (4, 3)
    @test size(ps_nb.reservoir_matrix) == (4, 4)

    esn_bias = ESNCell(3 => 4; use_bias = True(),
        init_input = _W_I, init_reservoir = _W_I, init_bias = _O32)

    ps_b = initialparameters(rng, esn_bias)
    @test haskey(ps_b, :bias)
    @test length(ps_b.bias) == 4
end

@testset "ESNCell: initialstates carries RNG replica" begin
    rng = MersenneTwister(2)
    esn = ESNCell(2 => 2)
    st = initialstates(rng, esn)
    @test haskey(st, :rng)
end

@testset "ESNCell: forward (vector) — identity + leak=1 gives linear map" begin
    esn = ESNCell(3 => 3, identity; use_bias = False(),
        init_input = _W_I, init_reservoir = _W_I, init_bias = _O32,
        init_state = _Z32, leak_coefficient = 1.0)

    ps = initialparameters(MersenneTwister(0), esn)
    st = NamedTuple()
    x = Float32[1, 2, 3]
    h0 = zeros(Float32, 3)

    (y_tuple, st2) = esn((x, (h0,)), ps, st)
    y, (hcarry,) = y_tuple
    @test y ≈ x
    @test hcarry ≈ y
    @test st2 === st
end

@testset "ESNCell: forward (vector) — leak extremes" begin
    esn0 = ESNCell(3 => 3, identity; use_bias = False(),
        init_input = _W_I, init_reservoir = _W_I, init_bias = _O32,
        init_state = _Z32, leak_coefficient = 0.0)

    ps0 = initialparameters(MersenneTwister(0), esn0)
    x = Float32[10, 20, 30]
    h0 = Float32[4, 5, 6]
    (y0_tuple, _) = esn0((x, (h0,)), ps0, NamedTuple())
    y0, _ = y0_tuple
    @test y0 ≈ h0

    esn1 = ESNCell(3 => 3, identity; use_bias = True(),
        init_input = _W_I, init_reservoir = _W_ZZ, init_bias = (rng, m) -> ones(Float32, m),
        init_state = _Z32, leak_coefficient = 1.0)

    ps1 = initialparameters(MersenneTwister(0), esn1)
    (y1_tuple, _) = esn1((x, (zeros(Float32, 3),)), ps1, NamedTuple())
    y1, _ = y1_tuple
    @test y1 ≈ x .+ 1.0f0
end

@testset "ESNCell: forward (matrix batch)" begin
    esn = ESNCell(3 => 3, identity; use_bias = False(),
        init_input = _W_I, init_reservoir = _W_I, init_bias = _O32,
        init_state = _Z32, leak_coefficient = 1.0)

    ps = initialparameters(MersenneTwister(0), esn)
    X = Float32[1 2; 3 4; 5 6]  # (3, 2)
    H0 = zeros(Float32, 3, 2)

    (Y_tuple, _) = esn((X, (H0,)), ps, NamedTuple())
    Y, _ = Y_tuple
    @test size(Y) == (3, 2)
    @test Y ≈ X
end

@testset "ESNCell: outer call computes its own initial hidden state" begin
    rng = MersenneTwister(123)
    esn = ESNCell(2 => 2, identity; use_bias = False(),
        init_input = _W_I, init_reservoir = _W_ZZ,
        init_state = init_state3, leak_coefficient = 1.0)

    ps = initialparameters(rng, esn)
    st = initialstates(rng, esn)
    x = Float32[7, 9]
    (y_tuple, st2) = esn(x, ps, st)
    y, _ = y_tuple
    @test y ≈ x
    @test haskey(st2, :rng)
end
