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
    return B == 1 ? zeros(Float32, m) : zeros(Float32, m, B)
end

@testset "MemoryCell: constructor & show" begin
    cell = MemoryCell(3 => 5; use_bias = False())
    io = IOBuffer()
    show(io, cell)
    shown = String(take!(io))
    @test occursin("MemoryCell(3 => 5", shown)
    @test occursin("use_bias=false", shown)
end

@testset "MemoryCell: initialparameters shapes & bias flag" begin
    rng = MersenneTwister(1)
    cell = MemoryCell(
        3 => 4;
        use_bias = False(), init_input = _W_I, init_reservoir = _W_I
    )
    ps = initialparameters(rng, cell)
    @test haskey(ps, :input_matrix)
    @test haskey(ps, :reservoir_matrix)
    @test size(ps.input_matrix) == (4, 3)
    @test size(ps.reservoir_matrix) == (4, 4)
    @test !haskey(ps, :bias)

    cell_b = MemoryCell(
        3 => 4;
        use_bias = True(), init_input = _W_I, init_reservoir = _W_I,
        init_bias = _O32
    )
    ps_b = initialparameters(rng, cell_b)
    @test haskey(ps_b, :bias)
    @test length(ps_b.bias) == 4
end

@testset "MemoryCell: forward is linear (no activation by default)" begin
    cell = MemoryCell(
        3 => 3;
        use_bias = False(), init_input = _W_I, init_reservoir = _W_ZZ,
        init_state = _Z32
    )
    ps = initialparameters(MersenneTwister(0), cell)
    x = Float32[1, 2, 3]
    h0 = Float32[10, 20, 30]
    (y_tuple, _) = cell((x, (h0,)), ps, NamedTuple())
    y, _ = y_tuple
    @test y ≈ x
end

@testset "MemoryCell: default cyclic recurrent shifts the state" begin
    # default init_reservoir is simple_cycle with unit weight.
    # the cyclic (shift) orthogonal matrix rotates entries of h(t-1).
    cell = MemoryCell(
        3 => 3;
        use_bias = False(), init_input = _W_ZZ, init_state = _Z32
    )
    ps = initialparameters(MersenneTwister(0), cell)
    h0 = Float32[1, 2, 3]
    zero_inp = zeros(Float32, 3)
    (y_tuple, _) = cell((zero_inp, (h0,)), ps, NamedTuple())
    y, _ = y_tuple
    # simple_cycle places the weight at W[i, j] where i = j + 1 (and i = 1, j = D).
    # so (C * h)[1] = h[3], (C * h)[2] = h[1], (C * h)[3] = h[2].
    @test y ≈ Float32[h0[3], h0[1], h0[2]]
end

@testset "MemoryCell: outer call computes its own initial hidden state" begin
    rng = MersenneTwister(123)
    cell = MemoryCell(
        2 => 2;
        use_bias = False(), init_input = _W_I, init_reservoir = _W_ZZ,
        init_state = init_state3
    )
    ps = initialparameters(rng, cell)
    st = initialstates(rng, cell)
    x = Float32[7, 9]
    (y_tuple, st2) = cell(x, ps, st)
    y, _ = y_tuple
    @test y ≈ x
    @test haskey(st2, :rng)
end

@testset "MemoryCell: batch forward" begin
    cell = MemoryCell(
        3 => 3;
        use_bias = False(), init_input = _W_I, init_reservoir = _W_ZZ,
        init_state = _Z32
    )
    ps = initialparameters(MersenneTwister(0), cell)
    X = Float32[1 2; 3 4; 5 6]
    H0 = zeros(Float32, 3, 2)
    (Y_tuple, _) = cell((X, (H0,)), ps, NamedTuple())
    Y, _ = Y_tuple
    @test size(Y) == (3, 2)
    @test Y ≈ X
end
