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

@testset "ResRMNCell: constructor & show" begin
    cell = ResRMNCell(
        (3, 4) => 5;
        use_bias = False(), alpha = 0.4, beta = 0.7,
        init_input = _W_I, init_reservoir = _W_I,
        init_memory = _W_I, init_orthogonal = _W_I
    )
    io = IOBuffer()
    show(io, cell)
    shown = String(take!(io))
    @test occursin("ResRMNCell((3, 4) => 5", shown)
    @test occursin(Regex("alpha=0\\.4(f0)?"), shown)
    @test occursin(Regex("beta=0\\.7(f0)?"), shown)
    @test occursin("use_bias=false", shown)
end

@testset "ResRMNCell: initialparameters shapes" begin
    rng = MersenneTwister(1)
    cell = ResRMNCell(
        (3, 4) => 5;
        use_bias = False(),
        init_input = _W_I, init_reservoir = _W_I,
        init_memory = _W_I, init_orthogonal = _W_I
    )
    ps = initialparameters(rng, cell)
    @test haskey(ps, :input_matrix) && size(ps.input_matrix) == (5, 3)
    @test haskey(ps, :memory_matrix) && size(ps.memory_matrix) == (5, 4)
    @test haskey(ps, :reservoir_matrix) && size(ps.reservoir_matrix) == (5, 5)
    @test haskey(ps, :orthogonal_matrix) && size(ps.orthogonal_matrix) == (5, 5)
    @test !haskey(ps, :bias)
end

@testset "ResRMNCell: forward — alpha*O*h + beta*phi(W_in u + W_mem m + W_r h)" begin
    cell = ResRMNCell(
        (3, 3) => 3, identity;
        use_bias = False(),
        alpha = 0.4, beta = 0.7,
        init_input = _W_I, init_memory = _W_I,
        init_reservoir = _W_ZZ, init_orthogonal = _W_I,
        init_state = _Z32
    )
    ps = initialparameters(MersenneTwister(0), cell)
    u = Float32[1, 0, 2]
    m = Float32[0, 1, 0]
    h0 = Float32[0, 3, 1]
    (y_tuple, _) = cell(((u, m), (h0,)), ps, NamedTuple())
    y, _ = y_tuple
    # expected = 0.4 * h0 + 0.7 * (u + m)
    @test y ≈ 0.4f0 .* h0 .+ 0.7f0 .* (u .+ m)
end

@testset "ResRMNCell: reduces to ResESNCell when W_mem = 0" begin
    # equivalence: ResRMNCell with zero memory kernel equals ResESNCell on u.
    shared_in = _W_I
    shared_res = _W_ZZ
    shared_ortho = _W_I
    rmn = ResRMNCell(
        (3, 4) => 3, identity;
        use_bias = False(),
        alpha = 0.3, beta = 0.8,
        init_input = shared_in, init_memory = _W_ZZ,
        init_reservoir = shared_res, init_orthogonal = shared_ortho,
        init_state = _Z32
    )
    res = ResESNCell(
        3 => 3, identity;
        use_bias = False(),
        alpha = 0.3, beta = 0.8,
        init_input = shared_in, init_reservoir = shared_res,
        init_orthogonal = shared_ortho, init_state = _Z32
    )

    ps_rmn = initialparameters(MersenneTwister(0), rmn)
    ps_res = initialparameters(MersenneTwister(0), res)
    u = Float32[1, 2, 3]
    m = Float32[5, 6, 7, 8]
    h0 = Float32[0.1, 0.2, 0.3]
    (y_rmn_tuple, _) = rmn(((u, m), (h0,)), ps_rmn, NamedTuple())
    (y_res_tuple, _) = res((u, (h0,)), ps_res, NamedTuple())
    y_rmn, _ = y_rmn_tuple
    y_res, _ = y_res_tuple
    @test y_rmn ≈ y_res
end

@testset "ResRMNCell: outer call computes its own initial hidden state" begin
    rng = MersenneTwister(123)
    cell = ResRMNCell(
        (2, 2) => 2, identity;
        use_bias = False(),
        alpha = 1.0, beta = 1.0,
        init_input = _W_I, init_memory = _W_ZZ,
        init_reservoir = _W_ZZ, init_orthogonal = _W_ZZ,
        init_state = init_state3
    )
    ps = initialparameters(rng, cell)
    st = initialstates(rng, cell)
    u = Float32[7, 9]
    m = Float32[0, 0]
    (y_tuple, st2) = cell((u, m), ps, st)
    y, _ = y_tuple
    @test y ≈ u
    @test haskey(st2, :rng)
end

@testset "ResRMNCell: batch forward" begin
    cell = ResRMNCell(
        (3, 2) => 3, identity;
        use_bias = False(),
        alpha = 1.0, beta = 1.0,
        init_input = _W_I, init_memory = _W_ZZ,
        init_reservoir = _W_ZZ, init_orthogonal = _W_ZZ,
        init_state = _Z32
    )
    ps = initialparameters(MersenneTwister(0), cell)
    U = Float32[1 2; 3 4; 5 6]
    M = Float32[0 0; 0 0]
    H0 = zeros(Float32, 3, 2)
    (Y_tuple, _) = cell(((U, M), (H0,)), ps, NamedTuple())
    Y, _ = Y_tuple
    @test size(Y) == (3, 2)
    @test Y ≈ U
end
