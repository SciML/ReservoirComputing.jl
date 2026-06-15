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

function build_memresesn(
        in_dims::Integer, mem_dims::Integer, res_dims::Integer;
        activation = identity,
        use_bias = False(),
        init_input = _W_I,
        init_reservoir = _W_ZZ,
        init_memory = _W_I,
        init_orthogonal = _W_ZZ,
        init_bias = _O32,
        init_state = init_state3,
        alpha = 1.0,
        beta = 1.0,
    )
    return MemoryResESNCell(
        (in_dims, mem_dims) => res_dims, activation;
        use_bias = use_bias,
        init_input = init_input,
        init_reservoir = init_reservoir,
        init_memory = init_memory,
        init_orthogonal = init_orthogonal,
        init_bias = init_bias,
        init_state = init_state,
        alpha = alpha,
        beta = beta,
    )
end

@testset "MemoryResESNCell" begin
    @testset "constructor & show" begin
        cell = build_memresesn(3, 4, 5)
        io = IOBuffer()
        show(io, cell)
        shown = String(take!(io))

        @test occursin("MemoryResESNCell(", shown)
        @test occursin("use_bias=false", shown)

        cell_ab = build_memresesn(3, 4, 5; alpha = 0.25, beta = 0.75)
        io2 = IOBuffer()
        show(io2, cell_ab)
        shown2 = String(take!(io2))
        @test occursin("alpha=0.25", shown2)
        @test occursin("beta=0.75", shown2)
    end

    @testset "initialparameters shapes (no bias)" begin
        rng = MersenneTwister(1)
        in_dims, mem_dims, res_dims = 3, 4, 5

        cell = build_memresesn(in_dims, mem_dims, res_dims)
        ps = initialparameters(rng, cell)

        @test haskey(ps, :input_matrix)
        @test haskey(ps, :reservoir_matrix)
        @test haskey(ps, :memory_matrix)
        @test haskey(ps, :orthogonal_matrix)
        @test !haskey(ps, :bias)
        @test size(ps.input_matrix) == (res_dims, in_dims)
        @test size(ps.reservoir_matrix) == (res_dims, res_dims)
        @test size(ps.memory_matrix) == (res_dims, mem_dims)
        @test size(ps.orthogonal_matrix) == (res_dims, res_dims)
    end

    @testset "initialparameters includes optional bias" begin
        rng = MersenneTwister(2)
        in_dims, mem_dims, res_dims = 3, 4, 5

        cell = build_memresesn(
            in_dims, mem_dims, res_dims;
            use_bias = True(), init_bias = _O32
        )
        ps = initialparameters(rng, cell)

        @test haskey(ps, :bias)
        @test size(ps.bias) == (res_dims,)
    end

    @testset "initialstates has rng" begin
        rng = MersenneTwister(3)
        cell = build_memresesn(3, 4, 5)
        st = initialstates(rng, cell)
        @test haskey(st, :rng)
    end

    @testset "forward: direct input + memory contribution combine inside φ" begin
        rng = MersenneTwister(4)
        D = 3
        # α=0 isolates the φ(W_in·u + W_m·m + W_r·h) branch.
        cell = build_memresesn(
            D, D, D;
            activation = identity,
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_memory = _W_I,
            init_orthogonal = _W_ZZ,
            alpha = 0.0,
            beta = 1.0,
        )
        ps = initialparameters(rng, cell)
        st = initialstates(rng, cell)

        u = Float32[1, 2, 3]
        m = Float32[10, 20, 30]
        h0 = zeros(Float32, D)

        (y_tuple, _) = cell((u, (h0, m)), ps, st)
        h_new, carry = y_tuple

        @test h_new ≈ u .+ m
        @test first(carry) ≈ h_new
    end

    @testset "forward: orthogonal skip honors α" begin
        rng = MersenneTwister(5)
        D = 3
        # β=0 isolates the α·O·h branch.
        cell = build_memresesn(
            D, D, D;
            activation = identity,
            init_input = _W_ZZ,
            init_reservoir = _W_ZZ,
            init_memory = _W_ZZ,
            init_orthogonal = _W_I,
            alpha = 0.5,
            beta = 0.0,
        )
        ps = initialparameters(rng, cell)
        st = initialstates(rng, cell)

        u = Float32[7, 8, 9]
        m = Float32[1, 2, 3]
        h0 = Float32[4, -2, 6]

        (y_tuple, _) = cell((u, (h0, m)), ps, st)
        h_new, _ = y_tuple

        @test h_new ≈ 0.5f0 .* h0
    end

    @testset "forward: W_m = 0 reduces to ResESNCell" begin
        rng = MersenneTwister(6)
        D = 4
        ortho = qr(randn(Float32, D, D)).Q[:, :] |> Matrix{Float32}

        cell_mem = build_memresesn(
            D, D, D;
            activation = tanh,
            init_input = (rng, m, n) -> Float32(0.3) .* (_I32(m, n)),
            init_reservoir = (rng, m, n) -> Float32(0.1) .* (_I32(m, n)),
            init_memory = _W_ZZ,
            init_orthogonal = (rng, m, n) -> ortho,
            alpha = 0.7,
            beta = 0.4,
        )

        cell_ref = ResESNCell(
            D => D, tanh;
            init_input = (rng, m, n) -> Float32(0.3) .* (_I32(m, n)),
            init_reservoir = (rng, m, n) -> Float32(0.1) .* (_I32(m, n)),
            init_orthogonal = (rng, m, n) -> ortho,
            init_state = init_state3,
            alpha = 0.7, beta = 0.4,
        )

        ps_mem = initialparameters(rng, cell_mem)
        st_mem = initialstates(rng, cell_mem)
        ps_ref = initialparameters(rng, cell_ref)
        st_ref = initialstates(rng, cell_ref)

        u = Float32[0.5, -0.3, 0.2, 0.8]
        m = Float32[1.5, -2.1, 0.9, -0.6]
        h0 = Float32[0.1, -0.2, 0.3, -0.4]

        (yt_mem, _) = cell_mem((u, (h0, m)), ps_mem, st_mem)
        (yt_ref, _) = cell_ref((u, (h0,)), ps_ref, st_ref)

        @test first(yt_mem) ≈ first(yt_ref)
    end

    @testset "forward: matrix batch" begin
        rng = MersenneTwister(7)
        D, B = 3, 2
        cell = build_memresesn(
            D, D, D;
            activation = identity,
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_memory = _W_I,
            init_orthogonal = _W_ZZ,
            alpha = 0.0,
            beta = 1.0,
        )
        ps = initialparameters(rng, cell)
        st = initialstates(rng, cell)

        U = Float32[1 2; 3 4; 5 6]
        M = Float32[10 20; 30 40; 50 60]
        H0 = zeros(Float32, D, B)

        (y_tuple, _) = cell((U, (H0, M)), ps, st)
        h_new, _ = y_tuple

        @test h_new ≈ U .+ M
    end

    @testset "forward: implicit init builds hidden and memory states" begin
        rng = MersenneTwister(8)
        D = 3
        cell = build_memresesn(
            D, D, D;
            activation = identity,
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_memory = _W_ZZ,
            init_orthogonal = _W_ZZ,
            alpha = 0.0, beta = 1.0,
        )
        ps = initialparameters(rng, cell)
        st = initialstates(rng, cell)

        u = Float32[1, 2, 3]
        (y_tuple, _) = cell(u, ps, st)
        h_new, _ = y_tuple
        @test h_new ≈ u
    end
end
