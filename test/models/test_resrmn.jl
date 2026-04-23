using Test
using Random
using ReservoirComputing
using LinearAlgebra
using Static

const _I32 = (m, n) -> Matrix{Float32}(I, m, n)
const _Z32 = m -> zeros(Float32, m)
const _O32 = (rng, m) -> zeros(Float32, m)
const _W_I = (rng, m, n) -> _I32(m, n)
const _W_ZZ = (rng, m, n) -> zeros(Float32, m, n)

function init_state3(rng::AbstractRNG, m::Integer, B::Integer)
    return B == 1 ? zeros(Float32, m) : zeros(Float32, m, B)
end

function _with_identity_readout(ps::NamedTuple; out_dims::Integer, in_dims::Integer)
    ro_ps = haskey(ps.readout, :bias) ?
        (weight = _I32(out_dims, in_dims), bias = _Z32(out_dims)) :
        (weight = _I32(out_dims, in_dims),)
    return merge(ps, (readout = ro_ps,))
end

function build_identity_resrmn(
        in_dims::Int, mem_dims::Int, res_dims::Int, out_dims::Int;
        state_modifiers = (), readout_activation = identity,
        alpha = 1.0, beta = 1.0,
        # by default zero-out the memory path so reservoir output == u
        init_memory = _W_ZZ
    )
    return ResRMN(
        in_dims, mem_dims, res_dims, out_dims, identity;
        alpha = alpha, beta = beta,
        init_input = _W_I, init_memory = init_memory,
        init_reservoir = _W_ZZ, init_orthogonal = _W_ZZ,
        init_bias = _O32, init_state = init_state3,
        use_bias = False(),
        init_memory_input = _W_ZZ, init_memory_reservoir = _W_ZZ,
        init_memory_bias = _O32, init_memory_state = init_state3,
        use_memory_bias = False(),
        state_modifiers = state_modifiers,
        readout_activation = readout_activation
    )
end

@testset "ResRMN: constructor & parameter/state shapes" begin
    rng = MersenneTwister(42)
    in_dims, mem_dims, res_dims, out_dims = 3, 5, 6, 4
    model = build_identity_resrmn(in_dims, mem_dims, res_dims, out_dims)
    ps, st = setup(rng, model)

    @test haskey(ps, :memory)
    @test haskey(ps.memory, :input_matrix) &&
        size(ps.memory.input_matrix) == (mem_dims, in_dims)
    @test haskey(ps.memory, :reservoir_matrix) &&
        size(ps.memory.reservoir_matrix) == (mem_dims, mem_dims)

    @test haskey(ps, :reservoir)
    @test haskey(ps.reservoir, :input_matrix) &&
        size(ps.reservoir.input_matrix) == (res_dims, in_dims)
    @test haskey(ps.reservoir, :memory_matrix) &&
        size(ps.reservoir.memory_matrix) == (res_dims, mem_dims)
    @test haskey(ps.reservoir, :reservoir_matrix) &&
        size(ps.reservoir.reservoir_matrix) == (res_dims, res_dims)
    @test haskey(ps.reservoir, :orthogonal_matrix) &&
        size(ps.reservoir.orthogonal_matrix) == (res_dims, res_dims)

    @test haskey(ps, :readout)
    @test size(ps.readout.weight) == (out_dims, res_dims)

    @test haskey(st, :memory)
    @test haskey(st, :reservoir)
    @test haskey(st, :states_modifiers)
    @test haskey(st, :readout)
    @test st.states_modifiers isa Tuple
end

@testset "ResRMN: forward (vector) with identity pipeline -> y == x" begin
    rng = MersenneTwister(0)
    D = 3
    model = build_identity_resrmn(D, D, D, D)
    ps, st = setup(rng, model)
    ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

    x = Float32[1, 2, 3]
    X = reshape(x, :, 1)
    Y, st2 = model(X, ps, st)
    @test size(Y) == (D, 1)
    @test vec(Y) ≈ x
    @test haskey(st2, :memory) && haskey(st2, :reservoir) &&
        haskey(st2, :states_modifiers) && haskey(st2, :readout)
end

@testset "ResRMN: batch forward with identity pipeline -> Y == X" begin
    rng = MersenneTwister(1)
    D, B = 3, 2
    model = build_identity_resrmn(D, D, D, D)
    ps, st = setup(rng, model)
    ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

    X = Float32[1 2; 3 4; 5 6]
    Y, _ = model(X, ps, st)
    @test size(Y) == (D, B)
    @test Y ≈ X
end

@testset "ResRMN: state_modifiers are applied" begin
    rng = MersenneTwister(2)
    D = 3
    model = build_identity_resrmn(
        D, D, D, D;
        state_modifiers = (x -> 2.0f0 .* x,)
    )
    ps, st = setup(rng, model)
    ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

    x = Float32[1, 2, 3]
    y, _ = model(x, ps, st)
    @test y ≈ 2.0f0 .* x
end

@testset "ResRMN: readout_activation is honored" begin
    rng = MersenneTwister(4)
    D = 3
    model = build_identity_resrmn(
        D, D, D, D;
        readout_activation = x -> max.(x, 0.0f0)
    )
    ps, st = setup(rng, model)
    ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

    x = Float32[-1, 0.5, -3]
    y, _ = model(x, ps, st)
    @test y ≈ max.(x, 0.0f0)
end

@testset "ResRMN: cyclic memory + W_mem propagates past inputs into reservoir" begin
    # use W_mem = I to route the memory state into the reservoir, and pick
    # simple_cycle (library default) for the memory recurrent matrix.
    rng = MersenneTwister(7)
    model = ResRMN(
        2, 3, 3, 3, identity;
        alpha = 0.0f0, beta = 1.0f0,
        init_input = _W_I, init_memory = _W_I,
        init_reservoir = _W_ZZ, init_orthogonal = _W_ZZ,
        init_state = init_state3,
        init_memory_input = _W_I,
        init_memory_state = init_state3
    )
    ps, st = setup(rng, model)
    data = Float32[1 0 0; 0 1 0]  # (2, 3)
    states, _ = collectstates(model, data, ps, st)
    # step 1: m=[1,0,0], u=[1,0], h = u + m (first 2 entries) → [2,0,0]
    # step 2: m = cycle(m) + W_in^m*u = [0,1,0] + [0,1,0] = [0,2,0]
    #         h = [0,1] (first 2 of u padded by 0) + m = [0,3,0]
    # step 3: m = cycle([0,2,0]) + 0 = [0,0,2]
    #         h = 0 + m = [0,0,2]
    @test states[:, 1] ≈ Float32[2, 0, 0]
    @test states[:, 2] ≈ Float32[0, 3, 0]
    @test states[:, 3] ≈ Float32[0, 0, 2]
end

@testset "ResRMN: resetcarry! clears both memory and reservoir carries" begin
    rng = MersenneTwister(9)
    model = build_identity_resrmn(3, 3, 3, 3)
    ps, st = setup(rng, model)
    # run one step to populate carries
    X = reshape(Float32[1, 2, 3], :, 1)
    _, st2 = model(X, ps, st)
    @test st2.memory.carry !== nothing
    @test st2.reservoir.carry !== nothing
    st3 = resetcarry!(MersenneTwister(0), model, st2)
    @test st3.memory.carry === nothing
    @test st3.reservoir.carry === nothing
end
