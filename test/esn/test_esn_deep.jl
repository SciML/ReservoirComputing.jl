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
    B == 1 ? zeros(Float32, m) : zeros(Float32, m, B)
end

function _pin_identity_readout(ps::NamedTuple; out_dims::Integer, in_dims::Integer)
    ro_ps = haskey(ps.readout, :bias) ?
            (weight = _I32(out_dims, in_dims), bias = _Z32(out_dims)) :
            (weight = _I32(out_dims, in_dims),)
    merge(ps, (readout = ro_ps,))
end

@testset "DeepESN model" begin
    @testset "constructor & parameter/state shapes (no modifiers)" begin
        rng = MersenneTwister(1)
        in_dims = 4
        res_dims = [5, 6, 7]
        out_dims = 3
        desn = DeepESN(in_dims, res_dims, out_dims, identity;
            leak_coefficient = 1.0,
            init_reservoir = _W_ZZ,
            init_input = _W_I,
            init_bias = _O32,
            init_state = init_state3,
            use_bias = False(),
            state_modifiers = nothing,
            readout_activation = identity)
        ps, st = setup(rng, desn)
        @test length(desn.cells) == length(res_dims)
        @test length(ps.cells) == length(res_dims)
        @test length(st.cells) == length(res_dims)
        @test size(ps.readout.weight) == (out_dims, last(res_dims))
        @test length(ps.states_modifiers) == length(res_dims)
        @test length(st.states_modifiers) == length(res_dims)
    end

    @testset "forward: vector as batch=1, identity pipeline across layers" begin
        rng = MersenneTwister(2)
        D = 3
        depth = 3
        desn = DeepESN(D, fill(D, depth), D, identity;
            leak_coefficient = 1.0,
            init_reservoir = _W_ZZ,
            init_input = _W_I,
            init_bias = _O32,
            init_state = init_state3,
            use_bias = False(),
            state_modifiers = nothing,
            readout_activation = identity)
        ps, st = setup(rng, desn)
        ps = _pin_identity_readout(ps; out_dims = D, in_dims = D)
        x = Float32[1, 2, 3]
        X = reshape(x, :, 1)
        Y, st2 = desn(X, ps, st)
        @test size(Y) == (D, 1)
        @test vec(Y) ≈ x
        @test haskey(st2, :cells) && haskey(st2, :states_modifiers) && haskey(st2, :readout)
    end

    @testset "forward: batch matrix, identity across layers" begin
        rng = MersenneTwister(3)
        D, B = 4, 5
        desn = DeepESN(D, [D, D], D, identity;
            leak_coefficient = 1.0,
            init_reservoir = _W_ZZ,
            init_input = _W_I,
            init_bias = _O32,
            init_state = init_state3,
            use_bias = False(),
            state_modifiers = nothing,
            readout_activation = identity)
        ps, st = setup(rng, desn)
        ps = _pin_identity_readout(ps; out_dims = D, in_dims = D)
        X = reshape(Float32.(1:(D * B)), D, B)
        Y, _ = desn(X, ps, st)
        @test size(Y) == (D, B)
        @test Y ≈ X
    end

    @testset "state_modifiers per layer are applied in order" begin
        rng = MersenneTwister(4)
        D = 3
        mods = (x -> x .+ 1.0f0, x -> 2.0f0 .* x)
        desn = DeepESN(D, [D, D], D, identity;
            leak_coefficient = 1.0,
            init_reservoir = _W_ZZ,
            init_input = _W_I,
            init_bias = _O32,
            init_state = init_state3,
            use_bias = False(),
            state_modifiers = mods,
            readout_activation = identity)
        ps, st = setup(rng, desn)
        ps = _pin_identity_readout(ps; out_dims = D, in_dims = D)
        x = Float32[0, 1, -2]
        X = reshape(x, :, 1)
        Y, _ = desn(X, ps, st)
        @test vec(Y) ≈ 2.0f0 .* (x .+ 1.0f0)
    end

    @testset "depth convenience constructor" begin
        rng = MersenneTwister(5)
        D = 2
        desn = DeepESN(D, D, D, identity; depth = 4,
            leak_coefficient = 1.0,
            init_reservoir = _W_ZZ,
            init_input = _W_I,
            init_bias = _O32,
            init_state = init_state3,
            use_bias = False(),
            state_modifiers = nothing,
            readout_activation = identity)
        @test length(desn.cells) == 4
        ps, st = setup(rng, desn)
        ps = _pin_identity_readout(ps; out_dims = D, in_dims = D)
        X = Float32[1 2 3; 4 5 6]
        Y, _ = desn(X, ps, st)
        @test size(Y) == size(X)
        @test Y ≈ X
    end

    @testset "resetcarry! sets carries using a function" begin
        rng = MersenneTwister(6)
        D = 3
        desn = DeepESN(D, [4, 5], 2, identity;
            leak_coefficient = 1.0,
            init_reservoir = _W_ZZ,
            init_input = _W_I,
            init_bias = _O32,
            init_state = init_state3,
            use_bias = False(),
            state_modifiers = nothing,
            readout_activation = identity)
        ps, st = setup(rng, desn)
        f = (rng, m) -> ones(Float32, m)
        st2 = resetcarry!(rng, desn, st; init_carry = f)
        @test length(st2.cells) == 2
        @test st2.cells[1].carry !== nothing
        @test st2.cells[2].carry !== nothing
        @test length(first(st2.cells[1].carry)) == desn.cells[1].cell.out_dims
        @test length(first(st2.cells[2].carry)) == desn.cells[2].cell.out_dims
    end

    @testset "resetcarry! with per-layer initializers" begin
        rng = MersenneTwister(7)
        D = 3
        desn = DeepESN(D, [D, D, D], D, identity;
            leak_coefficient = 1.0,
            init_reservoir = _W_ZZ,
            init_input = _W_I,
            init_bias = _O32,
            init_state = init_state3,
            use_bias = False(),
            state_modifiers = nothing,
            readout_activation = identity)
        st = initialstates(rng, desn)
        inits = ((rng, m) -> fill(1.0f0, m), nothing, (rng, m) -> fill(3.0f0, m))
        st2 = resetcarry!(rng, desn, st; init_carry = inits)
        @test st2.cells[1].carry !== nothing
        @test st2.cells[2].carry === nothing
        @test st2.cells[3].carry !== nothing
        @test first(st2.cells[1].carry) ≈ fill(1.0f0, desn.cells[1].cell.out_dims)
        @test first(st2.cells[3].carry) ≈ fill(3.0f0, desn.cells[3].cell.out_dims)
    end

    @testset "collectstates over matrix preserves identity mapping" begin
        rng = MersenneTwister(8)
        D = 3
        Tlen = 5
        desn = DeepESN(D, [D, D], D, identity;
            leak_coefficient = 1.0,
            init_reservoir = _W_ZZ,
            init_input = _W_I,
            init_bias = _O32,
            init_state = init_state3,
            use_bias = False(),
            state_modifiers = nothing,
            readout_activation = identity)
        ps, st = setup(rng, desn)
        X = reshape(Float32.(1:(D * Tlen)), D, Tlen)
        S, st2 = collectstates(desn, X, ps, st)
        @test size(S) == size(X)
        @test S ≈ X
        @test haskey(st2, :cells)
    end

    @testset "collectstates over vector" begin
        rng = MersenneTwister(9)
        D = 4
        desn = DeepESN(D, [D], D, identity;
            leak_coefficient = 1.0,
            init_reservoir = _W_ZZ,
            init_input = _W_I,
            init_bias = _O32,
            init_state = init_state3,
            use_bias = False(),
            state_modifiers = nothing,
            readout_activation = identity)
        ps, st = setup(rng, desn)
        x = Float32[1, 2, 3, 4]
        S, _ = collectstates(desn, x, ps, st)
        @test size(S) == (D, 1)
        @test vec(S) ≈ x
    end
end
