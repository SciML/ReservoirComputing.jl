using Test
using Random
using ReservoirComputing
using Static
using LinearAlgebra

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

@testset "HybridESN model" begin
    @testset "constructor & parameter/state shapes" begin
        rng = MersenneTwister(1)
        km_dims, in_dims, res_dims = 2, 3, 5
        out_dims = res_dims + km_dims
        km = x -> ones(Float32, km_dims, size(x, 2))
        hesn = HybridESN(km, km_dims, in_dims, res_dims, out_dims, identity;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            leak_coefficient = 1.0,
            state_modifiers = (),
            readout_activation = identity)
        ps, st = setup(rng, hesn)
        @test haskey(ps, :reservoir) && haskey(ps, :knowledge_model) &&
              haskey(ps, :states_modifiers) && haskey(ps, :readout)
        @test size(ps.reservoir.input_matrix) == (res_dims, in_dims + km_dims)
        @test size(ps.readout.weight) == (out_dims, res_dims + km_dims)
        @test haskey(st, :reservoir) && haskey(st, :knowledge_model) &&
              haskey(st, :states_modifiers) && haskey(st, :readout)
    end

    @testset "forward: vector as batch=1, identity pipeline with concatenated features" begin
        rng = MersenneTwister(2)
        km_dims, in_dims = 2, 3
        res_dims = in_dims + km_dims
        out_dims = res_dims + km_dims
        km = x -> ones(Float32, km_dims, size(x, 2))
        hesn = HybridESN(km, km_dims, in_dims, res_dims, out_dims, identity;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            leak_coefficient = 1.0,
            state_modifiers = (),
            readout_activation = identity)
        ps, st = setup(rng, hesn)
        ps = _pin_identity_readout(ps; out_dims = out_dims, in_dims = res_dims + km_dims)
        x = Float32[1, 2, 3]
        X = reshape(x, :, 1)
        Y, _ = hesn(X, ps, st)
        k = ones(Float32, km_dims, 1)
        expected = vcat(k, vcat(k, X))
        @test size(Y) == (out_dims, 1)
        @test Y ≈ expected
    end

    @testset "forward: batch matrix" begin
        rng = MersenneTwister(3)
        km_dims, in_dims, B = 1, 4, 5
        res_dims = in_dims + km_dims
        out_dims = res_dims + km_dims
        km = x -> fill(2.0f0, km_dims, size(x, 2))
        hesn = HybridESN(km, km_dims, in_dims, res_dims, out_dims, identity;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            leak_coefficient = 1.0,
            state_modifiers = (),
            readout_activation = identity)
        ps, st = setup(rng, hesn)
        ps = _pin_identity_readout(ps; out_dims = out_dims, in_dims = res_dims + km_dims)
        X = reshape(Float32.(1:(in_dims * B)), in_dims, B)
        Y, _ = hesn(X, ps, st)
        k = fill(2.0f0, km_dims, B)
        expected = vcat(k, vcat(k, X))
        @test size(Y) == (out_dims, B)
        @test Y ≈ expected
    end

    @testset "state_modifiers are applied before readout" begin
        rng = MersenneTwister(4)
        km_dims, in_dims = 2, 2
        res_dims = in_dims + km_dims
        out_dims = res_dims + km_dims
        km = x -> ones(Float32, km_dims, size(x, 2))
        hesn = HybridESN(km, km_dims, in_dims, res_dims, out_dims, identity;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            leak_coefficient = 1.0,
            state_modifiers = (x -> 2.0f0 .* x,),
            readout_activation = identity)
        ps, st = setup(rng, hesn)
        ps = _pin_identity_readout(ps; out_dims = out_dims, in_dims = res_dims + km_dims)
        x = Float32[3, -1]
        X = reshape(x, :, 1)
        k = ones(Float32, km_dims, 1)
        xin = vcat(k, X)
        r_mod = 2.0f0 .* xin
        expected = vcat(k, r_mod)
        Y, _ = hesn(X, ps, st)
        @test size(Y) == (out_dims, 1)
        @test Y ≈ expected
    end

    @testset "readout_activation is honored" begin
        rng = MersenneTwister(5)
        km_dims, in_dims = 1, 3
        res_dims = in_dims + km_dims
        out_dims = res_dims + km_dims
        km = x -> -ones(Float32, km_dims, size(x, 2))
        hesn = HybridESN(km, km_dims, in_dims, res_dims, out_dims, identity;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            leak_coefficient = 1.0,
            state_modifiers = (),
            readout_activation = x -> max.(x, 0.0f0))
        ps, st = setup(rng, hesn)
        ps = _pin_identity_readout(ps; out_dims = out_dims, in_dims = res_dims + km_dims)
        X = reshape(Float32[-2, 0, 5], :, 1)
        k = -ones(Float32, km_dims, 1)
        expected_linear = vcat(k, vcat(k, X))
        Y, _ = hesn(X, ps, st)
        @test Y ≈ max.(expected_linear, 0.0f0)
    end
end
