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

function _with_identity_readout(ps::NamedTuple; out_dims::Integer, in_dims::Integer)
    ro_ps = haskey(ps.readout, :bias) ?
            (weight = _I32(out_dims, in_dims), bias = _Z32(out_dims)) :
            (weight = _I32(out_dims, in_dims),)
    return merge(ps, (readout = ro_ps,))
end

model_name(::Type{M}) where {M} = string(nameof(M))

mix_kw(::Type{ESN}) = :leak_coefficient
mix_kw(::Type{ES2N}) = :proximity
mix_kw(::Type{EuSN}) = :leak_coefficient

reservoir_param_keys(::Type{ESN}) = (:input_matrix, :reservoir_matrix)
reservoir_param_keys(::Type{ES2N}) = (:input_matrix, :reservoir_matrix, :orthogonal_matrix)
reservoir_param_keys(::Type{EuSN}) = (:input_matrix, :reservoir_matrix)

default_reservoir_kwargs(::Type{ESN}) = NamedTuple()
default_reservoir_kwargs(::Type{ES2N}) = (init_orthogonal = _W_I,)
default_reservoir_kwargs(::Type{EuSN}) = NamedTuple()

function build_model(::Type{M}, in_dims::Int, res_dims::Int, out_dims::Int, activation;
        state_modifiers = (),
        readout_activation = identity,
        mix::Real = 1.0,
        use_bias = False(),
        init_input = _W_I,
        init_reservoir = _W_ZZ,
        init_bias = _O32,
        init_state = init_state3,
        extra::NamedTuple = NamedTuple()
) where {M}
    base = (use_bias = use_bias,
        init_input = init_input,
        init_reservoir = init_reservoir,
        init_bias = init_bias,
        init_state = init_state)

    mixnt = NamedTuple{(mix_kw(M),)}((mix,))
    kw = merge(base, default_reservoir_kwargs(M), mixnt, extra)

    return M(in_dims, res_dims, out_dims, activation;
        state_modifiers = state_modifiers,
        readout_activation = readout_activation,
        kw...)
end

function test_esn_family_contract(::Type{M}) where {M}
    @testset "$(model_name(M)): constructor & parameter/state shapes" begin
        rng = MersenneTwister(42)
        in_dims, res_dims, out_dims = 3, 5, 4

        model = build_model(M, in_dims, res_dims, out_dims, identity;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            mix = 1.0
        )

        ps, st = setup(rng, model)

        @test haskey(ps, :reservoir)
        for k in reservoir_param_keys(M)
            @test haskey(ps.reservoir, k)
        end
        @test !haskey(ps.reservoir, :bias)
        @test size(ps.reservoir.input_matrix) == (res_dims, in_dims)
        @test size(ps.reservoir.reservoir_matrix) == (res_dims, res_dims)
        if M === ES2N
            @test size(ps.reservoir.orthogonal_matrix) == (res_dims, res_dims)
        end

        @test haskey(ps, :readout)
        @test haskey(ps.readout, :weight)
        @test size(ps.readout.weight) == (out_dims, res_dims)

        @test haskey(st, :reservoir)
        @test haskey(st, :states_modifiers)
        @test haskey(st, :readout)
        @test st.states_modifiers isa Tuple
    end

    @testset "$(model_name(M)): forward (vector) with identity pipeline -> y == x (dimensions matched)" begin
        rng = MersenneTwister(0)
        D = 3

        model = build_model(M, D, D, D, identity;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            mix = 1.0
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[1, 2, 3]
        X = reshape(x, :, 1)

        Y, st2 = model(X, ps, st)

        @test size(Y) == (D, 1)
        @test vec(Y) ≈ x
        @test haskey(st2, :reservoir) && haskey(st2, :states_modifiers) &&
              haskey(st2, :readout)
    end

    @testset "$(model_name(M)): forward (batch matrix) with identity pipeline -> Y == X" begin
        rng = MersenneTwister(1)
        D, B = 3, 2

        model = build_model(M, D, D, D, identity;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            mix = 1.0
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        X = Float32[1 2; 3 4; 5 6]
        Y, _ = model(X, ps, st)

        @test size(Y) == (D, B)
        @test Y ≈ X
    end

    @testset "$(model_name(M)): state_modifiers are applied (single modifier doubles features)" begin
        rng = MersenneTwister(2)
        D = 3

        model = build_model(M, D, D, D, identity;
            state_modifiers = (x -> 2.0f0 .* x,),
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            mix = 1.0
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[1, 2, 3]
        y, _ = model(x, ps, st)
        @test y ≈ 2.0f0 .* x
    end

    @testset "$(model_name(M)): multiple state_modifiers apply in order" begin
        rng = MersenneTwister(3)
        D = 3
        mods = (x -> x .+ 1.0f0, x -> 3.0f0 .* x)

        model = build_model(M, D, D, D, identity;
            state_modifiers = mods,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            mix = 1.0
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[0, 1, 2]
        y, _ = model(x, ps, st)
        @test y ≈ 3.0f0 .* (x .+ 1.0f0)
    end

    @testset "$(model_name(M)): outer call computes its own initial hidden state through reservoir cell" begin
        rng = MersenneTwister(123)
        D = 2

        model = build_model(M, D, D, D, identity;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_state = init_state3,
            mix = 1.0
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[7, 9]
        y, st2 = model(x, ps, st)

        @test y ≈ x
        @test haskey(st2, :reservoir)
        @test haskey(st2, :states_modifiers)
        @test haskey(st2, :readout)
    end

    @testset "$(model_name(M)): readout_activation is honored" begin
        rng = MersenneTwister(4)
        D = 3

        model = build_model(M, D, D, D, identity;
            readout_activation = x -> max.(x, 0.0f0),
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            mix = 1.0
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[-1, 0.5, -3]
        y, _ = model(x, ps, st)
        @test y ≈ max.(x, 0.0f0)
    end
end

@testset "ESN-family model contract" begin
    for M in (ESN, ES2N, EuSN)
        test_esn_family_contract(M)
    end
end
