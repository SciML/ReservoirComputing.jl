using Test
using Random
using ReservoirComputing
using LinearAlgebra
using Static

const _I32_lf = (m, n) -> Matrix{Float32}(LinearAlgebra.I, m, n)
const _Z32_lf = m -> zeros(Float32, m)
const _O32_lf = (rng, m) -> zeros(Float32, m)
const _W_I_lf = (rng, m, n) -> _I32_lf(m, n)
const _W_ZZ_lf = (rng, m, n) -> zeros(Float32, m, n)

function init_state3_lf(rng::AbstractRNG, m::Integer, B::Integer)
    return B == 1 ? zeros(Float32, m) : zeros(Float32, m, B)
end

function _with_identity_readout_lf(ps::NamedTuple; out_dims::Integer, in_dims::Integer)
    ro_ps = haskey(ps.readout, :bias) ?
        (weight = _I32_lf(out_dims, in_dims), bias = _Z32_lf(out_dims)) :
        (weight = _I32_lf(out_dims, in_dims),)
    return merge(ps, (readout = ro_ps,))
end

@testset "LIFESNCell: convenience constructor" begin
    cell = LIFESNCell(3 => 5, identity; lookback_horizon = 3,
        init_input = _W_I_lf, init_reservoir = _W_ZZ_lf, use_bias = false)
    @test cell isa LocalInformationFlow
    @test cell.lookback_horizon == 3

    cell2 = LIFESNCell(3 => 5, tanh)
    @test cell2 isa LocalInformationFlow
    @test cell2.lookback_horizon == 2
end

@testset "LIFESN: constructor & parameter/state shapes" begin
    rng = MersenneTwister(42)
    in_dims, res_dims, out_dims = 3, 5, 4

    model = LIFESN(
        in_dims, res_dims, out_dims, identity;
        lookback_horizon = 3,
        use_bias = False(),
        init_input = _W_I_lf,
        init_reservoir = _W_ZZ_lf,
        init_bias = _O32_lf,
        init_state = init_state3_lf
    )

    ps, st = setup(rng, model)

    @test haskey(ps, :reservoir)
    @test haskey(ps, :readout)
    @test haskey(ps.readout, :weight)
    @test size(ps.readout.weight) == (out_dims, res_dims)

    @test haskey(st, :reservoir)
    @test haskey(st, :states_modifiers)
    @test haskey(st, :readout)
    @test st.states_modifiers isa Tuple
end

@testset "LIFESN: forward (vector) with identity pipeline" begin
    rng = MersenneTwister(0)
    D = 3

    model = LIFESN(
        D, D, D, identity;
        lookback_horizon = 2,
        use_bias = False(),
        init_input = _W_I_lf,
        init_reservoir = _W_ZZ_lf,
        init_bias = _O32_lf,
        init_state = init_state3_lf
    )

    ps, st = setup(rng, model)
    ps = _with_identity_readout_lf(ps; out_dims = D, in_dims = D)

    x = Float32[1, 2, 3]
    X = reshape(x, :, 1)

    Y, st2 = model(X, ps, st)

    @test size(Y) == (D, 1)
    @test vec(Y) ≈ x
    @test haskey(st2, :reservoir) && haskey(st2, :states_modifiers) &&
        haskey(st2, :readout)
end

@testset "LIFESN: forward (batch matrix) with identity pipeline" begin
    rng = MersenneTwister(1)
    D, B = 3, 2

    model = LIFESN(
        D, D, D, identity;
        lookback_horizon = 2,
        use_bias = False(),
        init_input = _W_I_lf,
        init_reservoir = _W_ZZ_lf,
        init_bias = _O32_lf,
        init_state = init_state3_lf
    )

    ps, st = setup(rng, model)
    ps = _with_identity_readout_lf(ps; out_dims = D, in_dims = D)

    X = Float32[1 2; 3 4; 5 6]
    Y, _ = model(X, ps, st)

    @test size(Y) == (D, B)
    @test Y ≈ X
end

@testset "LIFESN: state_modifiers are applied" begin
    rng = MersenneTwister(2)
    D = 3

    model = LIFESN(
        D, D, D, identity;
        lookback_horizon = 2,
        state_modifiers = (x -> 2.0f0 .* x,),
        use_bias = False(),
        init_input = _W_I_lf,
        init_reservoir = _W_ZZ_lf,
        init_bias = _O32_lf,
        init_state = init_state3_lf
    )

    ps, st = setup(rng, model)
    ps = _with_identity_readout_lf(ps; out_dims = D, in_dims = D)

    x = Float32[1, 2, 3]
    y, _ = model(x, ps, st)
    @test y ≈ 2.0f0 .* x
end

@testset "LIFESN: multiple state_modifiers apply in order" begin
    rng = MersenneTwister(3)
    D = 3
    mods = (x -> x .+ 1.0f0, x -> 3.0f0 .* x)

    model = LIFESN(
        D, D, D, identity;
        lookback_horizon = 2,
        state_modifiers = mods,
        use_bias = False(),
        init_input = _W_I_lf,
        init_reservoir = _W_ZZ_lf,
        init_bias = _O32_lf,
        init_state = init_state3_lf
    )

    ps, st = setup(rng, model)
    ps = _with_identity_readout_lf(ps; out_dims = D, in_dims = D)

    x = Float32[0, 1, 2]
    y, _ = model(x, ps, st)
    @test y ≈ 3.0f0 .* (x .+ 1.0f0)
end

@testset "LIFESN: outer call computes initial hidden state through reservoir cell" begin
    rng = MersenneTwister(123)
    D = 2

    model = LIFESN(
        D, D, D, identity;
        lookback_horizon = 2,
        use_bias = False(),
        init_input = _W_I_lf,
        init_reservoir = _W_ZZ_lf,
        init_state = init_state3_lf
    )

    ps, st = setup(rng, model)
    ps = _with_identity_readout_lf(ps; out_dims = D, in_dims = D)

    x = Float32[7, 9]
    y, st2 = model(x, ps, st)

    @test y ≈ x
    @test haskey(st2, :reservoir)
    @test haskey(st2, :states_modifiers)
    @test haskey(st2, :readout)
end

@testset "LIFESN: readout_activation is honored" begin
    rng = MersenneTwister(4)
    D = 3

    model = LIFESN(
        D, D, D, identity;
        lookback_horizon = 2,
        readout_activation = x -> max.(x, 0.0f0),
        use_bias = False(),
        init_input = _W_I_lf,
        init_reservoir = _W_ZZ_lf,
        init_bias = _O32_lf,
        init_state = init_state3_lf
    )

    ps, st = setup(rng, model)
    ps = _with_identity_readout_lf(ps; out_dims = D, in_dims = D)

    x = Float32[-1, 0.5, -3]
    y, _ = model(x, ps, st)
    @test y ≈ max.(x, 0.0f0)
end

@testset "LIFESN: different lookback horizons" begin
    rng = MersenneTwister(10)
    in_dims, res_dims, out_dims = 2, 10, 2

    for H in [1, 2, 4]
        model = LIFESN(in_dims, res_dims, out_dims, tanh; lookback_horizon = H)
        ps, st = setup(rng, model)

        x = randn(rng, Float32, in_dims)
        y, st2 = model(x, ps, st)
        @test length(y) == out_dims
    end
end

@testset "LIFESN: show" begin
    model = LIFESN(2, 4, 1, tanh; lookback_horizon = 3)
    s = sprint(show, model)
    @test occursin("LIFESN(", s)
    @test occursin("LinearReadout", s)
end

@testset "LIFESN: train! with StandardRidge" begin
    rng = MersenneTwister(99)
    in_dims, res_dims, out_dims = 1, 30, 1
    T = 200

    train_data = randn(rng, Float64, in_dims, T)
    target_data = sin.(cumsum(train_data, dims = 2))

    model = LIFESN(in_dims, res_dims, out_dims, tanh; lookback_horizon = 3)
    ps, st = setup(rng, model)

    ps, st = train!(model, train_data, target_data, ps, st, StandardRidge())

    @test haskey(ps.readout, :weight)

    pred, _ = model(train_data[:, end], ps, st)
    @test length(pred) == out_dims
end
