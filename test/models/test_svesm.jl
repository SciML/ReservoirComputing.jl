using Test
using Random
using ReservoirComputing
using LinearAlgebra
using LIBSVM
using Static

const _I32_sv = (m, n) -> Matrix{Float32}(LinearAlgebra.I, m, n)
const _O32_sv = (rng, m) -> zeros(Float32, m)
const _W_I_sv = (rng, m, n) -> _I32_sv(m, n)
const _W_ZZ_sv = (rng, m, n) -> zeros(Float32, m, n)

function init_state3_sv(rng::AbstractRNG, m::Integer, B::Integer)
    return B == 1 ? zeros(Float32, m) : zeros(Float32, m, B)
end

@testset "SVESM: constructor & parameter/state shapes" begin
    rng = MersenneTwister(42)
    in_dims, res_dims, out_dims = 3, 5, 4

    model = SVESM(
        in_dims, res_dims, out_dims, identity;
        use_bias = False(),
        init_input = _W_I_sv,
        init_reservoir = _W_ZZ_sv,
        init_bias = _O32_sv,
        init_state = init_state3_sv
    )

    ps, st = setup(rng, model)

    @test haskey(ps, :reservoir)
    @test haskey(ps.reservoir, :input_matrix)
    @test haskey(ps.reservoir, :reservoir_matrix)
    @test size(ps.reservoir.input_matrix) == (res_dims, in_dims)
    @test size(ps.reservoir.reservoir_matrix) == (res_dims, res_dims)

    @test haskey(ps, :readout)
    @test ps.readout == NamedTuple()

    @test haskey(st, :reservoir)
    @test haskey(st, :states_modifiers)
    @test haskey(st, :readout)
    @test st.states_modifiers isa Tuple

    @test model.readout isa SVMReadout
    @test model.readout.in_dims == res_dims
    @test model.readout.out_dims == out_dims
end

@testset "SVESM: show" begin
    model = SVESM(2, 4, 1, tanh)
    s = sprint(show, model)
    @test occursin("SVESM(", s)
    @test occursin("SVMReadout", s)
end

@testset "SVESM: train! with EpsilonSVR" begin
    rng = MersenneTwister(99)
    in_dims, res_dims, out_dims = 1, 30, 1
    T = 200

    train_data = randn(rng, Float64, in_dims, T)
    target_data = sin.(cumsum(train_data, dims = 2))

    model = SVESM(in_dims, res_dims, out_dims, tanh)
    ps, st = setup(rng, model)

    svr = LIBSVM.EpsilonSVR(cost = 10.0, epsilon = 0.01)
    ps, st = train!(model, train_data, target_data, ps, st, svr)

    @test haskey(ps.readout, :models)

    pred, _ = model(train_data[:, end], ps, st)
    @test length(pred) == out_dims
end

@testset "SVESM: train! with NuSVR and state_modifiers" begin
    rng = MersenneTwister(77)
    in_dims, res_dims, out_dims = 2, 20, 2
    T = 100

    train_data = randn(rng, Float64, in_dims, T)
    target_data = vcat(
        sum(train_data, dims = 1),
        prod(train_data, dims = 1)
    )

    model = SVESM(
        in_dims, res_dims, out_dims, tanh;
        state_modifiers = (Pad(1.0),)
    )
    ps, st = setup(rng, model)

    svr = LIBSVM.NuSVR()
    ps, st = train!(model, train_data, target_data, ps, st, svr)

    @test haskey(ps.readout, :models)
    @test ps.readout.models isa AbstractVector
    @test length(ps.readout.models) == out_dims

    pred, _ = model(train_data[:, end], ps, st)
    @test length(pred) == out_dims
end
