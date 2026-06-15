using ReservoirComputing
using Random
using Test
using LuxCore
using LinearAlgebra

const res_size = 20
const in_size = 3
const out_size = 3
const train_len = 50

@testset "EIESN Integration Test" begin
    rng = MersenneTwister(1)

    model = EIESN(in_size, res_size, out_size)
    @test model isa ReservoirComputing.AbstractReservoirComputer
    @test model isa ReservoirComputing.AbstractEchoStateNetwork

    io = IOBuffer()
    show(io, model)
    shown = String(take!(io))
    @test occursin("EIESN", shown)

    ps = initialparameters(rng, model)
    st = initialstates(rng, model)

    @test haskey(ps, :reservoir)
    @test haskey(ps.reservoir, :input_matrix)
    @test haskey(ps.reservoir, :reservoir_matrix)
    @test haskey(ps.reservoir, :bias_ex)
    @test haskey(ps.reservoir, :bias_inh)

    input_data = rand(Float32, in_size, train_len)

    (output, new_st) = model(input_data, ps, st)
    @test size(output) == (out_size, train_len)

    cell = model.reservoir
    (states, _) = cell(input_data, ps.reservoir, st.reservoir)
    @test !all(iszero, states)
    @test size(states) == (res_size, train_len)

    model_custom = EIESN(
        in_size, res_size, out_size,
        (tanh, identity);
        use_bias = false
    )
    ps_custom = initialparameters(rng, model_custom)
    st_custom = initialstates(rng, model_custom)

    @test !haskey(ps_custom.reservoir, :bias_ex)
    @test !haskey(ps_custom.reservoir, :bias_inh)

    (out_custom, _) = model_custom(input_data, ps_custom, st_custom)
    @test size(out_custom) == (out_size, train_len)
end
