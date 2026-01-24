using Test
using Random
using ReservoirComputing
using Static
using LinearAlgebra

@testset "InputDelayESN" begin
    rng = MersenneTwister(123)

    @testset "constructor wiring & dimensions" begin
        in_dims = 3
        res_dims = 5
        out_dims = 2
        num_delays = 2

        idesn = InputDelayESN(
            in_dims, res_dims, out_dims;
            num_delays = num_delays, stride = 1
        )

        @test idesn isa InputDelayESN
        @test idesn.input_modifiers isa Tuple
        @test first(idesn.input_modifiers) isa DelayLayer
        @test Int(first(idesn.input_modifiers).in_dims) == in_dims
        @test idesn.reservoir isa StatefulLayer
        @test idesn.reservoir.cell isa ESNCell
        @test Int(idesn.reservoir.cell.in_dims) == in_dims * (num_delays + 1)
        @test Int(idesn.reservoir.cell.out_dims) == res_dims
        @test idesn.states_modifiers isa Tuple
        @test isempty(idesn.states_modifiers)
        @test idesn.readout isa LinearReadout
        @test Int(idesn.readout.in_dims) == res_dims
    end

    @testset "with state modifiers" begin
        in_dims = 3
        res_dims = 5
        out_dims = 2

        idesn = InputDelayESN(
            in_dims, res_dims, out_dims;
            states_modifiers = (NLAT1,)
        )

        @test !isempty(idesn.states_modifiers)
        @test idesn.states_modifiers[1].func === NLAT1
    end

    @testset "num_delays changes reservoir input dim" begin
        in_dims = 2
        res_dims = 6
        out_dims = 1

        idesn1 = InputDelayESN(in_dims, res_dims, out_dims; num_delays = 0)
        idesn2 = InputDelayESN(in_dims, res_dims, out_dims; num_delays = 3)

        @test Int(idesn1.reservoir.cell.in_dims) == in_dims * (0 + 1)
        @test Int(idesn2.reservoir.cell.in_dims) == in_dims * (3 + 1)
    end
end
