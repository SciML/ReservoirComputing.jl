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

        reservoir = idesn.reservoir
        @test reservoir isa StatefulLayer
        @test reservoir.cell isa ESNCell

        mods = idesn.input_modifiers
        @test mods isa Tuple
        @test !isempty(mods)
        @test first(mods) isa DelayLayer

        dl = first(mods)
        @test Int(dl.in_dims) == in_dims
        @test Int(dl.num_delays) == num_delays
        @test Int(dl.stride) == 1

        @test Int(reservoir.cell.in_dims) == in_dims * (num_delays + 1)
        @test Int(reservoir.cell.out_dims) == res_dims

        ro = idesn.readout
        @test ro isa LinearReadout
        @test Int(ro.in_dims) == res_dims
        @test Int(ro.out_dims) == out_dims
    end

    @testset "num_delays changes reservoir input dim" begin
        in_dims = 2
        res_dims = 6
        out_dims = 1

        idesn1 = InputDelayESN(in_dims, res_dims, out_dims; num_delays = 0)
        idesn2 = InputDelayESN(in_dims, res_dims, out_dims; num_delays = 3)

        cell1 = idesn1.reservoir.cell
        cell2 = idesn2.reservoir.cell

        @test Int(cell1.in_dims) == in_dims * (0 + 1)
        @test Int(cell2.in_dims) == in_dims * (3 + 1)
    end
end
