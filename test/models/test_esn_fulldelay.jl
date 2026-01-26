using Test
using Random
using ReservoirComputing
using Static
using LinearAlgebra

@testset "FullDelayESN" begin
    rng = MersenneTwister(123)

    @testset "constructor wiring & dimensions" begin
        in_dims = 3
        res_dims = 5
        out_dims = 2
        num_input_delays = 2
        num_state_delays = 3

        fesn = FullDelayESN(
            in_dims, res_dims, out_dims;
            num_input_delays = num_input_delays,
            num_state_delays = num_state_delays,
            input_stride = 1,
            state_stride = 1
        )

        @test fesn isa FullDelayESN

        @test fesn.input_delay isa DelayLayer
        @test Int(fesn.input_delay.in_dims) == in_dims
        @test Int(fesn.input_delay.num_delays) == num_input_delays

        @test fesn.reservoir isa StatefulLayer
        @test fesn.reservoir.cell isa ESNCell
        @test Int(fesn.reservoir.cell.in_dims) == in_dims * (num_input_delays + 1)
        @test Int(fesn.reservoir.cell.out_dims) == res_dims

        @test fesn.state_delay isa DelayLayer
        @test Int(fesn.state_delay.in_dims) == res_dims
        @test Int(fesn.state_delay.num_delays) == num_state_delays

        @test fesn.states_modifiers isa Tuple
        @test isempty(fesn.states_modifiers)

        @test fesn.readout isa LinearReadout
        @test Int(fesn.readout.in_dims) == res_dims * (num_state_delays + 1)
        @test Int(fesn.readout.out_dims) == out_dims
    end

    @testset "with state modifiers" begin
        in_dims = 3
        res_dims = 5
        out_dims = 2

        fesn = FullDelayESN(
            in_dims, res_dims, out_dims;
            states_modifiers = (NLAT1,)
        )

        @test !isempty(fesn.states_modifiers)
        @test fesn.states_modifiers[1].func === NLAT1
    end

    @testset "delays change dimensions independently" begin
        in_dims = 2
        res_dims = 6
        out_dims = 1

        fesn1 = FullDelayESN(
            in_dims, res_dims, out_dims;
            num_input_delays = 0, num_state_delays = 0
        )

        fesn2 = FullDelayESN(
            in_dims, res_dims, out_dims;
            num_input_delays = 3, num_state_delays = 2
        )

        @test Int(fesn1.reservoir.cell.in_dims) == in_dims * (0 + 1)
        @test Int(fesn2.reservoir.cell.in_dims) == in_dims * (3 + 1)

        @test Int(fesn1.readout.in_dims) == res_dims * (0 + 1)
        @test Int(fesn2.readout.in_dims) == res_dims * (2 + 1)
    end
end
