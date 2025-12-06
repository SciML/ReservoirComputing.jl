@testset "DelayESN" begin
    rng = MersenneTwister(123)

    @testset "constructor wiring & dimensions" begin
        in_dims = 3
        res_dims = 5
        out_dims = 2
        num_delays = 2

        desn = DelayESN(in_dims, res_dims, out_dims;
            num_delays = num_delays, stride = 1)

        @test desn isa ReservoirComputer

        reservoir = desn.reservoir
        @test reservoir isa StatefulLayer

        mods = desn.state_modifiers
        @test mods isa Tuple
        @test !isempty(mods)
        @test first(mods) isa DelayLayer

        dl = first(mods)
        @test Int(dl.in_dims) == res_dims
        @test Int(dl.num_delays) == num_delays
        @test Int(dl.stride) == 1

        ro = desn.readout
        @test ro isa LinearReadout
        @test Int(ro.in_dims) == res_dims * (num_delays + 1)
        @test Int(ro.out_dims) == out_dims
    end

    @testset "setup and forward pass shapes" begin
        in_dims = 4
        res_dims = 10
        out_dims = 3
        num_delays = 1

        desn = DelayESN(in_dims, res_dims, out_dims;
            num_delays = num_delays, stride = 1)

        ps, st = setup(rng, desn)

        x = rand(rng, Float32, in_dims)
        y, st2 = desn(x, ps, st)
        @test size(y) == (out_dims,)

        X = rand(rng, Float32, in_dims, 7)
        Y, st3 = desn(X, ps, st2)
        @test size(Y) == (out_dims, 7)

        @test propertynames(st3) == propertynames(st2)
    end

    @testset "num_delays changes readout input dim" begin
        in_dims = 2
        res_dims = 6
        out_dims = 1

        desn1 = DelayESN(in_dims, res_dims, out_dims; num_delays = 0)
        desn2 = DelayESN(in_dims, res_dims, out_dims; num_delays = 3)

        ro1 = desn1.readout
        ro2 = desn2.readout

        @test Int(ro1.in_dims) == res_dims * (0 + 1)
        @test Int(ro2.in_dims) == res_dims * (3 + 1)
    end
end
