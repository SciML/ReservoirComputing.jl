include("../shared/generic_testsetup.jl")
using .GenericTestSetup

begin
    using Random
    using Serialization
    using Static
    using ReservoirComputing

    function roundtrip(x)
        io = IOBuffer()
        serialize(io, x)
        seekstart(io)
        return deserialize(io)
    end

    @testset "layers and models survive serialization round-trips" begin
        objects = Any[
            LinearReadout(3 => 2; use_bias = true),
            Collect(),
            DelayLayer(3; num_delays = 2),
            NonlinearFeaturesLayer(abs; include_input = true),
            ESNCell(3 => 5),
            ResESNCell(3 => 5; alpha = 0.8, beta = 0.6),
            ES2NCell(3 => 5; proximity = 0.5),
            EuSNCell(3 => 5; diffusion = 0.4),
            EIESNCell(3 => 5; use_bias = false),
            AdditiveEIESNCell(3 => 5; use_bias = true),
            MemoryESNCell((3, 4) => 5),
            MemoryResESNCell((3, 4) => 5),
            RMNCell(MemoryESNCell((3, 4) => 5), ESNCell(3 => 4, identity)),
            LocalInformationFlow(ESNCell, 3 => 5, 2),
            ReservoirChain(identity, Collect(), LinearReadout(3 => 2; include_collect = false)),
            ESN(3, 5, 2),
            ResESN(3, 5, 2),
            ES2N(3, 5, 2),
            EuSN(3, 5, 2),
            EIESN(3, 5, 2),
            AdditiveEIESN(3, 5, 2),
            DeepESN(3, [4, 5], 2),
            HybridESN(identity, 3, 3, 5, 2),
            InputDelayESN(3, 5, 2; num_delays = 1),
            StateDelayESN(3, 5, 2; num_delays = 1),
            DelayESN(3, 5, 2; num_input_delays = 1, num_state_delays = 1),
            LIFESN(3, 5, 2; lookback_horizon = 2),
            NGRC(3, 2; num_delays = 1, features = (abs,), ro_dims = 12),
            RMNESN(3, 4, 5, 2),
            RMNResESN(3, 4, 5, 2),
        ]

        for obj in objects
            rt = roundtrip(obj)
            @test typeof(rt) == typeof(obj)
            @test !isempty(sprint(show, rt))
        end
    end

    @testset "setup and forward are reproducible for fixed RNG seeds" begin
        model = ESN(
            3,
            5,
            2,
            identity;
            init_input = dense_init(Float32; value = 0.2),
            init_reservoir = dense_init(Float32; value = 0),
            init_state = dense_init(Float32; value = 0.1),
            leak_coefficient = 1.0f0,
        )
        rng1 = MersenneTwister(505)
        rng2 = MersenneTwister(505)
        ps1, st1 = setup(rng1, model)
        ps2, st2 = setup(rng2, model)
        @test ps1 == ps2

        x = Float32[0.1, 0.2, 0.3]
        y1, st1_after = model(x, ps1, st1)
        y2, st2_after = model(x, ps2, st2)
        @test y1 == y2
        @test propertynames(st1_after) == propertynames(st2_after)
    end

    @testset "trained readout serialization preserves predictions" begin
        rc = ReservoirChain(identity, Collect(), LinearReadout(3 => 2; include_collect = false))
        rng = MersenneTwister(606)
        ps, st = setup(rng, rc)
        data = reshape(Float32.(1:18), 3, 6) ./ 10
        y1 = data[1, :] .+ data[2, :]
        y2 = data[3, :] .- data[1, :]
        target = vcat(
            reshape(y1, 1, length(y1)),
            reshape(y2, 1, length(y2)),
        )
        ps_trained, st_trained = train!(rc, data, target, ps, st, StandardRidge(Float32, 0.0f0))
        pred, _ = predict(rc, data, ps_trained, st_trained)

        ps_rt = roundtrip(ps_trained)
        st_rt = roundtrip(st_trained)
        pred_rt, _ = predict(rc, data, ps_rt, st_rt)
        @test pred_rt ≈ pred
        @test eltype(pred_rt) === Float32
    end
end
