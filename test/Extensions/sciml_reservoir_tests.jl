# SciML reservoir interface contracts.
begin
    using Test
    using Random
    using ReservoirComputing

    const SciMLReservoirError = @static isdefined(Base, :FieldError) ?
        Union{ErrorException, FieldError} : ErrorException

    function sciml_reservoir_fixture()
        reservoir = SciMLProblemReservoir(
            (placeholder = true,),
            TerminalStateSampling(),
            (0.0, 1.0),
        )
        model = ReservoirComputer(reservoir, LinearReadout(1 => 1))
        ps, st = setup(MersenneTwister(0), model)
        return model, randn(Float32, 1, 5), ps, st
    end

    @testset "AbstractSampler hierarchy" begin
        @test TerminalStateSampling <: AbstractSampler
        @test TerminalStateSampling() isa AbstractSampler
    end

    @testset "SciMLProblemReservoir construction" begin
        prob = (placeholder = true,)
        sampler = TerminalStateSampling()
        tspan = (0.0, 1.0)

        res = SciMLProblemReservoir(prob, sampler, tspan)
        @test res isa SciMLProblemReservoir
        @test res isa AbstractSciMLProblemReservoir
        @test res.prob === prob
        @test res.sampler === sampler
        @test res.tspan === tspan
        @test isempty(res.args)
        @test isempty(res.kwargs)

        res_with_args =
            SciMLProblemReservoir(prob, sampler, tspan, :solver_arg; reltol = 1.0e-3)
        @test res_with_args.args == (:solver_arg,)
        @test res_with_args.kwargs[:reltol] == 1.0e-3
    end

    @testset "SciMLProblemReservoir as Lux layer" begin
        prob = (placeholder = true,)
        res = SciMLProblemReservoir(prob, TerminalStateSampling(), (0.0, 1.0))
        rng = MersenneTwister(0)
        ps = initialparameters(rng, res)
        st = initialstates(rng, res)
        @test ps == NamedTuple()
        @test st == NamedTuple()
    end

    @testset "Continuous _collectstates errors without extension" begin
        model, data, ps, st = sciml_reservoir_fixture()
        @test_throws SciMLReservoirError collectstates(model, data, ps, st)
    end

    @testset "SciMLProblemReservoir rejects protected solve kwargs" begin
        prob = (placeholder = true,)
        sampler = TerminalStateSampling()
        tspan = (0.0, 1.0)
        for badkw in (:saveat, :save_everystep, :dense)
            @test_throws ArgumentError SciMLProblemReservoir(
                prob,
                sampler,
                tspan;
                (badkw => true,)...,
            )
        end
        # User kwargs that do not collide should still go through.
        res_ok = SciMLProblemReservoir(prob, sampler, tspan; reltol = 1.0e-6)
        @test res_ok.kwargs[:reltol] == 1.0e-6
    end

    @testset "Continuous _predict errors without extension" begin
        model, data, ps, st = sciml_reservoir_fixture()
        initialdata = randn(Float32, 1)
        @test_throws SciMLReservoirError predict(model, data, ps, st)
        @test_throws SciMLReservoirError predict(model, 3, ps, st; initialdata)
    end

    # DeepESN has `:cells`, not `:reservoir`; keep predict dispatch field-agnostic.
    @testset "predict works on reservoir computers without a :reservoir field" begin
        rng = MersenneTwister(0)
        desn = DeepESN(3, [16, 16], 3)
        ps, st = setup(rng, desn)
        data = randn(3, 5)
        out, _ = predict(desn, data, ps, st)
        @test size(out) == (3, 5)
        @test all(isfinite, out)
        out_ar, _ = predict(desn, 3, ps, st; initialdata = randn(3))
        @test size(out_ar) == (3, 3)
        @test all(isfinite, out_ar)
    end

end
