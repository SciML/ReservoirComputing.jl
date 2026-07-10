# SciML reservoir interface contracts.
begin
    using Test
    using Random
    using ReservoirComputing

    @testset "AbstractSampler hierarchy" begin
        @test TerminalStateSampling <: AbstractSampler
        @test TerminalStateSampling() isa AbstractSampler
    end

    @testset "SciMLProblemReservoir construction" begin
        # Use a placeholder `prob` — PR1 keeps the type fully untyped, so any value works.
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
        prob = (placeholder = true,)
        res = SciMLProblemReservoir(prob, TerminalStateSampling(), (0.0, 1.0))
        rc = ReservoirComputer(res, LinearReadout(1 => 1))
        rng = MersenneTwister(0)
        ps, st = setup(rng, rc)
        data = randn(Float32, 1, 5)
        @test_throws ErrorException collectstates(rc, data, ps, st)
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
        prob = (placeholder = true,)
        res = SciMLProblemReservoir(prob, TerminalStateSampling(), (0.0, 1.0))
        rc = ReservoirComputer(res, LinearReadout(1 => 1))
        rng = MersenneTwister(0)
        ps, st = setup(rng, rc)
        data = randn(Float32, 1, 5)
        @test_throws ErrorException predict(rc, data, ps, st)
        @test_throws ErrorException predict(rc, 3, ps, st; initialdata = randn(Float32, 1))
    end

    # `DeepESN` is an `AbstractReservoirComputer` subtype whose leading field is
    # `:cells`, not `:reservoir`. The new two-level `predict` dispatch must not
    # unconditionally reach for `rc.reservoir`, or DeepESN crashes with a
    # `FieldError` (originally surfaced by the docs `@example` block in
    # `tutorials/deep_esn.md`). This testset locks in the `hasfield` guard.
    @testset "predict works on reservoir computers without a :reservoir field" begin
        rng = MersenneTwister(0)
        # `rand_sparse`'s sparsity defaults need a wide-enough reservoir for the
        # spectral-radius rescaling to avoid degenerate NaNs.
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
