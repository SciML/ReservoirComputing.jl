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

    res_with_args = SciMLProblemReservoir(prob, sampler, tspan, :solver_arg; reltol = 1e-3)
    @test res_with_args.args == (:solver_arg,)
    @test res_with_args.kwargs[:reltol] == 1e-3
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
