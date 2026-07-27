using Test
using Random
using ReservoirComputing
using LinearSolve

# Delete with src/deprecated.jl at v1.0.

@testset "StandardRidge is deprecated alias of RidgeRegression" begin
    @test StandardRidge === RidgeRegression
    @test StandardRidge(1.0e-3) isa RidgeRegression
    @test StandardRidge(1.0e-3).reg == 1.0e-3
    @test StandardRidge().reg == 0.0
    @test StandardRidge(Float32, 1.0f-2).reg isa Float32
end

@testset "train! is deprecated and matches train" begin
    rng = MersenneTwister(31)
    in_dims, res_dims, out_dims = 2, 12, 1
    n_steps = 20
    train_data = randn(rng, Float32, in_dims, n_steps)
    target_data = randn(rng, Float32, out_dims, n_steps)

    model = ESN(in_dims, res_dims, out_dims)
    ps, st = setup(rng, model)
    ridge = RidgeRegression(2.0e-3)

    ps_kw, _ = train(
        model, train_data, target_data, ps, st; objective = ridge
    )
    result = @test_deprecated train!(model, train_data, target_data, ps, st, ridge)
    ps_pos = result[1]
    @test ps_pos.readout.weight ≈ ps_kw.readout.weight
end

@testset "train! still accepts solver kwarg" begin
    rng = MersenneTwister(37)
    in_dims, res_dims, out_dims = 3, 10, 2
    n_steps = 28
    train_data = randn(rng, Float32, in_dims, n_steps)
    target_data = randn(rng, Float32, out_dims, n_steps)

    model = ESN(in_dims, res_dims, out_dims)
    ps, st = setup(rng, model)

    result = @test_deprecated train!(
        model, train_data, target_data, ps, st, RidgeRegression(1.0e-3);
        solver = SVDFactorization(),
    )
    ps_trained = result[1]
    @test size(ps_trained.readout.weight) == (out_dims, res_dims)
    @test all(isfinite, ps_trained.readout.weight)
end
