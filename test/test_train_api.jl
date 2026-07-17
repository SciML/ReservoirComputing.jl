using Test
using Random
using LinearAlgebra
using ReservoirComputing
using LinearSolve

@testset "train model-level smoke" begin
    rng = MersenneTwister(42)
    in_dims, res_dims, out_dims = 3, 12, 2
    n_steps = 35
    train_data = randn(rng, Float32, in_dims, n_steps)
    target_data = randn(rng, Float32, out_dims, n_steps)

    model = ESN(in_dims, res_dims, out_dims)
    ps, st = setup(rng, model)

    ps_new, st_new = train(
        model, train_data, target_data, ps, st;
        objective = RidgeRegression(1.0e-3),
    )

    @test size(ps_new.readout.weight) == (out_dims, res_dims)
    @test all(isfinite, ps_new.readout.weight)
end

@testset "train model-level: objective and solver kwargs" begin
    rng = MersenneTwister(7)
    in_dims, res_dims, out_dims = 3, 10, 2
    n_steps = 30
    train_data = randn(rng, Float32, in_dims, n_steps)
    target_data = randn(rng, Float32, out_dims, n_steps)

    model = ESN(in_dims, res_dims, out_dims)
    ps, st = setup(rng, model)
    regularization = 1.0e-3

    ps_default, _ = train(
        model, train_data, target_data, ps, st;
        objective = RidgeRegression(regularization),
    )
    ps_ls, _ = train(
        model, train_data, target_data, ps, st;
        objective = RidgeRegression(regularization),
        solver = QRFactorization(),
    )
    ps_legacy, _ = train(
        model, train_data, target_data, ps, st;
        objective = RidgeRegression(regularization),
        solver = QRSolver(),
    )

    @test ps_default.readout.weight == ps_ls.readout.weight
    @test size(ps_ls.readout.weight) == (out_dims, res_dims)
    @test ps_default.readout.weight ≈ ps_legacy.readout.weight rtol = 1.0e-3
end

@testset "train model-level: washout and return_states" begin
    rng = MersenneTwister(11)
    in_dims, res_dims, out_dims = 3, 12, 2
    n_steps = 25
    washout = 4
    train_data = randn(rng, Float32, in_dims, n_steps)
    target_data = randn(rng, Float32, out_dims, n_steps)

    model = ESN(in_dims, res_dims, out_dims)
    ps, st = setup(rng, model)

    (ps_trained, st_trained), states = train(
        model, train_data, target_data, ps, st;
        objective = RidgeRegression(1.0e-4),
        washout = washout,
        return_states = true,
    )

    @test size(states) == (res_dims, n_steps - washout)
    @test size(ps_trained.readout.weight) == (out_dims, res_dims)
    @test all(isfinite, ps_trained.readout.weight)
end

@testset "train feature-level: solver nothing equals QRFactorization" begin
    rng = MersenneTwister(23)
    n_features, n_samples, n_outputs = 5, 40, 2
    states = randn(rng, Float64, n_features, n_samples)
    targets = randn(rng, Float64, n_outputs, n_samples)
    regularization = 1.0e-2

    weights_default = train(RidgeRegression(regularization), states, targets)
    weights_nothing = train(
        RidgeRegression(regularization), states, targets; solver = nothing
    )
    weights_ls = train(
        RidgeRegression(regularization), states, targets; solver = QRFactorization()
    )

    @test weights_default == weights_nothing
    @test weights_default == weights_ls
    @test size(weights_default) == (n_outputs, n_features)
end
