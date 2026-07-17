using Test
using Random
using ReservoirComputing
using MLJLinearModels
using LIBSVM

# Model-level `train` with optional training backends (#473).

@testset "train model-level: MLJ ridge" begin
    rng = MersenneTwister(11)
    in_dims, res_dims, out_dims = 3, 12, 2
    n_steps = 40
    train_data = randn(rng, Float64, in_dims, n_steps)
    target_data = randn(rng, Float64, out_dims, n_steps)

    model = ESN(in_dims, res_dims, out_dims)
    ps, st = setup(rng, model)

    regressor = MLJLinearModels.RidgeRegression(1.0e-3; fit_intercept = false)
    ps_trained, st_trained = train(
        model, train_data, target_data, ps, st;
        objective = regressor,
    )

    @test haskey(ps_trained.readout, :weight)
    @test size(ps_trained.readout.weight) == (out_dims, res_dims)
    @test all(isfinite, ps_trained.readout.weight)

    prediction, _ = model(train_data[:, end], ps_trained, st_trained)
    @test size(prediction) == (out_dims, 1)
    @test all(isfinite, prediction)
end

@testset "train model-level: MLJ fit_intercept=true rejected" begin
    rng = MersenneTwister(13)
    in_dims, res_dims, out_dims = 2, 12, 1
    n_steps = 30
    train_data = randn(rng, Float64, in_dims, n_steps)
    target_data = randn(rng, Float64, out_dims, n_steps)

    model = ESN(in_dims, res_dims, out_dims)
    ps, st = setup(rng, model)

    regressor = MLJLinearModels.RidgeRegression(1.0e-2; fit_intercept = true)
    @test_throws ArgumentError train(
        model, train_data, target_data, ps, st;
        objective = regressor,
    )
end

@testset "train model-level: MLJ washout and return_states" begin
    rng = MersenneTwister(17)
    in_dims, res_dims, out_dims = 3, 12, 2
    n_steps = 35
    washout = 5
    train_data = randn(rng, Float64, in_dims, n_steps)
    target_data = randn(rng, Float64, out_dims, n_steps)

    model = ESN(in_dims, res_dims, out_dims)
    ps, st = setup(rng, model)

    regressor = LassoRegression(1.0e-2; fit_intercept = false)
    (ps_trained, _), states = train(
        model, train_data, target_data, ps, st;
        objective = regressor,
        washout = washout,
        return_states = true,
    )

    @test size(states) == (res_dims, n_steps - washout)
    @test size(ps_trained.readout.weight) == (out_dims, res_dims)
    @test all(isfinite, ps_trained.readout.weight)
end

@testset "train model-level: MLJ solver kwarg reaches fit" begin
    rng = MersenneTwister(19)
    in_dims, res_dims, out_dims = 2, 12, 1
    n_steps = 40
    train_data = randn(rng, Float64, in_dims, n_steps)
    target_data = randn(rng, Float64, out_dims, n_steps)

    model = ESN(in_dims, res_dims, out_dims)
    ps, st = setup(rng, model)

    regressor = LassoRegression(1.0e-2; fit_intercept = false)
    ps_trained, _ = train(
        model, train_data, target_data, ps, st;
        objective = regressor,
        solver = ProxGrad(),
    )

    @test size(ps_trained.readout.weight) == (out_dims, res_dims)
    @test all(isfinite, ps_trained.readout.weight)
end

@testset "train model-level: LIBSVM single-output SVESM" begin
    rng = MersenneTwister(23)
    in_dims, res_dims, out_dims = 1, 20, 1
    n_steps = 80
    train_data = randn(rng, Float64, in_dims, n_steps)
    target_data = sin.(cumsum(train_data; dims = 2))

    model = SVESM(in_dims, res_dims, out_dims, tanh)
    ps, st = setup(rng, model)

    svr = EpsilonSVR(cost = 10.0, epsilon = 0.01)
    ps_trained, st_trained = train(
        model, train_data, target_data, ps, st;
        objective = svr,
    )

    @test haskey(ps_trained.readout, :models)
    prediction, _ = model(train_data[:, end], ps_trained, st_trained)
    @test length(prediction) == out_dims
    @test all(isfinite, prediction)
end

@testset "train model-level: LIBSVM multi-output and washout" begin
    rng = MersenneTwister(29)
    in_dims, res_dims, out_dims = 2, 16, 2
    n_steps = 60
    washout = 4
    train_data = randn(rng, Float64, in_dims, n_steps)
    target_data = vcat(sum(train_data; dims = 1), prod(train_data; dims = 1))

    model = SVESM(in_dims, res_dims, out_dims, tanh)
    ps, st = setup(rng, model)

    svr = NuSVR()
    (ps_trained, _), states = train(
        model, train_data, target_data, ps, st;
        objective = svr,
        washout = washout,
        return_states = true,
    )

    @test size(states) == (res_dims, n_steps - washout)
    @test haskey(ps_trained.readout, :models)
    @test ps_trained.readout.models isa AbstractVector
    @test length(ps_trained.readout.models) == out_dims
end

@testset "train model-level: LIBSVM rejects LinearSolve solver kwarg" begin
    rng = MersenneTwister(31)
    in_dims, res_dims, out_dims = 1, 16, 1
    n_steps = 50
    train_data = randn(rng, Float64, in_dims, n_steps)
    target_data = randn(rng, Float64, out_dims, n_steps)

    model = SVESM(in_dims, res_dims, out_dims, tanh)
    ps, st = setup(rng, model)

    svr = EpsilonSVR()
    @test_throws ArgumentError train(
        model, train_data, target_data, ps, st;
        objective = svr,
        solver = QRFactorization(),
    )
end

@testset "train model-level: LIBSVM solver=nothing is fine" begin
    rng = MersenneTwister(37)
    in_dims, res_dims, out_dims = 1, 16, 1
    n_steps = 50
    train_data = randn(rng, Float64, in_dims, n_steps)
    target_data = randn(rng, Float64, out_dims, n_steps)

    model = SVESM(in_dims, res_dims, out_dims, tanh)
    ps, st = setup(rng, model)

    svr = EpsilonSVR()
    ps_trained, _ = train(
        model, train_data, target_data, ps, st;
        objective = svr,
        solver = nothing,
    )
    @test haskey(ps_trained.readout, :models)
end
