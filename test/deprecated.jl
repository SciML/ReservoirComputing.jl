using Test
using Random
using ReservoirComputing
using LuxCore: setup
using LinearSolve: SVDFactorization

# Delete with src/deprecated.jl at v1.0.

# `@test_deprecated` returns the expression value under `--depwarn=yes`/`no`, but under
# `--depwarn=error` it becomes `@test_throws ErrorException` and returns a `Test.Pass`.
# Never unpack its return value. Assert deprecation separately, then obtain values via a
# second call to the deprecated API when that call can complete (depwarn ≠ error), or via
# the delegated non-deprecated implementation when depwarn=error (call throws).
const _DEPWARN_ERROR = Base.JLOptions().depwarn == Int8(2)

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
    @test_deprecated train!(model, train_data, target_data, ps, st, ridge)
    ps_pos, _ = if _DEPWARN_ERROR
        train(model, train_data, target_data, ps, st; objective = ridge)
    else
        train!(model, train_data, target_data, ps, st, ridge)
    end
    @test ps_pos.readout.weight ≈ ps_kw.readout.weight
end

@testset "train(objective, states, target_data) is deprecated and matches fitting" begin
    rng = MersenneTwister(53)
    states = randn(rng, Float64, 5, 30)
    targets = randn(rng, Float64, 2, 30)
    ridge = RidgeRegression(1.0e-3)

    weights_new = ReservoirComputing.__fit_readout(
        ridge, states, targets; solver = ReservoirComputing.QRFactorization()
    )
    @test_deprecated train(
        ridge, states, targets; solver = ReservoirComputing.QRFactorization()
    )
    result = if _DEPWARN_ERROR
        ReservoirComputing.__fit_readout(
            ridge, states, targets; solver = ReservoirComputing.QRFactorization()
        )
    else
        train(
            ridge, states, targets; solver = ReservoirComputing.QRFactorization()
        )
    end
    @test result ≈ weights_new
end

@testset "toepliz_init is deprecated alias of toeplitz_init" begin
    @test toepliz_init === toeplitz_init
    @test toepliz_init(MersenneTwister(7), 5, 5) == toeplitz_init(MersenneTwister(7), 5, 5)
end

@testset "train! still accepts solver kwarg" begin
    rng = MersenneTwister(37)
    in_dims, res_dims, out_dims = 3, 10, 2
    n_steps = 28
    train_data = randn(rng, Float32, in_dims, n_steps)
    target_data = randn(rng, Float32, out_dims, n_steps)

    model = ESN(in_dims, res_dims, out_dims)
    ps, st = setup(rng, model)

    @test_deprecated train!(
        model, train_data, target_data, ps, st, RidgeRegression(1.0e-3);
        solver = SVDFactorization(),
    )
    ps_trained, _ = if _DEPWARN_ERROR
        train(
            model, train_data, target_data, ps, st;
            objective = RidgeRegression(1.0e-3), solver = SVDFactorization(),
        )
    else
        train!(
            model, train_data, target_data, ps, st, RidgeRegression(1.0e-3);
            solver = SVDFactorization(),
        )
    end
    @test size(ps_trained.readout.weight) == (out_dims, res_dims)
    @test all(isfinite, ps_trained.readout.weight)
end
