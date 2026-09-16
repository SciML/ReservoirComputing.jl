using Test
using Random
using LinearAlgebra
using ReservoirComputing
using LuxCore: setup

@testset "addreadout! validates parameter structure" begin
    rng = MersenneTwister(901)

    model = ESN(2, 4, 1)
    invalid_ps = (; reservoir = NamedTuple(), state_modifiers = ())
    @test_throws ArgumentError ReservoirComputing.addreadout!(
        model, zeros(Float32, 1, 4), invalid_ps, NamedTuple()
    )

    chain = ReservoirChain(Collect(), LinearReadout(2 => 1))
    chain_ps, chain_st = setup(rng, chain)
    invalid_chain_ps = NamedTuple{(:layer_1,)}((chain_ps.layer_1,))
    @test_throws ArgumentError ReservoirComputing.addreadout!(
        chain, zeros(Float32, 1, 2), invalid_chain_ps, chain_st
    )
end

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

    weights_default = ReservoirComputing.__fit_readout(
        RidgeRegression(regularization), states, targets
    )
    weights_nothing = ReservoirComputing.__fit_readout(
        RidgeRegression(regularization), states, targets; solver = nothing
    )
    weights_ls = ReservoirComputing.__fit_readout(
        RidgeRegression(regularization), states, targets; solver = QRFactorization()
    )

    @test weights_default == weights_nothing
    @test weights_default == weights_ls
    @test size(weights_default) == (n_outputs, n_features)
end

@testset "ESN output feedback: train, collectstates, predict" begin
    rng = MersenneTwister(42)
    eye32(m, n) = Matrix{Float32}(I, m, n)
    init_I = (rng, m, n) -> eye32(m, n)
    init_Z = (rng, m, n) -> zeros(Float32, m, n)
    init_state0(rng, m, B) = B == 1 ? zeros(Float32, m) : zeros(Float32, m, B)

    model = ESN(
        3, 3, 3, identity;
        use_feedback = true,
        use_bias = false,
        leak_coefficient = 1.0,
        init_input = init_I,
        init_reservoir = init_Z,
        init_feedback = init_I,
        init_state = init_state0,
    )
    ps, st = setup(rng, model)
    @test size(ps.reservoir.feedback_matrix) == (3, 3)
    ps = merge(ps, (readout = (; weight = eye32(3, 3)),))

    data = Float32[1 0 0; 0 1 0; 0 0 1]
    teacher_zero = zeros(Float32, 3, 3)
    teacher_shift = Float32[0 1 0; 0 0 1; 0 0 0]

    @test_throws ArgumentError collectstates(model, data, ps, st)
    states_tf, _ = collectstates(model, (data, teacher_zero), ps, st)
    @test states_tf ≈ data

    Y_unlock, _ = predict(model, data, ps, st)
    @test Y_unlock ≈ Float32[1 1 1; 0 1 1; 0 0 1]

    Y_tf, _ = predict(model, (data, teacher_zero), ps, st)
    @test Y_tf ≈ data

    @test_throws ArgumentError predict(model, 3, ps, st; initialdata = data[:, 1])

    targets = data
    ps_auto, _ = train(
        model, data, targets, ps, st;
        objective = RidgeRegression(0.0),
    )
    ps_explicit, _ = train(
        model, (data, teacher_shift), targets, ps, st;
        objective = RidgeRegression(0.0),
    )
    @test ps_auto.readout.weight ≈ ps_explicit.readout.weight

    plain = ESN(3, 12, 2)
    ps_p, st_p = setup(MersenneTwister(42), plain)
    @test_throws ArgumentError train(
        plain, (data, teacher_zero), randn(Float32, 2, 3), ps_p, st_p
    )
end
