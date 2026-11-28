using Test
using Random
using LinearAlgebra
using ReservoirComputing
using LuxCore: setup
import LuxCore

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

@testset "train rejects targets with wrong number of rows" begin
    rng = MersenneTwister(77)
    X = randn(rng, Float32, 2, 30)
    Y = randn(rng, Float32, 2, 30)
    esn = ESN(2, 10, 2)
    ps, st = setup(rng, esn)
    @test_throws DimensionMismatch train(esn, X, Y[1:1, :], ps, st)
    @test_throws DimensionMismatch train(esn, X, vcat(Y, Y), ps, st)
    ps2, _ = train(esn, X, Y, ps, st)
    @test size(ps2.readout.weight) == (2, 10)

    chain = ReservoirChain(StatefulLayer(ESNCell(2 => 10)), LinearReadout(10 => 2))
    cps, cst = setup(rng, chain)
    @test_throws DimensionMismatch train(chain, X, Y[1:1, :], cps, cst)
    cps2, _ = train(chain, X, Y, cps, cst)
    @test size(last(cps2).weight) == (2, 10)
end

@testset "train on a Conceptor checks target rows" begin
    rng = MersenneTwister(78)
    X = rand(rng, Float32, 1, 30)
    Y = rand(rng, Float32, 1, 30)
    c = Conceptor(ESN(1, 10, 1))
    ps, st = setup(rng, c)
    ps2, _ = train(c, X, Y, ps, st)
    @test size(ps2.model.readout.weight) == (1, 10)
    @test_throws DimensionMismatch train(c, X, vcat(Y, Y), ps, st)
end

@testset "QRFactorization is LinearSolve's and is not ambiguous with it" begin
    user_module = Module()
    Core.eval(user_module, :(using ReservoirComputing, LinearSolve))
    linearsolve_qr = Core.eval(user_module, :(LinearSolve.QRFactorization))

    @test ReservoirComputing.QRFactorization === linearsolve_qr
    @test Core.eval(user_module, :(QRFactorization())) isa linearsolve_qr
end

struct CustomTrainableReadout <: ReservoirComputing.AbstractReservoirTrainableLayer
    n::Int
    m::Int
    named::Bool
end
function LuxCore.initialparameters(::AbstractRNG, l::CustomTrainableReadout)
    W = zeros(Float32, l.m, l.n)
    return l.named ? (; weight = W) : (; w = W)
end
LuxCore.initialstates(::AbstractRNG, ::CustomTrainableReadout) = NamedTuple()
function (l::CustomTrainableReadout)(x, ps, st)
    return (l.named ? ps.weight : ps.w) * x, st
end

@testset "train installs weights in custom trainable layers" begin
    rng = MersenneTwister(77)
    x = randn(rng, Float32, 2, 60)
    y = randn(rng, Float32, 1, 60)

    rc = ReservoirChain(StatefulLayer(ESNCell(2 => 8)), CustomTrainableReadout(8, 1, true))
    ps, st = setup(rng, rc)
    ps2, _ = train(rc, x, y, ps, st)
    @test size(ps2.layer_2.weight) == (1, 8)
    @test ps2.layer_2.weight != ps.layer_2.weight

    rc_bad = ReservoirChain(
        StatefulLayer(ESNCell(2 => 8)), CustomTrainableReadout(8, 1, false)
    )
    ps_bad, st_bad = setup(rng, rc_bad)
    @test_throws ArgumentError train(rc_bad, x, y, ps_bad, st_bad)

    for named in (true, false)
        rc_after = ReservoirChain(
            StatefulLayer(ESNCell(2 => 5)), LinearReadout(5 => 2),
            CustomTrainableReadout(2, 1, named)
        )
        ps_a, st_a = setup(rng, rc_after)
        x2 = randn(rng, Float32, 2, 60)
        y2 = randn(rng, Float32, 2, 60)
        ps_a2, _ = train(rc_after, x2, y2, ps_a, st_a)
        @test size(ps_a2.layer_3.weight) == (2, 5)
        @test ps_a2.layer_4 == ps_a.layer_4
    end
end

struct CustomMatrixReadout <: ReservoirComputing.AbstractReservoirTrainableLayer
    n::Int
    m::Int
end
function LuxCore.initialparameters(::AbstractRNG, l::CustomMatrixReadout)
    return (; weight = Matrix{Float64}(I, l.m, l.n))
end
LuxCore.initialstates(::AbstractRNG, ::CustomMatrixReadout) = NamedTuple()
(l::CustomMatrixReadout)(x, ps, st) = ps.weight * x, st

@testset "custom trainable readout is the only layer fitted" begin
    rng = MersenneTwister(78)
    x = randn(rng, 2, 80)
    target = [2.0 0.0; 0.0 3.0]
    rc = ReservoirChain(CustomMatrixReadout(2, 2), LinearReadout(2 => 2))
    ps, st = setup(rng, rc)
    ps = merge(ps, (; layer_3 = (; weight = Matrix{Float64}(I, 2, 2))))
    ps2, _ = train(rc, x, target * x, ps, st)
    @test ps2.layer_3 == ps.layer_3
    @test ps2.layer_1.weight ≈ target
    @test first(rc([1.0, 1.0], ps2, st)) ≈ [2.0, 3.0]

    rc_rows = ReservoirChain(CustomMatrixReadout(2, 1), LinearReadout(1 => 2))
    ps_r, st_r = setup(rng, rc_rows)
    ps_r2, _ = train(rc_rows, x, x[1:1, :], ps_r, st_r)
    @test size(ps_r2.layer_1.weight) == (1, 2)
    @test ps_r2.layer_3 == ps_r.layer_3
end

@testset "custom trainable readout validates fitted shape" begin
    rng = MersenneTwister(79)
    x = randn(rng, 2, 40)
    rc = ReservoirChain(CustomMatrixReadout(2, 1))
    ps, st = setup(rng, rc)
    @test_throws DimensionMismatch train(rc, x, x, ps, st)
    ps2, _ = train(rc, x, x[1:1, :], ps, st)
    @test size(ps2.layer_1.weight) == (1, 2)
end
