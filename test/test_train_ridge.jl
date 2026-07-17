using Test
using Random
using LinearAlgebra
using ReservoirComputing
using LinearSolve

# Ridge readout training regression (#473).

"""
Reference ridge solution matching the LinearSolve extension path and the
`RidgeRegression` docstring form, with package layout:

- `states` :: `(n_features, T)`
- `targets` :: `(n_outputs, T)`
- returns `W` :: `(n_outputs, n_features)` with `targets ≈ W * states`
"""
function reference_ridge(
        states::AbstractMatrix,
        targets::AbstractMatrix,
        regularization,
    )
    n_features = size(states, 1)
    λ = convert(eltype(states), regularization)
    gram = states * states' + λ * I(n_features)
    return Matrix((gram \ (states * targets'))')
end

function random_ridge_problem(
        rng::AbstractRNG,
        ::Type{T},
        n_features::Integer,
        n_samples::Integer,
        n_outputs::Integer,
    ) where {T <: Number}
    states = randn(rng, T, n_features, n_samples)
    true_weights = randn(rng, T, n_outputs, n_features)
    targets = true_weights * states + T(0.01) * randn(rng, T, n_outputs, n_samples)
    return states, targets, true_weights
end

@testset "RidgeRegression constructors" begin
    @test RidgeRegression().reg == 0.0
    @test RidgeRegression(1.0e-3).reg == 1.0e-3
    @test RidgeRegression(Float32, 1.0e-2).reg isa Float32
end

@testset "train(RidgeRegression): shape contract" begin
    rng = MersenneTwister(42)
    n_features, n_samples, n_outputs = 6, 40, 3
    states, targets, _ = random_ridge_problem(
        rng, Float64, n_features, n_samples, n_outputs
    )
    regularization = 1.0e-3

    weights = train(
        RidgeRegression(regularization), states, targets; solver = QRSolver()
    )
    @test size(weights) == (n_outputs, n_features)
    @test eltype(weights) <: Real
    @test all(isfinite, weights)
end

@testset "train(RidgeRegression): closed-form orthogonal features" begin
    # X = √(T) * I so XX' = T I, and with λ the Gram is (T+λ)I.
    n_features = 4
    n_samples = n_features
    n_outputs = 2
    scale = 2.0
    states = scale * Matrix{Float64}(I, n_features, n_samples)
    true_weights = [
        1.0 0.0 -1.0 0.5
        0.0 2.0 0.0 -0.5
    ]
    targets = true_weights * states
    regularization = 0.25

    expected = true_weights * (scale^2 / (scale^2 + regularization))
    # targets ≈ true_weights * states and states * states' = scale^2 I, so
    # W = Y X' (XX' + λI)^{-1} = true_weights * scale^2 * inv(scale^2 + λ) I

    for solver in (QRSolver(), QRFactorization(), SVDFactorization())
        @testset "$(typeof(solver))" begin
            weights = train(
                RidgeRegression(regularization), states, targets; solver = solver
            )
            @test size(weights) == (n_outputs, n_features)
            @test weights ≈ expected rtol = 1.0e-10
        end
    end
end

@testset "train(RidgeRegression): well-conditioned agreement" begin
    rng = MersenneTwister(7)
    n_features, n_samples, n_outputs = 5, 60, 2
    regularization = 1.0e-2
    states, targets, _ = random_ridge_problem(
        rng, Float64, n_features, n_samples, n_outputs
    )
    reference = reference_ridge(states, targets, regularization)

    weights_qr = train(
        RidgeRegression(regularization), states, targets; solver = QRSolver()
    )
    weights_ls_qr = train(
        RidgeRegression(regularization), states, targets; solver = QRFactorization()
    )
    weights_ls_svd = train(
        RidgeRegression(regularization), states, targets; solver = SVDFactorization()
    )

    @test weights_qr ≈ reference rtol = 1.0e-10
    @test weights_ls_qr ≈ reference rtol = 1.0e-10
    @test weights_ls_svd ≈ reference rtol = 1.0e-10
    @test weights_qr ≈ weights_ls_qr rtol = 1.0e-10
    @test weights_ls_qr ≈ weights_ls_svd rtol = 1.0e-10
end

@testset "train(RidgeRegression): Float32 agreement" begin
    rng = MersenneTwister(11)
    n_features, n_samples, n_outputs = 4, 50, 2
    regularization = 1.0f-3
    states, targets, _ = random_ridge_problem(
        rng, Float32, n_features, n_samples, n_outputs
    )
    reference = reference_ridge(states, targets, regularization)

    weights_qr = train(
        RidgeRegression(regularization), states, targets; solver = QRSolver()
    )
    weights_ls = train(
        RidgeRegression(regularization), states, targets; solver = QRFactorization()
    )

    @test size(weights_qr) == (n_outputs, n_features)
    @test size(weights_ls) == (n_outputs, n_features)
    @test weights_qr ≈ reference rtol = 1.0e-4
    @test weights_ls ≈ reference rtol = 1.0e-4
    # LinearSolve path preserves feature eltype via convert(eltype(states), reg).
    @test eltype(weights_ls) == Float32
end

@testset "train(RidgeRegression): T != n_features" begin
    rng = MersenneTwister(3)
    n_features, n_samples, n_outputs = 8, 20, 3
    regularization = 1.0e-4
    states, targets, _ = random_ridge_problem(
        rng, Float64, n_features, n_samples, n_outputs
    )
    reference = reference_ridge(states, targets, regularization)

    for solver in (QRSolver(), QRFactorization(), SVDFactorization())
        @testset "$(typeof(solver))" begin
            weights = train(
                RidgeRegression(regularization), states, targets; solver = solver
            )
            @test size(weights) == (n_outputs, n_features)
            @test weights ≈ reference rtol = 1.0e-9
        end
    end
end

@testset "train(RidgeRegression): zero regularization, overdetermined" begin
    rng = MersenneTwister(19)
    n_features, n_samples, n_outputs = 4, 50, 2
    states, targets, _ = random_ridge_problem(
        rng, Float64, n_features, n_samples, n_outputs
    )
    reference = reference_ridge(states, targets, 0.0)

    for solver in (QRSolver(), QRFactorization(), SVDFactorization())
        @testset "$(typeof(solver))" begin
            weights = train(RidgeRegression(0.0), states, targets; solver = solver)
            @test size(weights) == (n_outputs, n_features)
            @test all(isfinite, weights)
            @test weights ≈ reference rtol = 1.0e-9
        end
    end
end

@testset "train(RidgeRegression): default solver is QRFactorization" begin
    rng = MersenneTwister(23)
    states, targets, _ = random_ridge_problem(rng, Float64, 5, 30, 2)
    regularization = 1.0e-3

    weights_default = train(RidgeRegression(regularization), states, targets)
    weights_explicit = train(
        RidgeRegression(regularization), states, targets; solver = QRFactorization()
    )
    @test weights_default == weights_explicit
end

@testset "train(RidgeRegression): multi-output consistency" begin
    rng = MersenneTwister(29)
    n_features, n_samples = 6, 40
    regularization = 1.0e-2
    states = randn(rng, Float64, n_features, n_samples)
    targets = randn(rng, Float64, 4, n_samples)

    weights_all = train(
        RidgeRegression(regularization), states, targets; solver = QRFactorization()
    )
    @test size(weights_all) == (4, n_features)

    for output_index in 1:4
        weights_one = train(
            RidgeRegression(regularization),
            states,
            targets[output_index:output_index, :];
            solver = QRFactorization(),
        )
        @test weights_all[output_index:output_index, :] ≈ weights_one rtol = 1.0e-10
    end
end

@testset "train(RidgeRegression): DimensionMismatch on sample count" begin
    states = randn(Float64, 5, 10)
    targets = randn(Float64, 2, 9)
    @test_throws DimensionMismatch train(
        RidgeRegression(1.0e-3), states, targets; solver = QRSolver()
    )
    @test_throws DimensionMismatch train(
        RidgeRegression(1.0e-3), states, targets; solver = QRFactorization()
    )
end

@testset "train(RidgeRegression): unsupported solver" begin
    states = randn(Float64, 4, 20)
    targets = randn(Float64, 2, 20)
    @test_throws ArgumentError train(
        RidgeRegression(1.0e-3), states, targets; solver = :not_a_solver
    )
end

@testset "train(RidgeRegression): negative regularization rejected" begin
    states = randn(Float64, 4, 20)
    targets = randn(Float64, 2, 20)
    @test_throws ArgumentError train(
        RidgeRegression(-1.0e-3), states, targets; solver = QRSolver()
    )
    @test_throws ArgumentError train(
        RidgeRegression(-1.0e-3), states, targets; solver = QRFactorization()
    )
end

@testset "train(RidgeRegression): LinearSolve multi-RHS matches reference" begin
    rng = MersenneTwister(41)
    n_features, n_samples, n_outputs = 7, 45, 5
    regularization = 5.0e-3
    states, targets, _ = random_ridge_problem(
        rng, Float64, n_features, n_samples, n_outputs
    )
    reference = reference_ridge(states, targets, regularization)

    for solver in (QRFactorization(), SVDFactorization())
        @testset "$(typeof(solver))" begin
            weights = train(
                RidgeRegression(regularization), states, targets; solver = solver
            )
            @test size(weights) == (n_outputs, n_features)
            @test weights ≈ reference rtol = 1.0e-10
        end
    end
end

@testset "train(RidgeRegression): LinearSolve uses same augmented system as QRSolver" begin
    rng = MersenneTwister(43)
    n_features, n_samples, n_outputs = 6, 25, 3
    regularization = 1.0e-8
    states, targets, _ = random_ridge_problem(
        rng, Float64, n_features, n_samples, n_outputs
    )

    weights_legacy = train(
        RidgeRegression(regularization), states, targets; solver = QRSolver()
    )
    weights_ls = train(
        RidgeRegression(regularization), states, targets; solver = QRFactorization()
    )
    @test weights_ls ≈ weights_legacy rtol = 1.0e-10
end

@testset "_apply_washout" begin
    states = reshape(collect(1.0:20.0), 4, 5)
    targets = reshape(collect(100.0:109.0), 2, 5)

    states_wo, targets_wo = ReservoirComputing._apply_washout(states, targets, 2)
    @test size(states_wo) == (4, 3)
    @test size(targets_wo) == (2, 3)
    @test states_wo == states[:, 3:5]
    @test targets_wo == targets[:, 3:5]

    @test_throws ArgumentError ReservoirComputing._apply_washout(states, targets, -1)
    @test_throws ArgumentError ReservoirComputing._apply_washout(states, targets, 5)
end

@testset "train: ESN ridge smoke" begin
    rng = MersenneTwister(42)
    in_dims, res_dims, out_dims = 3, 12, 2
    n_steps = 40
    train_data = randn(rng, Float32, in_dims, n_steps)
    target_data = randn(rng, Float32, out_dims, n_steps)

    model = ESN(in_dims, res_dims, out_dims)
    ps, st = setup(rng, model)

    ps_trained, st_trained = train(
        model, train_data, target_data, ps, st;
        objective = RidgeRegression(1.0e-4),
    )
    @test haskey(ps_trained.readout, :weight)
    @test size(ps_trained.readout.weight) == (out_dims, res_dims)
    @test all(isfinite, ps_trained.readout.weight)

    (ps2, st2), collected = train(
        model, train_data, target_data, ps, st;
        objective = RidgeRegression(1.0e-4),
        return_states = true,
        washout = 5,
    )
    @test size(collected) == (res_dims, n_steps - 5)
    @test size(ps2.readout.weight) == (out_dims, res_dims)
    @test all(isfinite, ps2.readout.weight)

    prediction, _ = model(train_data[:, end], ps_trained, st_trained)
    @test size(prediction) == (out_dims, 1)
    @test all(isfinite, prediction)
end

@testset "train: LinearSolve solver kwarg" begin
    rng = MersenneTwister(99)
    in_dims, res_dims, out_dims = 3, 10, 2
    n_steps = 30
    train_data = randn(rng, Float32, in_dims, n_steps)
    target_data = randn(rng, Float32, out_dims, n_steps)

    model = ESN(in_dims, res_dims, out_dims)
    ps, st = setup(rng, model)

    ps_qr, _ = train(
        model, train_data, target_data, ps, st;
        objective = RidgeRegression(1.0e-3),
        solver = QRSolver(),
    )
    ps_ls, _ = train(
        model, train_data, target_data, ps, st;
        objective = RidgeRegression(1.0e-3),
        solver = QRFactorization(),
    )

    @test size(ps_qr.readout.weight) == (out_dims, res_dims)
    @test size(ps_ls.readout.weight) == (out_dims, res_dims)
    @test ps_qr.readout.weight ≈ ps_ls.readout.weight rtol = 1.0e-3
end

@testset "train(RidgeRegression): ill-conditioned λ=0 is finite (no tight agreement)" begin
    # Characterization only: n == T and λ == 0 is poorly conditioned.
    # Solvers need not agree; they must return finite weights of the right shape.
    rng = MersenneTwister(1)
    n_features = 20
    n_samples = 20
    n_outputs = 3
    states = randn(rng, Float32, n_features, n_samples)
    targets = randn(rng, Float32, n_outputs, n_samples)

    for solver in (QRSolver(), QRFactorization(), SVDFactorization())
        @testset "$(typeof(solver))" begin
            weights = train(RidgeRegression(0.0), states, targets; solver = solver)
            @test size(weights) == (n_outputs, n_features)
            @test all(isfinite, weights)
        end
    end
end
