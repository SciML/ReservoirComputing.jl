begin
    using Test
    using Random
    using ReservoirComputing
    using StateSpaceSets: StateSpaceSet
    using LuxCore: setup

    rng = MersenneTwister(42)
    in_dims, res_dims, out_dims = 3, 10, 2
    n_steps = 24
    train_data = randn(rng, Float32, in_dims, n_steps)
    target_data = randn(rng, Float32, out_dims, n_steps)
    train_set = StateSpaceSet(permutedims(train_data))
    target_set = StateSpaceSet(permutedims(target_data))

    model = ESN(in_dims, res_dims, out_dims)
    ps, st = setup(rng, model)

    @testset "StateSpaceSet train/predict" begin
        ps_matrix, st_matrix = train(
            model, train_data, target_data, ps, st;
            objective = RidgeRegression(1.0f-3),
            washout = 3,
        )
        ps_set, st_set = train(
            model, train_set, target_set, ps, st;
            objective = RidgeRegression(1.0f-3),
            washout = 3,
        )
        @test ps_set.readout.weight ≈ ps_matrix.readout.weight

        pred_matrix, _ = predict(model, train_data, ps_matrix, st_matrix)
        pred_set, _ = predict(model, train_set, ps_set, st_set)
        @test pred_set isa StateSpaceSet
        @test stack(pred_set) ≈ pred_matrix

        empty_set = StateSpaceSet{3, Float32}()
        @test_throws ArgumentError train(model, empty_set, empty_set, ps, st)
        @test_throws ArgumentError predict(model, empty_set, ps, st)
    end
end
