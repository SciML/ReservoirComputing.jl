# Optional extension loading contracts.
begin
    using Random
    using SparseArrays
    using CellularAutomata
    using MLJLinearModels
    using ReservoirComputing
    using LuxCore: initialparameters, setup

    @testset "SparseArrays extension returns sparse initializer output" begin
        @eval Main using SparseArrays
        rng = MersenneTwister(404)
        W = rand_sparse(rng, Float32, 8, 8; sparsity = 0.25, return_sparse = true)
        @test W isa SparseMatrixCSC
        @test size(W) == (8, 8)
        @test eltype(W) === Float32

        init = rand_sparse(; sparsity = 0.25, return_sparse = true)
        cell = ESNCell(3 => 8; init_reservoir = init)
        ps = initialparameters(rng, cell)
        @test ps.reservoir_matrix isa SparseMatrixCSC
        @test size(ps.reservoir_matrix) == (8, 8)
    end

    @testset "MLJLinearModels extension trains a matrix readout" begin
        states = Float64[
            1 2 3 4 5
            2 1 0 -1 -2
            1 1 1 1 1
        ]
        target = Float64[
            2 4 6 8 10
            1 0 -1 -2 -3
        ]
        regressor = MLJLinearModels.LinearRegression(fit_intercept = false)
        W = ReservoirComputing._fit_readout(regressor, states, target)
        @test size(W) == (2, 3)
        @test eltype(W) === Float64
        @test W * states ≈ target atol = 1.0e-5

        bad_regressor = MLJLinearModels.LinearRegression(fit_intercept = true)
        @test_throws ArgumentError ReservoirComputing._fit_readout(
            bad_regressor, states, target
        )
    end

    @testset "CellularAutomata extension clears a RECA carry" begin
        rng = MersenneTwister(405)
        reca = RECA(4, 4, DCA(90); input_encoding = RandomMapping(2, 8), generations = 2)
        _, st = setup(rng, reca)
        cleared = resetcarry!(rng, reca, st)
        @test cleared.reservoir.carry === nothing
    end
end
