# Optional extension loading contracts.
begin
    using Random
    using SparseArrays
    using CellularAutomata
    using MLJLinearModels
    using ReservoirComputing
    using LuxCore: initialparameters, setup

    @testset "SparseArrays extension returns sparse initializer output" begin
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

    @testset "return_sparse works from a module other than Main" begin
        code = """
        module M
        using SparseArrays, ReservoirComputing
        f() = rand_sparse(10, 10; return_sparse = true)
        end
        M.f() isa M.SparseMatrixCSC || error("not sparse")
        """
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
        @test success(pipeline(cmd; stdout = devnull, stderr = devnull))
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
        W = ReservoirComputing.__fit_readout(regressor, states, target)
        @test size(W) == (2, 3)
        @test eltype(W) === Float64
        @test W * states ≈ target atol = 1.0e-5

        bad_regressor = MLJLinearModels.LinearRegression(fit_intercept = true)
        @test_throws ArgumentError ReservoirComputing.__fit_readout(
            bad_regressor, states, target
        )
    end

    @testset "CellularAutomata extension draws RECA maps from rng" begin
        function build_reca(; kwargs...)
            return RECA(
                4, 2, DCA(90); input_encoding = RandomMapping(2, 8), generations = 2,
                kwargs...
            )
        end
        reca_maps(model) = model.reservoir.cell.enc.maps
        @test reca_maps(build_reca(; rng = MersenneTwister(1))) ==
            reca_maps(build_reca(; rng = MersenneTwister(1)))
        @test reca_maps(build_reca(; rng = MersenneTwister(1))) !=
            reca_maps(build_reca(; rng = MersenneTwister(2)))
        Random.seed!(407)
        default_maps = reca_maps(build_reca())
        Random.seed!(407)
        @test reca_maps(build_reca(; rng = Random.default_rng())) == default_maps
    end

    @testset "CellularAutomata extension clears a RECA carry" begin
        rng = MersenneTwister(405)
        reca = RECA(4, 4, DCA(90); input_encoding = RandomMapping(2, 8), generations = 2)
        _, st = setup(rng, reca)
        cleared = resetcarry!(rng, reca, st)
        @test cleared.reservoir.carry === nothing
    end

    @testset "CellularAutomata extension preserves input for Extend" begin
        rng = MersenneTwister(406)
        reca = RECA(
            4, 2, DCA(90);
            input_encoding = RandomMapping(2, 8),
            generations = 2,
            state_modifiers = (Extend(Collect()),)
        )
        ps, st = setup(rng, reca)
        data = Int[0 1 0; 1 0 1; 0 0 1; 1 1 0]
        states, _ = collectstates(reca, data, ps, st)
        @test states[1:4, :] == data
        @test size(ps.readout.weight, 2) == size(states, 1)
    end

    @testset "CellularAutomata extension accepts float input: $T" for T in (Float32, Float64)
        rng = MersenneTwister(407)
        reca = RECA(4, 2, DCA(90); input_encoding = RandomMapping(2, 8), generations = 2)
        ps, st = setup(rng, reca)
        data = T[0 1 0; 1 0 1; 0 0 1; 1 1 0]
        states, _ = collectstates(reca, data, ps, st)
        @test eltype(states) === T
        ints, _ = collectstates(reca, Int.(data), ps, st)
        @test states == ints
        trained, _ = train(
            reca, data, rand(rng, T, 2, 3), ps, st; objective = ReservoirComputing.RidgeRegression(1.0e-3)
        )
        @test all(isfinite, trained.readout.weight)
    end

    @testset "CellularAutomata extension keeps float states for CCA: $T" for T in (Float32, Float64)
        rng = MersenneTwister(408)
        reca = RECA(4, 2, CCA(0.1); input_encoding = RandomMapping(2, 8), generations = 2)
        ps, st = setup(rng, reca)
        data = T[0 1 0 1 1 0; 1 0 1 0 1 1; 0 0 1 1 0 1; 1 1 0 0 1 0]
        states, _ = collectstates(reca, data, ps, st)
        @test eltype(states) === T
        @test all(isfinite, states)
        @test any(!isinteger, states)
        cell = reca.reservoir.cell
        carry = fill(T(0.3), cell.enc.ca_size)
        (feats, (ca_next,)), _ = cell((data[:, 1], (carry,)), nothing, st.reservoir.cell)
        @test eltype(feats) === T
        @test eltype(ca_next) === T
        @test any(!isinteger, ca_next)
    end
end
