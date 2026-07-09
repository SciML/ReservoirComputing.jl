@testitem "rmncell" tags=[:layers, :cells, :memory] begin
    using Test
    using Random
    using LinearAlgebra
    using ReservoirComputing
    using Static

    const _I32 = (m, n) -> Matrix{Float32}(I, m, n)
    const _Z32 = m -> zeros(Float32, m)
    const _O32 = (rng, m) -> zeros(Float32, m)
    const _W_I = (rng, m, n) -> _I32(m, n)
    const _W_ZZ = (rng, m, n) -> zeros(Float32, m, n)

    function init_state3(rng::AbstractRNG, m::Integer, B::Integer)
        return B == 1 ? zeros(Float32, m) : zeros(Float32, m, B)
    end

    function build_rmncell(
        in_dims::Integer,
        mem_dims::Integer,
        res_dims::Integer;
        activation = identity,
        use_memory_bias = False(),
        use_bias = False(),
        init_memory_input = _W_I,
        init_memory_reservoir = _W_ZZ,
        init_memory_bias = _O32,
        init_memory_state = init_state3,
        init_input = _W_I,
        init_reservoir = _W_ZZ,
        init_memory = _W_I,
        init_bias = _O32,
        init_state = init_state3,
    )
        linear_reservoir = ESNCell(
            in_dims => mem_dims,
            identity;
            use_bias = use_memory_bias,
            init_input = init_memory_input,
            init_reservoir = init_memory_reservoir,
            init_bias = init_memory_bias,
            init_state = init_memory_state,
            leak_coefficient = 1.0,
        )

        nonlinear_reservoir = MemoryESNCell(
            (in_dims, mem_dims) => res_dims,
            activation;
            use_bias = use_bias,
            init_input = init_input,
            init_reservoir = init_reservoir,
            init_memory = init_memory,
            init_bias = init_bias,
            init_state = init_state,
            leak_coefficient = 1.0,
        )

        return RMNCell(nonlinear_reservoir, linear_reservoir)
    end

    @testset "RMNCell contract" begin
        @testset "RMNCell: constructor & show" begin
            cell = build_rmncell(3, 4, 5)

            io = IOBuffer()
            show(io, cell)
            shown = String(take!(io))

            @test occursin("RMNCell(", shown)
            @test occursin("nonlinear_reservoir", shown)
            @test occursin("linear_reservoir", shown)
            @test occursin("MemoryESNCell", shown)
            @test occursin("ESNCell", shown)
        end

        @testset "RMNCell: initialparameters shapes & nested keys" begin
            rng = MersenneTwister(1)
            in_dims, mem_dims, res_dims = 3, 4, 5

            cell = build_rmncell(
                in_dims,
                mem_dims,
                res_dims;
                use_memory_bias = False(),
                use_bias = False(),
                init_memory_input = _W_I,
                init_memory_reservoir = _W_ZZ,
                init_input = _W_I,
                init_reservoir = _W_ZZ,
                init_memory = _W_I,
            )

            ps = initialparameters(rng, cell)

            @test haskey(ps, :linear_reservoir)
            @test haskey(ps, :nonlinear_reservoir)

            lin_ps = ps.linear_reservoir
            nonlin_ps = ps.nonlinear_reservoir

            @test haskey(lin_ps, :input_matrix)
            @test haskey(lin_ps, :reservoir_matrix)
            @test !haskey(lin_ps, :bias)
            @test size(lin_ps.input_matrix) == (mem_dims, in_dims)
            @test size(lin_ps.reservoir_matrix) == (mem_dims, mem_dims)

            @test haskey(nonlin_ps, :input_matrix)
            @test haskey(nonlin_ps, :reservoir_matrix)
            @test haskey(nonlin_ps, :memory_matrix)
            @test !haskey(nonlin_ps, :bias)
            @test size(nonlin_ps.input_matrix) == (res_dims, in_dims)
            @test size(nonlin_ps.reservoir_matrix) == (res_dims, res_dims)
            @test size(nonlin_ps.memory_matrix) == (res_dims, mem_dims)
        end

        @testset "RMNCell: initialparameters includes optional biases" begin
            rng = MersenneTwister(2)
            in_dims, mem_dims, res_dims = 3, 4, 5

            cell = build_rmncell(
                in_dims,
                mem_dims,
                res_dims;
                use_memory_bias = True(),
                use_bias = True(),
                init_memory_bias = _O32,
                init_bias = _O32,
            )

            ps = initialparameters(rng, cell)

            @test haskey(ps.linear_reservoir, :bias)
            @test size(ps.linear_reservoir.bias) == (mem_dims,)

            @test haskey(ps.nonlinear_reservoir, :bias)
            @test size(ps.nonlinear_reservoir.bias) == (res_dims,)
        end

        @testset "RMNCell: initialstates shapes & nested keys" begin
            rng = MersenneTwister(3)

            cell = build_rmncell(3, 4, 5)
            st = initialstates(rng, cell)

            @test haskey(st, :linear_reservoir)
            @test haskey(st, :nonlinear_reservoir)
            @test haskey(st, :rng)
        end

        @testset "RMNCell: explicit forward updates memory first, then nonlinear reservoir" begin
            rng = MersenneTwister(4)
            D = 3

            cell = build_rmncell(
                D,
                D,
                D;
                activation = identity,
                use_memory_bias = False(),
                use_bias = False(),
                init_memory_input = _W_I,
                init_memory_reservoir = _W_ZZ,
                init_input = _W_ZZ,
                init_reservoir = _W_ZZ,
                init_memory = _W_I,
                init_memory_state = init_state3,
                init_state = init_state3,
            )

            ps = initialparameters(rng, cell)
            st = initialstates(rng, cell)

            x = Float32[1, 2, 3]
            h0 = zeros(Float32, D)
            m0 = zeros(Float32, D)

            (y_tuple, st2) = cell((x, (h0, m0)), ps, st)
            y, (hcarry, mcarry) = y_tuple

            @test y ≈ x
            @test hcarry ≈ y
            @test mcarry ≈ x
            @test st2 === st
        end

        @testset "RMNCell: explicit forward combines direct input and memory contribution" begin
            rng = MersenneTwister(5)
            D = 3

            cell = build_rmncell(
                D,
                D,
                D;
                activation = identity,
                use_memory_bias = False(),
                use_bias = False(),
                init_memory_input = _W_I,
                init_memory_reservoir = _W_ZZ,
                init_input = _W_I,
                init_reservoir = _W_ZZ,
                init_memory = _W_I,
                init_memory_state = init_state3,
                init_state = init_state3,
            )

            ps = initialparameters(rng, cell)
            st = initialstates(rng, cell)

            x = Float32[1, 2, 3]
            h0 = zeros(Float32, D)
            m0 = zeros(Float32, D)

            (y_tuple, _) = cell((x, (h0, m0)), ps, st)
            y, (hcarry, mcarry) = y_tuple

            @test y ≈ 2.0f0 .* x
            @test hcarry ≈ y
            @test mcarry ≈ x
        end

        @testset "RMNCell: explicit forward uses previous memory state through linear reservoir recurrence" begin
            rng = MersenneTwister(6)
            D = 3

            cell = build_rmncell(
                D,
                D,
                D;
                activation = identity,
                use_memory_bias = False(),
                use_bias = False(),
                init_memory_input = _W_ZZ,
                init_memory_reservoir = _W_I,
                init_input = _W_ZZ,
                init_reservoir = _W_ZZ,
                init_memory = _W_I,
                init_memory_state = init_state3,
                init_state = init_state3,
            )

            ps = initialparameters(rng, cell)
            st = initialstates(rng, cell)

            x = Float32[10, 20, 30]
            h0 = zeros(Float32, D)
            m0 = Float32[1, 2, 3]

            (y_tuple, _) = cell((x, (h0, m0)), ps, st)
            y, (hcarry, mcarry) = y_tuple

            @test y ≈ m0
            @test hcarry ≈ y
            @test mcarry ≈ m0
        end

        @testset "RMNCell: explicit forward supports matrix batches" begin
            rng = MersenneTwister(7)
            D, B = 3, 2

            cell = build_rmncell(
                D,
                D,
                D;
                activation = identity,
                use_memory_bias = False(),
                use_bias = False(),
                init_memory_input = _W_I,
                init_memory_reservoir = _W_ZZ,
                init_input = _W_ZZ,
                init_reservoir = _W_ZZ,
                init_memory = _W_I,
                init_memory_state = init_state3,
                init_state = init_state3,
            )

            ps = initialparameters(rng, cell)
            st = initialstates(rng, cell)

            X = Float32[1 2; 3 4; 5 6]
            H0 = zeros(Float32, D, B)
            M0 = zeros(Float32, D, B)

            (Y_tuple, _) = cell((X, (H0, M0)), ps, st)
            Y, (Hcarry, Mcary) = Y_tuple

            @test size(Y) == (D, B)
            @test Y ≈ X
            @test Hcarry ≈ Y
            @test Mcary ≈ X
        end

        @testset "RMNCell: outer call computes initial hidden and memory states" begin
            rng = MersenneTwister(8)
            D = 3

            cell = build_rmncell(
                D,
                D,
                D;
                activation = identity,
                use_memory_bias = False(),
                use_bias = False(),
                init_memory_input = _W_I,
                init_memory_reservoir = _W_ZZ,
                init_input = _W_ZZ,
                init_reservoir = _W_ZZ,
                init_memory = _W_I,
                init_memory_state = init_state3,
                init_state = init_state3,
            )

            ps = initialparameters(rng, cell)
            st = initialstates(rng, cell)

            x = Float32[1, 2, 3]

            (y_tuple, st2) = cell(x, ps, st)
            y, (hcarry, mcarry) = y_tuple

            @test y ≈ x
            @test hcarry ≈ y
            @test mcarry ≈ x

            @test haskey(st2, :linear_reservoir)
            @test haskey(st2, :nonlinear_reservoir)
            @test haskey(st2, :rng)
        end

        @testset "RMNCell: nonlinear activation is honored" begin
            rng = MersenneTwister(9)
            D = 3

            cell = build_rmncell(
                D,
                D,
                D;
                activation = x -> max.(x, 0.0f0),
                use_memory_bias = False(),
                use_bias = False(),
                init_memory_input = _W_ZZ,
                init_memory_reservoir = _W_ZZ,
                init_input = _W_I,
                init_reservoir = _W_ZZ,
                init_memory = _W_ZZ,
                init_memory_state = init_state3,
                init_state = init_state3,
            )

            ps = initialparameters(rng, cell)
            st = initialstates(rng, cell)

            x = Float32[-1, 0.5, -3]
            h0 = zeros(Float32, D)
            m0 = zeros(Float32, D)

            (y_tuple, _) = cell((x, (h0, m0)), ps, st)
            y, _ = y_tuple

            @test y ≈ max.(x, 0.0f0)
        end
    end

end
