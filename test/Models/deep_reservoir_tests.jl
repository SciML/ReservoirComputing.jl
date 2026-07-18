begin
    using Test
    using Random
    using ReservoirComputing

    @testset "DeepReservoir wrapper" begin
        rng = MersenneTwister(42)
        in_dims = 3
        res_dims = 5
        out_dims = 2

        @testset "make_stateful logic and per-layer granularity" begin
            # We use standard ESNCells to test the wrapper logic without needing Lux
            cell1 = ESNCell(in_dims => res_dims)
            cell2 = ESNCell(res_dims => out_dims)

            # Default behavior: code should automatically wrap both in a StatefulLayer
            desn_default = DeepReservoir((cell1, cell2), identity)
            @test desn_default.cells[1] isa ReservoirComputing.StatefulLayer
            @test desn_default.cells[2] isa ReservoirComputing.StatefulLayer

            # make_stateful = false: code should leave layers exactly as they are
            desn_false = DeepReservoir((cell1, cell2), identity; make_stateful = false)
            @test desn_false.cells[1] isa typeof(cell1)
            @test desn_false.cells[2] isa typeof(cell2)

            # Tuple granularity: wrap the first layer, leave the second layer as-is
            desn_mixed = DeepReservoir((cell1, cell2), identity; make_stateful = (true, false))
            @test desn_mixed.cells[1] isa ReservoirComputing.StatefulLayer
            @test desn_mixed.cells[2] isa typeof(cell2)
        end

        @testset "Composability Loop (Maintainer Request)" begin
            # An array containing the fundamental cell types in the library
            cells_to_test = (
                ESNCell(in_dims => res_dims),
                MemoryESNCell(in_dims => res_dims),
                ES2NCell(in_dims => res_dims),
            )

            # Prove the wrapper dynamically handles any cell type thrown at it
            for raw_cell in cells_to_test
                desn = DeepReservoir((raw_cell, raw_cell), identity)
                ps, st = setup(rng, desn)

                # Pass a single-timestep dummy vector through
                x = rand(Float32, in_dims)
                y, st_new = desn(x, ps, st)

                # Verify the forward pass completed and state was tracked
                @test size(y) == (res_dims,)
                @test haskey(st_new, :cells)
                @test length(st_new.cells) == 2
            end
        end

        @testset "collectstates with hybrid stack data flow" begin
            # Stack two cells, but force the second one to be stateless (mimicking a feedforward layer)
            cell1 = ESNCell(in_dims => res_dims)
            cell2 = ESNCell(res_dims => res_dims)

            desn = DeepReservoir((cell1, cell2), identity; make_stateful = (true, false))
            ps, st = setup(rng, desn)

            # Time-series data matrix: (in_dims x sequence_length)
            seq_len = 10
            X = rand(Float32, in_dims, seq_len)

            # Run the time-series through the network
            S, st_new = collectstates(desn, X, ps, st)

            # Verify the output state dimensions correctly span the sequence length
            @test size(S) == (res_dims, seq_len)
            @test haskey(st_new, :cells)
        end
    end
end
