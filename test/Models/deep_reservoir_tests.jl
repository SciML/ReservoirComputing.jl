begin
    using Test
    using Random
    using ReservoirComputing

    @testset "DeepReservoir wrapper" begin
        rng = MersenneTwister(777)

        in_dims = 20
        res_dims = 100
        out_dims = 10

        # Use a native library cell instead of `identity` to bypass LuxCore parsing issues
        dummy_readout = ESNCell(res_dims => out_dims)

        @testset "make_stateful logic and per-layer granularity" begin
            cell1 = ESNCell(in_dims => res_dims)
            cell2 = ESNCell(res_dims => res_dims)

            desn_default = DeepReservoir((cell1, cell2), dummy_readout)
            @test desn_default.cells[1] isa ReservoirComputing.StatefulLayer
            @test desn_default.cells[2] isa ReservoirComputing.StatefulLayer

            desn_false = DeepReservoir((cell1, cell2), dummy_readout; make_stateful = false)
            @test desn_false.cells[1] isa typeof(cell1)
            @test desn_false.cells[2] isa typeof(cell2)

            desn_mixed = DeepReservoir((cell1, cell2), dummy_readout; make_stateful = (true, false))
            @test desn_mixed.cells[1] isa ReservoirComputing.StatefulLayer
            @test desn_mixed.cells[2] isa typeof(cell2)
        end

        @testset "Composability Loop (Maintainer Request)" begin
            cells_to_test = (
                ESNCell(in_dims => res_dims),
                MemoryESNCell((in_dims, in_dims) => res_dims),
                ES2NCell(in_dims => res_dims),
            )

            # Test 1-layer deep reservoirs to ensure exact dimension matching
            for raw_cell in cells_to_test
                desn = DeepReservoir((raw_cell,), dummy_readout)
                ps, st = setup(rng, desn)

                x = rand(Float32, in_dims)
                y, st_new = desn(x, ps, st)

                @test size(y) == (out_dims,)
                @test haskey(st_new, :cells)
                @test length(st_new.cells) == 1
            end
        end

        @testset "collectstates with hybrid stack data flow" begin
            cell1 = ESNCell(in_dims => res_dims)
            cell2 = ESNCell(res_dims => res_dims)

            desn = DeepReservoir((cell1, cell2), dummy_readout; make_stateful = (true, false))
            ps, st = setup(rng, desn)

            seq_len = 10
            X = rand(Float32, in_dims, seq_len)

            S, st_new = collectstates(desn, X, ps, st)

            @test size(S) == (res_dims, seq_len)
            @test haskey(st_new, :cells)
        end
    end
end
