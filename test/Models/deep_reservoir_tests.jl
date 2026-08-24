begin
    using Test
    using Random
    using ReservoirComputing
    using CellularAutomata: DCA
    using LuxCore: setup

    @testset "DeepReservoir wrapper" begin
        rng = MersenneTwister(777)

        in_dims = 20
        res_dims = 100
        out_dims = 10

        dummy_readout = LinearReadout(res_dims => out_dims)

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

            feedforward = LinearReadout(in_dims => res_dims)
            dres_feedforward = DeepReservoir((feedforward,), dummy_readout)
            @test dres_feedforward.cells == (feedforward,)

            dres_broadcast = DeepReservoir(
                (cell1, cell2), dummy_readout; make_stateful = (true,)
            )
            @test all(layer -> layer isa ReservoirComputing.StatefulLayer,
                dres_broadcast.cells)
            @test_throws DimensionMismatch DeepReservoir(
                (cell1, cell2), dummy_readout;
                make_stateful = (true, false, true)
            )
            @test_throws ArgumentError DeepReservoir((), dummy_readout)
            @test_throws ArgumentError DeepReservoir(
                (LSMCell(in_dims => res_dims; tspan = (0.0, 1.0)),),
                dummy_readout
            )
        end

        @testset "recurrent cell composability" begin
            nonlinear = MemoryESNCell((in_dims, res_dims) => res_dims)
            linear = ESNCell(in_dims => res_dims)
            cells_to_test = (
                ESNCell(in_dims => res_dims),
                MemoryESNCell((in_dims, res_dims) => res_dims),
                ES2NCell(in_dims => res_dims),
                EuSNCell(in_dims => res_dims),
                EIESNCell(in_dims => res_dims),
                AdditiveEIESNCell(in_dims => res_dims),
                ResESNCell(in_dims => res_dims),
                MemoryResESNCell((in_dims, res_dims) => res_dims),
                LIFESNCell(in_dims => res_dims),
                RMNCell(nonlinear, linear),
            )

            for raw_cell in cells_to_test
                desn = DeepReservoir((raw_cell,), dummy_readout)
                ps, st = setup(rng, desn)

                x = rand(Float32, in_dims, 1)
                y, st_new = desn(x, ps, st)

                @test size(y) == (out_dims, 1)
                @test haskey(st_new, :cells)
                @test length(st_new.cells) == 1
            end

            reca_model = RECA(
                in_dims, out_dims, DCA(90);
                input_encoding = RandomMapping(2, res_dims), generations = 2
            )
            reca_cell = reca_model.reservoir.cell
            reca_readout = LinearReadout(reca_cell.enc.states_size => out_dims)
            dres_reca = DeepReservoir((reca_cell,), reca_readout)
            ps_reca, st_reca = setup(rng, dres_reca)
            y_reca, st_reca_new = dres_reca(
                Int.(rand(rng, Bool, in_dims)), ps_reca, st_reca
            )
            @test size(y_reca) == (out_dims,)
            @test only(st_reca_new.cells).carry !== nothing
        end

        @testset "state modifier normalization" begin
            cell1 = ESNCell(in_dims => res_dims)
            cell2 = ESNCell(res_dims => res_dims)

            single = DeepReservoir((cell1,), dummy_readout; state_modifiers = NLAT2())
            @test length(single.state_modifiers) == 1
            @test length(only(single.state_modifiers)) == 1

            sequence = DeepReservoir(
                (cell1,), dummy_readout; state_modifiers = (NLAT1(), NLAT2())
            )
            @test length(only(sequence.state_modifiers)) == 2

            broadcasted = DeepReservoir(
                (cell1, cell2), dummy_readout; state_modifiers = NLAT2()
            )
            @test all(modifiers -> length(modifiers) == 1, broadcasted.state_modifiers)

            @test_throws DimensionMismatch DeepReservoir(
                (cell1, cell2), dummy_readout;
                state_modifiers = (nothing, nothing, nothing)
            )
        end

        @testset "collectstates with hybrid stack data flow" begin
            cell1 = ESNCell(in_dims => res_dims)
            feedforward_layer = LinearReadout(res_dims => res_dims)

            desn = DeepReservoir((cell1, feedforward_layer), dummy_readout; make_stateful = (true, false))
            ps, st = setup(rng, desn)

            seq_len = 10
            X = rand(Float32, in_dims, seq_len)

            S, st_new = collectstates(desn, X, ps, st)

            @test size(S) == (res_dims, seq_len)
            @test haskey(st_new, :cells)
        end

        @testset "resetcarry! respects heterogeneous stacks" begin
            cell = ESNCell(in_dims => res_dims)
            feedforward = LinearReadout(res_dims => res_dims)
            dres = DeepReservoir(
                (cell, feedforward), dummy_readout;
                make_stateful = (true, false)
            )
            _, st = setup(rng, dres)
            initializer = (rng, dims) -> ones(Float32, dims)
            reset_st = resetcarry!(
                rng, dres, st; init_carry = (initializer, nothing)
            )
            @test only(reset_st.cells[1].carry) == ones(Float32, res_dims)
            @test reset_st.cells[2] == st.cells[2]
            @test_throws ArgumentError resetcarry!(
                rng, dres, st; init_carry = initializer
            )

            nonlinear = MemoryESNCell((in_dims, res_dims) => res_dims)
            linear = ESNCell(in_dims => res_dims)
            rmn = DeepReservoir(
                (RMNCell(nonlinear, linear),), dummy_readout
            )
            _, rmn_st = setup(rng, rmn)
            reset_rmn_st = resetcarry!(rng, rmn, rmn_st; init_carry = initializer)
            @test length(only(reset_rmn_st.cells).carry) == 2
            @test all(
                state -> state == ones(Float32, res_dims),
                only(reset_rmn_st.cells).carry
            )
        end

        @testset "training replaces the readout" begin
            dres = DeepReservoir(
                (ESNCell(in_dims => res_dims),), dummy_readout
            )
            ps, st = setup(rng, dres)
            train_data = rand(rng, Float32, in_dims, 12)
            target_data = rand(rng, Float32, out_dims, 12)
            trained_ps, trained_st = train(
                dres, train_data, target_data, ps, st;
                objective = RidgeRegression(1.0f-4)
            )
            @test size(trained_ps.readout.weight) == (out_dims, res_dims)
            @test length(trained_st.cells) == 1
        end
    end
end
