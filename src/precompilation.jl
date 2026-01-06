using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    using Random: Random
    using Static: Static

    @compile_workload begin
        rng = Random.MersenneTwister(0)
        input_size = 3
        reservoir_size = 5
        output_size = 2
        seq_length = 10

        # ESNCell creation and operations
        esn_cell = ESNCell(input_size => reservoir_size)
        ps_cell, st_cell = setup(rng, esn_cell)

        # Single input forward pass
        x_single = randn(Float32, input_size)
        out_single, st_cell2 = apply(esn_cell, x_single, ps_cell, st_cell)

        # Batch input forward pass
        x_batch = randn(Float32, input_size, 4)
        out_batch, st_cell3 = apply(esn_cell, x_batch, ps_cell, st_cell)

        # ESN model creation
        esn = ESN(input_size, reservoir_size, output_size, identity)
        ps, st = setup(rng, esn)

        # ESN forward pass (single)
        y_single, st2 = esn(x_single, ps, st)

        # ESN forward pass (batch)
        y_batch, st3 = esn(x_batch, ps, st)

        # Training workflow
        train_data = randn(Float32, input_size, seq_length)
        target_data = randn(Float32, output_size, seq_length)

        # collectstates
        states, st_after = collectstates(esn, train_data, ps, st)

        # train! function
        ps_trained, st_trained = train!(esn, train_data, target_data, ps, st, StandardRidge(1.0e-6))

        # ES2NCell and ES2N (another common model)
        es2n_cell = ES2NCell(input_size => reservoir_size)
        ps_es2n, st_es2n = setup(rng, es2n_cell)
        out_es2n, _ = apply(es2n_cell, x_single, ps_es2n, st_es2n)

        # EuSNCell and EuSN
        eusn_cell = EuSNCell(input_size => reservoir_size)
        ps_eusn, st_eusn = setup(rng, eusn_cell)
        out_eusn, _ = apply(eusn_cell, x_single, ps_eusn, st_eusn)

        # ReservoirChain (common usage pattern)
        chain = ReservoirChain(
            StatefulLayer(ESNCell(input_size => reservoir_size)),
            LinearReadout(reservoir_size => output_size)
        )
        ps_chain, st_chain = setup(rng, chain)
        y_chain, st_chain2 = chain(x_single, ps_chain, st_chain)

        # State modifiers
        pad = Pad()
        padded = pad(x_single)
        padded_mat = pad(x_batch)

        nlat1_out = NLAT1(x_single)
        nlat2_out = NLAT2(x_single)
        nlat3_out = NLAT3(x_single)

        ps_sq = PartialSquare(0.5)
        ps_out = ps_sq(x_single)

        ext_sq = ExtendedSquare(x_single)

        # LinearReadout standalone
        ro = LinearReadout(reservoir_size => output_size)
        ps_ro, st_ro = setup(rng, ro)
        y_ro, _ = ro(randn(Float32, reservoir_size), ps_ro, st_ro)

        # StatefulLayer
        sl = StatefulLayer(ESNCell(input_size => reservoir_size))
        ps_sl, st_sl = setup(rng, sl)
        out_sl, st_sl2 = sl(x_single, ps_sl, st_sl)

        # train (lower-level function)
        output_matrix = train(StandardRidge(1.0e-6), states, target_data)

        # resetcarry!
        st_reset = resetcarry!(rng, esn, st_trained)
    end
end
