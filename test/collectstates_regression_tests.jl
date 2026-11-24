begin
    using Test
    using Random
    using ReservoirComputing
    using LuxCore: setup
    using LinearAlgebra
    using Static

    const SEED = 42
    const IN_DIMS = 3
    const RES_DIMS = 8
    const OUT_DIMS = 2
    const T = 20

    _seeded_input(rng, m, n) = 0.5f0 .* randn(rng, Float32, m, n)
    _seeded_reservoir(rng, m, n) = 0.3f0 .* randn(rng, Float32, m, n)
    _seeded_bias(rng, m) = 0.1f0 .* randn(rng, Float32, m)
    init_state3(rng, m, B) = B == 1 ? zeros(Float32, m) : zeros(Float32, m, B)

    fixed_kwargs() = (
        use_bias = False(),
        init_input = _seeded_input,
        init_reservoir = _seeded_reservoir,
        init_bias = _seeded_bias,
        init_state = init_state3,
    )

    function build_models()
        fk = fixed_kwargs()
        return [
            (
                "ESN",
                ESN(IN_DIMS, RES_DIMS, OUT_DIMS, identity; fk..., leak_coefficient = 1.0),
            ),
            (
                "EuSN",
                EuSN(IN_DIMS, RES_DIMS, OUT_DIMS, identity; fk..., leak_coefficient = 1.0),
            ),
            (
                "ES2N",
                ES2N(
                    IN_DIMS,
                    RES_DIMS,
                    OUT_DIMS,
                    identity;
                    fk...,
                    proximity = 1.0,
                    init_orthogonal = _seeded_reservoir,
                ),
            ),
            (
                "ResESN",
                ResESN(
                    IN_DIMS,
                    RES_DIMS,
                    OUT_DIMS,
                    identity;
                    fk...,
                    beta = 1.0,
                    init_orthogonal = _seeded_reservoir,
                    alpha = 1.0,
                ),
            ),
            ("SVESM", SVESM(IN_DIMS, RES_DIMS, OUT_DIMS, identity; fk...)),
            ("LIFESN", LIFESN(IN_DIMS, RES_DIMS, OUT_DIMS, identity; fk...)),
        ]
    end

    # Byte-for-byte reproduction of the discrete `collectstates` body as it stood
    # before the two-level `__collectstates` refactor. The refactor must produce
    # the same output for every model that flows through the generic dispatch.
    function reference_collectstates(
            rc::ReservoirComputing.AbstractReservoirComputer,
            data::AbstractMatrix,
            ps,
            st::NamedTuple,
        )
        newst = st
        nsteps = size(data, 2)
        cols = eachcol(data)
        @assert !isempty(cols)
        x1 = first(cols)
        current_state, partial_st = ReservoirComputing.__partial_apply(rc, x1, ps, newst)
        state_dims = size(current_state, 1)
        states = similar(data, state_dims, nsteps)
        states[:, 1] .= current_state
        newst = merge(partial_st, (readout = newst.readout,))
        for (idx, inp) in Base.Iterators.drop(Base.enumerate(cols), 1)
            current_state, partial_st =
                ReservoirComputing.__partial_apply(rc, inp, ps, newst)
            states[:, idx] .= current_state
            newst = merge(partial_st, (readout = newst.readout,))
        end
        return states, newst
    end

    @testset "collectstates dispatch regression" begin
        for (name, model) in build_models()
            @testset "$name" begin
                rng_setup = MersenneTwister(SEED)
                ps, st = setup(rng_setup, model)
                data = randn(MersenneTwister(SEED + 1), Float32, IN_DIMS, T)

                states_new, _ = collectstates(model, data, ps, st)
                states_ref, _ = reference_collectstates(model, data, ps, st)

                @test size(states_new) == (RES_DIMS, T)
                @test size(states_new) == size(states_ref)
                @test states_new == states_ref
            end
        end
    end

    @testset "ReservoirChain collectstates inference and values" begin
        chain = ReservoirChain(
            StatefulLayer(ESNCell(IN_DIMS => 6)), Collect(), NLAT2(), Collect(),
            LinearReadout(12 => OUT_DIMS)
        )
        ps, st = setup(MersenneTwister(SEED), chain)
        data = randn(MersenneTwister(SEED + 1), Float32, IN_DIMS, T)

        @inferred collectstates(chain, data, ps, st)
        states, st_end = collectstates(chain, data, ps, st)

        ref = zeros(Float32, 12, T)
        cur = st
        for (t, x) in enumerate(eachcol(data))
            h, s1 = chain.layers.layer_1(x, ps.layer_1, cur.layer_1)
            h2, s3 = chain.layers.layer_3(h, ps.layer_3, cur.layer_3)
            h3, s4 = chain.layers.layer_4(h2, ps.layer_4, cur.layer_4)
            ref[:, t] .= vcat(h, h3)
            cur = merge(cur, (layer_1 = s1, layer_3 = s3, layer_4 = s4))
        end
        @test states == ref
        @test st_end == cur
        @test_throws ArgumentError collectstates(chain, zeros(Float32, IN_DIMS, 0), ps, st)

        chain1 = ReservoirChain(
            StatefulLayer(ESNCell(IN_DIMS => 6)), LinearReadout(6 => OUT_DIMS)
        )
        ps1, st1 = setup(MersenneTwister(SEED), chain1)
        @inferred collectstates(chain1, data, ps1, st1)
    end

end
