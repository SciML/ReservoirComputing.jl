@testitem "collectstates_regression" tags = [:core, :regression] begin
    using Test
    using Random
    using ReservoirComputing
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
    # before the two-level `_collectstates` refactor. The refactor must produce
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
        current_state, partial_st = ReservoirComputing._partial_apply(rc, x1, ps, newst)
        state_dims = size(current_state, 1)
        states = similar(data, state_dims, nsteps)
        states[:, 1] .= current_state
        newst = merge(partial_st, (readout = newst.readout,))
        for (idx, inp) in Base.Iterators.drop(Base.enumerate(cols), 1)
            current_state, partial_st =
                ReservoirComputing._partial_apply(rc, inp, ps, newst)
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

end
