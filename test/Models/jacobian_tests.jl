# Closed-loop ESN Jacobian.
begin
    using Test
    using Random
    using LinearAlgebra
    using ReservoirComputing
    using LuxCore: setup
    using ForwardDiff
    using NNlib: tanh_fast

    const _ATOL = 5.0f-3

    function _finite_diff_jacobian(esn, state, ps; ε = 1.0f-3)
        n = length(state)
        J = zeros(eltype(state), n, n)
        for j in 1:n
            state_plus, state_minus = copy(state), copy(state)
            state_plus[j] += ε
            state_minus[j] -= ε
            forward_plus = first(ReservoirComputing.__closed_loop_step(esn, state_plus, ps))
            forward_minus = first(ReservoirComputing.__closed_loop_step(esn, state_minus, ps))
            J[:, j] .= (forward_plus .- forward_minus) ./ (2 * ε)
        end
        return J
    end

    function _with_readout_weights(esn, ps, rng)
        out_dims, feat_dims = Int(esn.readout.out_dims), Int(esn.readout.in_dims)
        weight = randn(rng, Float32, out_dims, feat_dims) .* 0.05f0
        readout = haskey(ps.readout, :bias) ?
            (weight = weight, bias = zeros(Float32, out_dims)) : (weight = weight,)
        return merge(ps, (readout = readout,))
    end

    function _check_fd(esn, ps, st, state; atol = _ATOL)
        J, st_out = jacobian(esn, state, ps, st)
        @test st_out === st
        @test size(J) == (length(state), length(state))
        @test J ≈ _finite_diff_jacobian(esn, state, ps) atol = atol
        return J
    end

    @testset "analytical matches finite differences" begin
        rng = MersenneTwister(42)
        esn = ESN(
            3, 6, 3;
            init_reservoir = scaled_rand, use_bias = true, leak_coefficient = 0.7f0,
        )
        ps, st = setup(rng, esn)
        ps = _with_readout_weights(esn, ps, rng)
        state = randn(rng, Float32, 6)
        J = _check_fd(esn, ps, st, state)
        @test eltype(J) === Float32
        Jbuf = similar(J)
        @test first(jacobian!(Jbuf, esn, state, ps, st)) === Jbuf
        @test Jbuf ≈ J
    end

    @testset "ForwardDiff and tanh_fast" begin
        rng = MersenneTwister(7)
        for (activation, seed) in ((tanh, 7), (tanh_fast, 19))
            Random.seed!(rng, seed)
            esn = ESN(2, 5, 2, activation; init_reservoir = scaled_rand)
            ps, st = setup(rng, esn)
            ps = _with_readout_weights(esn, ps, rng)
            state = randn(rng, Float32, 5)
            J_an, _ = jacobian(esn, state, ps, st)
            J_ad, _ = jacobian(esn, state, ps, st; backend = :forwarddiff)
            @test J_an ≈ J_ad atol = 1.0f-5
        end
    end

    @testset "vector leak" begin
        rng = MersenneTwister(11)
        leak = Float32[0.5, 0.7, 0.9, 0.6, 0.8]
        esn = ESN(2, 5, 2; init_reservoir = scaled_rand, leak_coefficient = leak)
        ps, st = setup(rng, esn)
        ps = _with_readout_weights(esn, ps, rng)
        _check_fd(esn, ps, st, randn(rng, Float32, 5))
    end

    @testset "state modifiers" begin
        rng = MersenneTwister(13)
        for modifier in (NLAT1, NLAT2, NLAT3, PartialSquare(0.5), ExtendedSquare, Pad(1.0f0))
            res_dims = modifier isa Pad ? 5 : 6
            readout_in_dims = modifier === ExtendedSquare ? 12 :
                modifier isa Pad ? 6 : res_dims
            esn = ESN(
                2, res_dims, 2;
                init_reservoir = scaled_rand,
                state_modifiers = modifier,
                readout_in_dims = readout_in_dims,
            )
            ps, st = setup(rng, esn)
            ps = _with_readout_weights(esn, ps, rng)
            _check_fd(esn, ps, st, randn(rng, Float32, res_dims))
        end

        state = Float32[1, 2, 3, 4, 5]
        M = ReservoirComputing.__state_modifier_jacobian(NLAT2, state)
        @test M[3, 3] == 0 && M[3, 2] == state[1] && M[3, 1] == state[2]
        @test M[5, 5] == 0 && M[5, 4] == state[3] && M[5, 3] == state[4]
    end

    @testset "jacobians matches predict" begin
        rng = MersenneTwister(23)
        esn = ESN(3, 6, 3; init_reservoir = scaled_rand)
        ps, st = setup(rng, esn)
        ps = _with_readout_weights(esn, ps, rng)
        initialdata = Float32[0.1, -0.2, 0.3]
        Js, outputs, st_final = jacobians(esn, 4, ps, st; initialdata)
        predicted, st_pred = predict(esn, 4, ps, st; initialdata)
        @test outputs ≈ predicted
        @test size(Js) == (6, 6, 4)
        @test st_final.reservoir.carry == st_pred.reservoir.carry
        _, st_step = apply(esn, initialdata, ps, st)
        for t in 1:4
            state = ReservoirComputing.__carry_state_vector(st_step)
            @test Js[:, :, t] ≈ first(jacobian(esn, state, ps, st_step)) atol = 1.0f-6
            t < 4 && ((_, st_step) = apply(esn, outputs[:, t], ps, st_step))
        end
    end

    @testset "errors" begin
        rng = MersenneTwister(29)
        esn_bad = ESN(3, 5, 2; init_reservoir = scaled_rand)
        ps, st = setup(rng, esn_bad)
        state = randn(rng, Float32, 5)
        @test_throws DimensionMismatch jacobian(esn_bad, state, ps, st)

        esn_ext = ESN(
            3, 5, 3;
            init_reservoir = scaled_rand,
            state_modifiers = Extend(Collect()),
            readout_in_dims = 8,
        )
        ps_ext, st_ext = setup(rng, esn_ext)
        @test_throws ArgumentError jacobian(esn_ext, state, ps_ext, st_ext)

        esn = ESN(3, 5, 3; init_reservoir = scaled_rand)
        ps, st = setup(rng, esn)
        @test_throws ArgumentError jacobian(esn, state, ps, st; backend = :something)
        @test_throws DimensionMismatch jacobian!(zeros(Float32, 4, 4), esn, state, ps, st)
        @test_throws ArgumentError jacobians(esn, 0, ps, st; initialdata = Float32[1, 2, 3])
    end
end
