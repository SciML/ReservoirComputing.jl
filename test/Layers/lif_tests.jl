# Local-information-flow layer contracts.
begin
    using Test
    using Random
    using LinearAlgebra
    using ReservoirComputing
    using LuxCore: initialparameters, initialstates
    using Static

    const _W_I = (rng, m, n) -> Matrix{Float32}(I, m, n)
    const _W_Z = (rng, m, n) -> zeros(Float32, m, n)
    const _Z32 = (rng, dims...) -> zeros(Float32, dims...)

    function build_lif(cell_ctor, in_dims::Int, out_dims::Int, H::Int)
        attempts = (
            (; init_input = _W_I, init_reservoir = _W_Z, use_bias = false),
            (; init_input = _W_I, init_reservoir = _W_Z, use_bias = False()),
            (; init_input = _W_I, init_reservoir = _W_Z),
        )
        for kw in attempts
            try
                return LocalInformationFlow(
                    cell_ctor,
                    in_dims => out_dims,
                    H,
                    identity;
                    init_buffer = _Z32,
                    kw...,
                )
            catch
            end
        end
        error("Could not build LIF for $(cell_ctor).")
    end

    function build_local_lif(cell_ctor, H::Int; init_state = zeros32, init_buffer = zeros32)
        return LocalInformationFlow(cell_ctor, 1 => 30, H, tanh; init_buffer, init_state)
    end

    const _RNG_MAKERS = (
        () -> (Random.seed!(7); Random.default_rng()),
        () -> Xoshiro(7),
        () -> MersenneTwister(7),
    )

    function run_lif_sequence(layer, ps, st, inputs)
        y = nothing
        for u in inputs
            y, st = layer(Float32[u], ps, st)
        end
        return y
    end

    @testset "LocalInformationFlow: generic wrapper contract" begin
        cell_ctors = (ESNCell, ES2NCell, EuSNCell, EIESNCell, AdditiveEIESNCell)

        @testset "ps/st tree shape" begin
            rng = MersenneTwister(1)
            for ctor in cell_ctors
                lif = build_lif(ctor, 3, 5, 4)

                ps = initialparameters(rng, lif)
                st = initialstates(rng, lif)

                @test haskey(ps, :cell)
                @test haskey(st, :rng)
                @test haskey(st, :cell)
                @test haskey(st, :input_buffer)
                @test st.input_buffer === nothing
                @test haskey(st, :start_states)
                @test st.start_states === nothing
            end
        end

        @testset "buffer lazy init + shift/append (vector)" begin
            rng = MersenneTwister(2)
            H = 3
            in_dims = 3
            out_dims = 100

            x1 = Float32[10, 20, 30]
            x2 = Float32[1, 1, 1]
            x3 = Float32[7, 8, 9]

            for ctor in cell_ctors
                lif = build_lif(ctor, in_dims, out_dims, H)
                ps = initialparameters(rng, lif)
                st0 = initialstates(rng, lif)

                (y1_tuple, st1) = lif(x1, ps, st0)
                y1, _ = y1_tuple
                @test st1.input_buffer !== nothing
                @test length(st1.input_buffer) == H - 1
                @test st1.input_buffer[end] == x1

                (y2_tuple, st2) = lif(x2, ps, st1)
                y2, _ = y2_tuple
                @test st2.input_buffer[1] == x1
                @test st2.input_buffer[end] == x2

                (y3_tuple, st3) = lif(x3, ps, st2)
                y3, _ = y3_tuple
                @test st3.input_buffer[1] == x2
                @test st3.input_buffer[end] == x3
            end
        end

        @testset "buffer lazy init + shift/append (matrix batch)" begin
            rng = MersenneTwister(3)
            H = 4
            in_dims = 3
            out_dims = 100

            X1 = Float32[1 2; 3 4; 5 6]
            X2 = Float32[6 5; 4 3; 2 1]
            X3 = Float32[0 1; 0 1; 0 1]
            X4 = Float32[9 9; 9 9; 9 9]

            for ctor in cell_ctors
                lif = build_lif(ctor, in_dims, out_dims, H)
                ps = initialparameters(rng, lif)
                st0 = initialstates(rng, lif)

                (_, st1) = lif(X1, ps, st0)
                @test st1.input_buffer !== nothing
                @test length(st1.input_buffer) == H - 1

                # Determine ordering: where did X1 get written after the first call?
                buf1 = st1.input_buffer
                x1_at_first = buf1[1] == X1
                x1_at_last = buf1[end] == X1
                @test x1_at_first ⊻ x1_at_last  # exactly one should be true

                (_, st2) = lif(X2, ps, st1)
                buf2 = st2.input_buffer

                if x1_at_first
                    # newest-first convention: [X2, X1, 0]
                    @test buf2[1] == X2
                    @test buf2[2] == X1
                else
                    # oldest-first convention: [0, X1, X2] OR [X1, X2, 0] depending on fill policy,
                    # but your implementation (Base.tail... , inp) is oldest-first with left shift:
                    @test buf2[end] == X2
                    @test buf2[end - 1] == X1
                end

                (_, st3) = lif(X3, ps, st2)
                buf3 = st3.input_buffer
                if x1_at_first
                    @test buf3[1] == X3
                    @test buf3[2] == X2
                    @test buf3[3] == X1
                else
                    @test buf3[end] == X3
                    @test buf3[end - 1] == X2
                    @test buf3[end - 2] == X1
                end

                (_, st4) = lif(X4, ps, st3)
                buf4 = st4.input_buffer
                if x1_at_first
                    # [X4, X3, X2]
                    @test buf4[1] == X4
                    @test buf4[2] == X3
                    @test buf4[3] == X2
                else
                    # [..., X2, X3, X4]
                    @test buf4[end] == X4
                    @test buf4[end - 1] == X3
                    @test buf4[end - 2] == X2
                end
            end
        end

        @testset "horizon=1 disables buffering" begin
            rng = MersenneTwister(4)
            H = 1
            in_dims = 3
            out_dims = 100

            x = Float32[1, 2, 3]

            for ctor in cell_ctors
                lif = build_lif(ctor, in_dims, out_dims, H)
                ps = initialparameters(rng, lif)
                st0 = initialstates(rng, lif)

                (y_tuple, st1) = lif(x, ps, st0)
                y, _ = y_tuple

                @test st1.input_buffer == () || st1.input_buffer === nothing
                @test st1.start_states !== nothing
            end
        end

        @testset "output depends only on the last lookback_horizon inputs" begin
            seq_a = [5.0, -3.0, 1.0, 0.2, 0.4, 0.6, 0.8]
            seq_b = [-4.0, 2.0, 7.0, 0.2, 0.4, 0.6, 0.8]

            for ctor in cell_ctors
                lif = StatefulLayer(build_local_lif(ctor, 3))
                ps, st = setup(MersenneTwister(5), lif)

                y_a = run_lif_sequence(lif, ps, st, seq_a)
                y_b = run_lif_sequence(lif, ps, st, seq_b)

                @test y_a ≈ y_b rtol = 1.0e-6
            end
        end

        @testset "lookback_horizon=1 uses no past input" begin
            for ctor in cell_ctors
                lif = StatefulLayer(build_local_lif(ctor, 1; init_state = randn32))
                ps, st = setup(MersenneTwister(5), lif)

                y_long = run_lif_sequence(lif, ps, st, [5.0, -3.0, 0.8])
                y_short = run_lif_sequence(lif, ps, st, [0.1, 0.8])

                @test y_long ≈ y_short rtol = 1.0e-6
            end
        end

        @testset "the carried state does not influence the output" begin
            x1 = Float32[0.3]
            x2 = Float32[-0.7]
            x3 = Float32[1.1]
            arbitrary = (fill(100.0f0, 30, 1),)

            for ctor in cell_ctors
                lif = build_local_lif(ctor, 3)
                ps = initialparameters(MersenneTwister(6), lif)
                st = initialstates(MersenneTwister(6), lif)

                (_, carry1), st1 = lif(x1, ps, st)
                (_, carry2), st2 = lif((x2, carry1), ps, st1)

                (y_normal, carry_normal), st_normal = lif((x3, carry2), ps, st2)
                (y_other, carry_other), st_other = lif((x3, arbitrary), ps, st2)
                (y_none, _), _ = lif(x3, ps, st2)

                @test y_other ≈ y_normal rtol = 1.0e-6
                @test y_none ≈ y_normal rtol = 1.0e-6
                @test only(carry_other) ≈ only(carry_normal) rtol = 1.0e-6
                @test st_other.input_buffer == st_normal.input_buffer
            end
        end

        @testset "a random init_state is the same for every window" begin
            seq = [5.0, -3.0, 1.0, 0.2, 0.4, 0.6, 0.8]
            seq_b = [-4.0, 2.0, 7.0, 0.2, 0.4, 0.6, 0.8]
            seq_c = [5.0, -3.0, 1.0, 0.2, 0.4, -1.5, 0.8]

            for make_rng in _RNG_MAKERS, ctor in cell_ctors
                lif = StatefulLayer(build_local_lif(ctor, 3; init_state = randn32))
                ps, st = setup(make_rng(), lif)

                y_long = run_lif_sequence(lif, ps, st, seq)
                y_short = run_lif_sequence(lif, ps, st, seq[(end - 3):end])
                @test run_lif_sequence(lif, ps, st, seq) == y_long
                @test y_long ≈ run_lif_sequence(lif, ps, st, seq_b) rtol = 1.0e-6
                @test y_long ≈ y_short rtol = 1.0e-6
                @test !(y_long ≈ run_lif_sequence(lif, ps, st, seq_c))

                zero_lif = StatefulLayer(build_local_lif(ctor, 3))
                ps_zero, st_zero = setup(make_rng(), zero_lif)
                @test !(y_long ≈ run_lif_sequence(zero_lif, ps_zero, st_zero, seq))
            end
        end

        @testset "lazy initialization does not advance the RNG of the state" begin
            x = Float32[0.3]

            for make_rng in _RNG_MAKERS, ctor in cell_ctors
                lif = build_local_lif(ctor, 3; init_state = randn32, init_buffer = randn32)
                ps = initialparameters(MersenneTwister(6), lif)
                st = initialstates(make_rng(), lif)
                rng_before = copy(st.rng)

                (y_1, _), st_1 = lif(x, ps, st)
                (y_2, _), st_2 = lif(x, ps, st)

                @test st.rng == rng_before
                @test st.input_buffer === nothing
                @test st.start_states === nothing
                @test y_1 == y_2
                @test st_1.input_buffer == st_2.input_buffer
                @test st_1.start_states == st_2.start_states
                @test st_1.rng == st_2.rng
                @test st_1.rng === st.rng

                (_, _), st_3 = lif(x, ps, st_1)
                @test st_3.rng === st_1.rng
                @test st_3.start_states === st_1.start_states
            end
        end

        @testset "the buffer keeps its own copy of the inputs" begin
            for ctor in cell_ctors
                lif = build_local_lif(ctor, 3; init_state = randn32)
                ps = initialparameters(MersenneTwister(8), lif)
                st = initialstates(MersenneTwister(8), lif)

                x1 = Float32[0.4]
                x2 = Float32[-0.6]
                (_, _), st_1 = lif(x1, ps, st)
                (_, _), st_2 = lif(x2, ps, st_1)
                (y_before, _), _ = lif(Float32[0.9], ps, st_2)

                fill!(x1, 100.0f0)
                fill!(x2, -100.0f0)
                (y_after, _), _ = lif(Float32[0.9], ps, st_2)

                @test st_2.input_buffer == (Float32[0.4], Float32[-0.6])
                @test y_after == y_before
            end
        end

        @testset "a saved state without start_states is accepted" begin
            x1 = Float32[0.3]
            x2 = Float32[-0.7]
            x3 = Float32[1.1]
            x4 = Float32[0.2]

            for make_rng in _RNG_MAKERS, ctor in cell_ctors
                lif = build_local_lif(ctor, 3; init_state = randn32)
                ps = initialparameters(MersenneTwister(9), lif)
                st = initialstates(make_rng(), lif)
                (_, _), st_1 = lif(x1, ps, st)
                (_, _), warm = lif(x2, ps, st_1)

                old_layout = (rng = warm.rng, input_buffer = warm.input_buffer, cell = warm.cell)
                rng_before = copy(old_layout.rng)
                rng_after = copy(old_layout.rng)
                start_states = ReservoirComputing.init_hidden_states(rng_after, lif, x3)
                new_layout = (
                    rng = old_layout.rng,
                    input_buffer = old_layout.input_buffer,
                    start_states = start_states,
                    cell = old_layout.cell,
                )

                (y_old, carry_old), st_old = lif(x3, ps, old_layout)
                (y_new, carry_new), st_new = lif(x3, ps, new_layout)

                @test !haskey(old_layout, :start_states)
                @test old_layout.rng == rng_before
                @test haskey(st_old, :start_states)
                @test st_old.start_states == start_states
                @test st_old.rng == rng_before
                @test st_old.input_buffer == st_new.input_buffer
                @test y_old == y_new
                @test carry_old == carry_new

                (_, _), st_next = lif(x4, ps, st_old)
                @test st_next.start_states === st_old.start_states
                @test st_next.rng === st_old.rng
            end
        end
    end

end
