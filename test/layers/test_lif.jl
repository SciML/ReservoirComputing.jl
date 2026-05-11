using Test
using Random
using LinearAlgebra
using ReservoirComputing
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
                cell_ctor, in_dims => out_dims, H, identity;
                init_buffer = _Z32, kw...
            )
        catch
        end
    end
    error("Could not build LIF for $(cell_ctor).")
end

@testset "LocalInformationFlow: generic wrapper contract" begin
    cell_ctors = (
        ESNCell,
        ES2NCell,
        EuSNCell,
        EIESNCell,
        AdditiveEIESNCell,
    )

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
        end
    end
end
