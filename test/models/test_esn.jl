using Test
using Random
using ReservoirComputing
using LinearAlgebra
using Static

const _I32 = (m, n) -> Matrix{Float32}(I, m, n)
const _Z32 = m -> zeros(Float32, m)
const _O32 = (rng, m) -> zeros(Float32, m)
const _W_I = (rng, m, n) -> _I32(m, n)
const _W_ZZ = (rng, m, n) -> zeros(Float32, m, n)

function init_state3(rng::AbstractRNG, m::Integer, B::Integer)
    return B == 1 ? zeros(Float32, m) : zeros(Float32, m, B)
end

function _with_identity_readout(ps::NamedTuple; out_dims::Integer, in_dims::Integer)
    ro_ps = haskey(ps.readout, :bias) ?
        (weight = _I32(out_dims, in_dims), bias = _Z32(out_dims)) :
        (weight = _I32(out_dims, in_dims),)
    return merge(ps, (readout = ro_ps,))
end

model_name(::Type{M}) where {M} = string(nameof(M))

mix_kw(::Type{ESN}) = :leak_coefficient
mix_kw(::Type{ES2N}) = :proximity
mix_kw(::Type{EuSN}) = :leak_coefficient
mix_kw(::Type{ResESN}) = :beta

reservoir_param_keys(::Type{ESN}) = (:input_matrix, :reservoir_matrix)
reservoir_param_keys(::Type{ES2N}) = (:input_matrix, :reservoir_matrix, :orthogonal_matrix)
reservoir_param_keys(::Type{EuSN}) = (:input_matrix, :reservoir_matrix)
reservoir_param_keys(::Type{ResESN}) = (:input_matrix, :reservoir_matrix, :orthogonal_matrix)

default_reservoir_kwargs(::Type{ESN}) = NamedTuple()
default_reservoir_kwargs(::Type{ES2N}) = (init_orthogonal = _W_I,)
default_reservoir_kwargs(::Type{EuSN}) = NamedTuple()
default_reservoir_kwargs(::Type{ResESN}) = (init_orthogonal = _W_I, alpha = 1.0)

function build_model(
        ::Type{M}, in_dims::Int, res_dims::Int, out_dims::Int, activation;
        state_modifiers = (),
        readout_activation = identity,
        mix::Real = 1.0,
        use_bias = False(),
        init_input = _W_I,
        init_reservoir = _W_ZZ,
        init_bias = _O32,
        init_state = init_state3,
        extra::NamedTuple = NamedTuple()
    ) where {M}
    base = (
        use_bias = use_bias,
        init_input = init_input,
        init_reservoir = init_reservoir,
        init_bias = init_bias,
        init_state = init_state,
    )

    mixnt = NamedTuple{(mix_kw(M),)}((mix,))
    kw = merge(base, default_reservoir_kwargs(M), mixnt, extra)

    return M(
        in_dims, res_dims, out_dims, activation;
        state_modifiers = state_modifiers,
        readout_activation = readout_activation,
        kw...
    )
end

function test_esn_family_contract(::Type{M}) where {M}
    @testset "$(model_name(M)): constructor & parameter/state shapes" begin
        rng = MersenneTwister(42)
        in_dims, res_dims, out_dims = 3, 5, 4

        model = build_model(
            M, in_dims, res_dims, out_dims, identity;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            mix = 1.0
        )

        ps, st = setup(rng, model)

        @test haskey(ps, :reservoir)
        for k in reservoir_param_keys(M)
            @test haskey(ps.reservoir, k)
        end
        @test !haskey(ps.reservoir, :bias)
        @test size(ps.reservoir.input_matrix) == (res_dims, in_dims)
        @test size(ps.reservoir.reservoir_matrix) == (res_dims, res_dims)
        if M === ES2N || M === ResESN
            @test size(ps.reservoir.orthogonal_matrix) == (res_dims, res_dims)
        end

        @test haskey(ps, :readout)
        @test haskey(ps.readout, :weight)
        @test size(ps.readout.weight) == (out_dims, res_dims)

        @test haskey(st, :reservoir)
        @test haskey(st, :states_modifiers)
        @test haskey(st, :readout)
        @test st.states_modifiers isa Tuple
    end

    @testset "$(model_name(M)): forward (vector) with identity pipeline -> y == x (dimensions matched)" begin
        rng = MersenneTwister(0)
        D = 3

        model = build_model(
            M, D, D, D, identity;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            mix = 1.0
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[1, 2, 3]
        X = reshape(x, :, 1)

        Y, st2 = model(X, ps, st)

        @test size(Y) == (D, 1)
        @test vec(Y) ≈ x
        @test haskey(st2, :reservoir) && haskey(st2, :states_modifiers) &&
            haskey(st2, :readout)
    end

    @testset "$(model_name(M)): forward (batch matrix) with identity pipeline -> Y == X" begin
        rng = MersenneTwister(1)
        D, B = 3, 2

        model = build_model(
            M, D, D, D, identity;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            mix = 1.0
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        X = Float32[1 2; 3 4; 5 6]
        Y, _ = model(X, ps, st)

        @test size(Y) == (D, B)
        @test Y ≈ X
    end

    @testset "$(model_name(M)): state_modifiers are applied (single modifier doubles features)" begin
        rng = MersenneTwister(2)
        D = 3

        model = build_model(
            M, D, D, D, identity;
            state_modifiers = (x -> 2.0f0 .* x,),
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            mix = 1.0
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[1, 2, 3]
        y, _ = model(x, ps, st)
        @test y ≈ 2.0f0 .* x
    end

    @testset "$(model_name(M)): multiple state_modifiers apply in order" begin
        rng = MersenneTwister(3)
        D = 3
        mods = (x -> x .+ 1.0f0, x -> 3.0f0 .* x)

        model = build_model(
            M, D, D, D, identity;
            state_modifiers = mods,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            mix = 1.0
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[0, 1, 2]
        y, _ = model(x, ps, st)
        @test y ≈ 3.0f0 .* (x .+ 1.0f0)
    end

    @testset "$(model_name(M)): outer call computes its own initial hidden state through reservoir cell" begin
        rng = MersenneTwister(123)
        D = 2

        model = build_model(
            M, D, D, D, identity;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_state = init_state3,
            mix = 1.0
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[7, 9]
        y, st2 = model(x, ps, st)

        @test y ≈ x
        @test haskey(st2, :reservoir)
        @test haskey(st2, :states_modifiers)
        @test haskey(st2, :readout)
    end

    return @testset "$(model_name(M)): readout_activation is honored" begin
        rng = MersenneTwister(4)
        D = 3

        model = build_model(
            M, D, D, D, identity;
            readout_activation = x -> max.(x, 0.0f0),
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            mix = 1.0
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[-1, 0.5, -3]
        y, _ = model(x, ps, st)
        @test y ≈ max.(x, 0.0f0)
    end
end

function build_model(
        ::Type{RMNESN}, in_dims::Int, res_dims::Int, out_dims::Int, activation;
        mem_dims::Int = res_dims,
        state_modifiers = (),
        readout_activation = identity,
        use_bias = False(),
        init_input = _W_I,
        init_reservoir = _W_ZZ,
        init_bias = _O32,
        init_state = init_state3,
        extra::NamedTuple = NamedTuple()
    )
    base = (
        use_bias = use_bias,
        init_input = init_input,
        init_reservoir = init_reservoir,
        init_bias = init_bias,
        init_state = init_state,
        use_memory_bias = False(),
        init_memory_input = _W_ZZ,
        init_memory_reservoir = _W_ZZ,
        init_memory_bias = _O32,
        init_memory_state = init_state3,
        init_memory = _W_ZZ,
    )

    kw = merge(base, extra)

    return RMNESN(
        in_dims, mem_dims, res_dims, out_dims, activation;
        state_modifiers = state_modifiers,
        readout_activation = readout_activation,
        kw...
    )
end

function test_esn_family_contract(::Type{RMNESN})
    @testset "RMNESN: constructor & parameter/state shapes" begin
        rng = MersenneTwister(42)
        in_dims, mem_dims, res_dims, out_dims = 3, 4, 5, 2

        model = build_model(
            RMNESN, in_dims, res_dims, out_dims, identity;
            mem_dims = mem_dims,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_I,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_I,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)

        @test haskey(ps, :reservoir)
        @test haskey(ps.reservoir, :linear_reservoir)
        @test haskey(ps.reservoir, :nonlinear_reservoir)

        lin_ps = ps.reservoir.linear_reservoir
        nonlin_ps = ps.reservoir.nonlinear_reservoir

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

        @test haskey(ps, :readout)
        @test haskey(ps.readout, :weight)
        @test size(ps.readout.weight) == (out_dims, res_dims)

        @test haskey(st, :reservoir)
        @test haskey(st, :states_modifiers)
        @test haskey(st, :readout)
        @test st.states_modifiers isa Tuple
        @test st.reservoir isa NamedTuple
    end

    @testset "RMNESN: optional memory and nonlinear biases are parameterized" begin
        rng = MersenneTwister(43)
        in_dims, mem_dims, res_dims, out_dims = 3, 4, 5, 2

        model = build_model(
            RMNESN, in_dims, res_dims, out_dims, identity;
            mem_dims = mem_dims,
            use_bias = True(),
            init_bias = _O32,
            extra = (
                use_memory_bias = True(),
                init_memory_bias = _O32,
            )
        )

        ps, _ = setup(rng, model)

        @test haskey(ps.reservoir.linear_reservoir, :bias)
        @test size(ps.reservoir.linear_reservoir.bias) == (mem_dims,)

        @test haskey(ps.reservoir.nonlinear_reservoir, :bias)
        @test size(ps.reservoir.nonlinear_reservoir.bias) == (res_dims,)
    end

    @testset "RMNESN: forward (vector) with identity pipeline -> y == x (dimensions matched)" begin
        rng = MersenneTwister(0)
        D = 3

        model = build_model(
            RMNESN, D, D, D, identity;
            mem_dims = D,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_ZZ,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_ZZ,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[1, 2, 3]
        X = reshape(x, :, 1)

        Y, st2 = model(X, ps, st)

        @test size(Y) == (D, 1)
        @test vec(Y) ≈ x
        @test haskey(st2, :reservoir) && haskey(st2, :states_modifiers) &&
            haskey(st2, :readout)
    end

    @testset "RMNESN: forward (batch matrix) with identity pipeline -> Y == X" begin
        rng = MersenneTwister(1)
        D, B = 3, 2

        model = build_model(
            RMNESN, D, D, D, identity;
            mem_dims = D,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_ZZ,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_ZZ,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        X = Float32[1 2; 3 4; 5 6]
        Y, _ = model(X, ps, st)

        @test size(Y) == (D, B)
        @test Y ≈ X
    end

    @testset "RMNESN: memory reservoir contributes to nonlinear reservoir" begin
        rng = MersenneTwister(2)
        D = 3

        model = build_model(
            RMNESN, D, D, D, identity;
            mem_dims = D,
            use_bias = False(),
            init_input = _W_ZZ,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_I,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_I,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[1, 2, 3]
        y, _ = model(x, ps, st)

        @test vec(y) ≈ x
    end

    @testset "RMNESN: direct input and memory contribution are combined" begin
        rng = MersenneTwister(3)
        D = 3

        model = build_model(
            RMNESN, D, D, D, identity;
            mem_dims = D,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_I,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_I,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[1, 2, 3]
        y, _ = model(x, ps, st)

        @test vec(y) ≈ 2.0f0 .* x
    end

    @testset "RMNESN: state_modifiers are applied (single modifier doubles features)" begin
        rng = MersenneTwister(4)
        D = 3

        model = build_model(
            RMNESN, D, D, D, identity;
            mem_dims = D,
            state_modifiers = (x -> 2.0f0 .* x,),
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_ZZ,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_ZZ,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[1, 2, 3]
        y, _ = model(x, ps, st)

        @test vec(y) ≈ 2.0f0 .* x
    end

    @testset "RMNESN: multiple state_modifiers apply in order" begin
        rng = MersenneTwister(5)
        D = 3
        mods = (x -> x .+ 1.0f0, x -> 3.0f0 .* x)

        model = build_model(
            RMNESN, D, D, D, identity;
            mem_dims = D,
            state_modifiers = mods,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_ZZ,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_ZZ,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[0, 1, 2]
        y, _ = model(x, ps, st)

        @test vec(y) ≈ 3.0f0 .* (x .+ 1.0f0)
    end

    @testset "RMNESN: outer call computes its own initial hidden state through reservoir cell" begin
        rng = MersenneTwister(123)
        D = 2

        model = build_model(
            RMNESN, D, D, D, identity;
            mem_dims = D,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_ZZ,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_ZZ,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[7, 9]
        y, st2 = model(x, ps, st)

        @test vec(y) ≈ x
        @test haskey(st2, :reservoir)
        @test haskey(st2, :states_modifiers)
        @test haskey(st2, :readout)
    end

    return @testset "RMNESN: readout_activation is honored" begin
        rng = MersenneTwister(6)
        D = 3

        model = build_model(
            RMNESN, D, D, D, identity;
            mem_dims = D,
            readout_activation = x -> max.(x, 0.0f0),
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_ZZ,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_ZZ,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[-1, 0.5, -3]
        y, _ = model(x, ps, st)

        @test vec(y) ≈ max.(x, 0.0f0)
    end
end

function build_model(
        ::Type{RMNResESN}, in_dims::Int, res_dims::Int, out_dims::Int, activation;
        mem_dims::Int = res_dims,
        state_modifiers = (),
        readout_activation = identity,
        use_bias = False(),
        init_input = _W_I,
        init_reservoir = _W_ZZ,
        init_bias = _O32,
        init_state = init_state3,
        alpha::Real = 1.0,
        beta::Real = 1.0,
        extra::NamedTuple = NamedTuple()
    )
    base = (
        use_bias = use_bias,
        init_input = init_input,
        init_reservoir = init_reservoir,
        init_orthogonal = _W_I,
        init_bias = init_bias,
        init_state = init_state,
        use_memory_bias = False(),
        init_memory_input = _W_ZZ,
        init_memory_reservoir = _W_ZZ,
        init_memory_bias = _O32,
        init_memory_state = init_state3,
        init_memory = _W_ZZ,
        alpha = alpha,
        beta = beta,
    )

    kw = merge(base, extra)

    return RMNResESN(
        in_dims, mem_dims, res_dims, out_dims, activation;
        state_modifiers = state_modifiers,
        readout_activation = readout_activation,
        kw...
    )
end

function test_esn_family_contract(::Type{RMNResESN})
    @testset "RMNResESN: constructor & parameter/state shapes" begin
        rng = MersenneTwister(42)
        in_dims, mem_dims, res_dims, out_dims = 3, 4, 5, 2

        model = build_model(
            RMNResESN, in_dims, res_dims, out_dims, identity;
            mem_dims = mem_dims,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_I,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_I,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)

        @test haskey(ps, :reservoir)
        @test haskey(ps.reservoir, :linear_reservoir)
        @test haskey(ps.reservoir, :nonlinear_reservoir)

        lin_ps = ps.reservoir.linear_reservoir
        nonlin_ps = ps.reservoir.nonlinear_reservoir

        @test haskey(lin_ps, :input_matrix)
        @test haskey(lin_ps, :reservoir_matrix)
        @test !haskey(lin_ps, :bias)
        @test size(lin_ps.input_matrix) == (mem_dims, in_dims)
        @test size(lin_ps.reservoir_matrix) == (mem_dims, mem_dims)

        @test haskey(nonlin_ps, :input_matrix)
        @test haskey(nonlin_ps, :reservoir_matrix)
        @test haskey(nonlin_ps, :memory_matrix)
        @test haskey(nonlin_ps, :orthogonal_matrix)
        @test !haskey(nonlin_ps, :bias)
        @test size(nonlin_ps.input_matrix) == (res_dims, in_dims)
        @test size(nonlin_ps.reservoir_matrix) == (res_dims, res_dims)
        @test size(nonlin_ps.memory_matrix) == (res_dims, mem_dims)
        @test size(nonlin_ps.orthogonal_matrix) == (res_dims, res_dims)

        @test haskey(ps, :readout)
        @test haskey(ps.readout, :weight)
        @test size(ps.readout.weight) == (out_dims, res_dims)

        @test haskey(st, :reservoir)
        @test haskey(st, :states_modifiers)
        @test haskey(st, :readout)
        @test st.states_modifiers isa Tuple
        @test st.reservoir isa NamedTuple
    end

    @testset "RMNResESN: optional memory and nonlinear biases are parameterized" begin
        rng = MersenneTwister(43)
        in_dims, mem_dims, res_dims, out_dims = 3, 4, 5, 2

        model = build_model(
            RMNResESN, in_dims, res_dims, out_dims, identity;
            mem_dims = mem_dims,
            use_bias = True(),
            init_bias = _O32,
            extra = (
                use_memory_bias = True(),
                init_memory_bias = _O32,
            )
        )

        ps, _ = setup(rng, model)

        @test haskey(ps.reservoir.linear_reservoir, :bias)
        @test size(ps.reservoir.linear_reservoir.bias) == (mem_dims,)

        @test haskey(ps.reservoir.nonlinear_reservoir, :bias)
        @test size(ps.reservoir.nonlinear_reservoir.bias) == (res_dims,)
    end

    @testset "RMNResESN: forward (vector) with identity pipeline and α=β=1, O=I, W_m=0 -> y == x" begin
        rng = MersenneTwister(0)
        D = 3

        model = build_model(
            RMNResESN, D, D, D, identity;
            mem_dims = D,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_ZZ,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_ZZ,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[1, 2, 3]
        X = reshape(x, :, 1)

        Y, st2 = model(X, ps, st)

        @test size(Y) == (D, 1)
        @test vec(Y) ≈ x
        @test haskey(st2, :reservoir) && haskey(st2, :states_modifiers) &&
            haskey(st2, :readout)
    end

    @testset "RMNResESN: forward (batch matrix) with identity pipeline -> Y == X" begin
        rng = MersenneTwister(1)
        D, B = 3, 2

        model = build_model(
            RMNResESN, D, D, D, identity;
            mem_dims = D,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_ZZ,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_ZZ,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        X = Float32[1 2; 3 4; 5 6]
        Y, _ = model(X, ps, st)

        @test size(Y) == (D, B)
        @test Y ≈ X
    end

    @testset "RMNResESN: memory reservoir contributes to nonlinear reservoir" begin
        rng = MersenneTwister(2)
        D = 3

        model = build_model(
            RMNResESN, D, D, D, identity;
            mem_dims = D,
            use_bias = False(),
            init_input = _W_ZZ,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_I,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_I,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[1, 2, 3]
        y, _ = model(x, ps, st)

        @test vec(y) ≈ x
    end

    @testset "RMNResESN: direct input and memory contribution are combined" begin
        rng = MersenneTwister(3)
        D = 3

        model = build_model(
            RMNResESN, D, D, D, identity;
            mem_dims = D,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_I,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_I,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[1, 2, 3]
        y, _ = model(x, ps, st)

        @test vec(y) ≈ 2.0f0 .* x
    end

    @testset "RMNResESN: α scales the orthogonal skip when β=0" begin
        rng = MersenneTwister(11)
        D = 3
        ps_init_state = (rng, m, B) -> begin
            B == 1 ? Float32[1, 2, 3] : repeat(Float32[1, 2, 3], 1, B)
        end

        model = build_model(
            RMNResESN, D, D, D, identity;
            mem_dims = D,
            alpha = 0.5,
            beta = 0.0,
            init_input = _W_ZZ,
            init_reservoir = _W_ZZ,
            init_state = ps_init_state,
            extra = (
                init_memory_input = _W_ZZ,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_ZZ,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[0, 0, 0]
        y, _ = model(x, ps, st)

        @test vec(y) ≈ 0.5f0 .* Float32[1, 2, 3]
    end

    @testset "RMNResESN: state_modifiers are applied (single modifier doubles features)" begin
        rng = MersenneTwister(4)
        D = 3

        model = build_model(
            RMNResESN, D, D, D, identity;
            mem_dims = D,
            state_modifiers = (x -> 2.0f0 .* x,),
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_ZZ,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_ZZ,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[1, 2, 3]
        y, _ = model(x, ps, st)

        @test vec(y) ≈ 2.0f0 .* x
    end

    @testset "RMNResESN: multiple state_modifiers apply in order" begin
        rng = MersenneTwister(5)
        D = 3
        mods = (x -> x .+ 1.0f0, x -> 3.0f0 .* x)

        model = build_model(
            RMNResESN, D, D, D, identity;
            mem_dims = D,
            state_modifiers = mods,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_ZZ,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_ZZ,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[0, 1, 2]
        y, _ = model(x, ps, st)

        @test vec(y) ≈ 3.0f0 .* (x .+ 1.0f0)
    end

    @testset "RMNResESN: outer call computes its own initial hidden state through reservoir cell" begin
        rng = MersenneTwister(123)
        D = 2

        model = build_model(
            RMNResESN, D, D, D, identity;
            mem_dims = D,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_ZZ,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_ZZ,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[7, 9]
        y, st2 = model(x, ps, st)

        @test vec(y) ≈ x
        @test haskey(st2, :reservoir)
        @test haskey(st2, :states_modifiers)
        @test haskey(st2, :readout)
    end

    return @testset "RMNResESN: readout_activation is honored" begin
        rng = MersenneTwister(6)
        D = 3

        model = build_model(
            RMNResESN, D, D, D, identity;
            mem_dims = D,
            readout_activation = x -> max.(x, 0.0f0),
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            extra = (
                init_memory_input = _W_ZZ,
                init_memory_reservoir = _W_ZZ,
                init_memory = _W_ZZ,
                init_memory_state = init_state3,
            )
        )

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[-1, 0.5, -3]
        y, _ = model(x, ps, st)

        @test vec(y) ≈ max.(x, 0.0f0)
    end
end

@testset "ESN-family model contract" begin
    for M in (ESN, ES2N, EuSN, ResESN, RMNESN, RMNResESN)
        test_esn_family_contract(M)
    end
end
