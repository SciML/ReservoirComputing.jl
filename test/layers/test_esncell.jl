using Test
using Random
using LinearAlgebra
using ReservoirComputing
using Static

const _I32 = (m, n) -> Matrix{Float32}(I, m, n)
const _Z32 = m -> zeros(Float32, m)
const _O32 = (rng, m) -> zeros(Float32, m)
const _W_I = (rng, m, n) -> _I32(m, n)
const _W_ZZ = (rng, m, n) -> zeros(Float32, m, n)

function init_state3(rng::AbstractRNG, m::Integer, B::Integer)
    B == 1 ? zeros(Float32, m) : zeros(Float32, m, B)
end

cell_name(::Type{C}) where {C} = string(nameof(C))

mix_kw(::Type{ESNCell}) = :leak_coefficient
mix_kw(::Type{ES2NCell}) = :proximity

# Whatever show() actually prints:
mix_label(::Type{ESNCell}) = "leak_coefficient"
mix_label(::Type{ES2NCell}) = "proximity"

default_extra_ctor_kwargs(::Type{ESNCell}) = NamedTuple()
default_extra_ctor_kwargs(::Type{ES2NCell}) = (init_orthogonal = _W_I,)

extra_param_keys(::Type{ESNCell}) = ()
extra_param_keys(::Type{ES2NCell}) = (:orthogonal_matrix,)

function build_cell(::Type{C}, in_dims::Integer, out_dims::Integer;
        activation = tanh,
        mix::Real = 1.0,
        use_bias = False(),
        init_input = _W_I,
        init_reservoir = _W_I,
        init_bias = _O32,
        init_state = _Z32,
        extra::NamedTuple = NamedTuple()
) where {C}
    base = (use_bias = use_bias,
        init_input = init_input,
        init_reservoir = init_reservoir,
        init_bias = init_bias,
        init_state = init_state)

    mixnt = NamedTuple{(mix_kw(C),)}((mix,))

    kw = merge(base, default_extra_ctor_kwargs(C), mixnt, extra)

    return C(in_dims => out_dims, activation; kw...)
end

function test_echo_state_cell_contract(::Type{C}) where {C}
    @testset "$(cell_name(C)): constructor & show" begin
        cell = build_cell(C, 3, 5; mix = 0.3, use_bias = False())
        io = IOBuffer()
        show(io, cell)
        shown = String(take!(io))

        @test occursin("$(cell_name(C))(3 => 5", shown)
        @test occursin(Regex("$(mix_label(C))=0\\.3(f0)?"), shown)
        @test occursin("use_bias=false", shown)
    end

    @testset "$(cell_name(C)): initialparameters shapes & bias flag" begin
        rng = MersenneTwister(1)

        cell_nobias = build_cell(C, 3, 4; use_bias = False(),
            init_input = _W_I, init_reservoir = _W_I, init_bias = _O32)

        ps_nb = initialparameters(rng, cell_nobias)
        @test haskey(ps_nb, :input_matrix)
        @test haskey(ps_nb, :reservoir_matrix)
        @test size(ps_nb.input_matrix) == (4, 3)
        @test size(ps_nb.reservoir_matrix) == (4, 4)
        @test !haskey(ps_nb, :bias)

        for k in extra_param_keys(C)
            @test haskey(ps_nb, k)
        end
        if C === ES2NCell
            @test size(ps_nb.orthogonal_matrix) == (4, 4)
        end

        cell_bias = build_cell(C, 3, 4; use_bias = True(),
            init_input = _W_I, init_reservoir = _W_I, init_bias = _O32)

        ps_b = initialparameters(rng, cell_bias)
        @test haskey(ps_b, :bias)
        @test length(ps_b.bias) == 4
    end

    @testset "$(cell_name(C)): initialstates carries RNG replica" begin
        rng = MersenneTwister(2)
        cell = build_cell(C, 2, 2)
        st = initialstates(rng, cell)
        @test haskey(st, :rng)
    end

    @testset "$(cell_name(C)): forward (vector) — identity + mix=1 gives linear map" begin
        cell = build_cell(C, 3, 3;
            activation = identity,
            mix = 1.0,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_state = _Z32)

        ps = initialparameters(MersenneTwister(0), cell)
        x = Float32[1, 2, 3]
        h0 = zeros(Float32, 3)

        (y_tuple, st2) = cell((x, (h0,)), ps, NamedTuple())
        y, (hcarry,) = y_tuple
        @test y ≈ x
        @test hcarry ≈ y
        @test st2 === NamedTuple()
    end

    @testset "$(cell_name(C)): forward (vector) — mix extremes" begin
        cell0 = build_cell(C, 3, 3;
            activation = identity,
            mix = 0.0,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_I,
            init_state = _Z32)

        ps0 = initialparameters(MersenneTwister(0), cell0)
        x = Float32[10, 20, 30]
        h0 = Float32[4, 5, 6]
        (y0_tuple, _) = cell0((x, (h0,)), ps0, NamedTuple())
        y0, _ = y0_tuple
        @test y0 ≈ h0

        cell1 = build_cell(C, 3, 3;
            activation = identity,
            mix = 1.0,
            use_bias = True(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_bias = (rng, m) -> ones(Float32, m),
            init_state = _Z32)

        ps1 = initialparameters(MersenneTwister(0), cell1)
        (y1_tuple, _) = cell1((x, (zeros(Float32, 3),)), ps1, NamedTuple())
        y1, _ = y1_tuple
        @test y1 ≈ x .+ 1.0f0
    end

    @testset "$(cell_name(C)): forward (matrix batch)" begin
        cell = build_cell(C, 3, 3;
            activation = identity,
            mix = 1.0,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_state = _Z32)

        ps = initialparameters(MersenneTwister(0), cell)
        X = Float32[1 2; 3 4; 5 6]  # (3, 2)
        H0 = zeros(Float32, 3, 2)

        (Y_tuple, _) = cell((X, (H0,)), ps, NamedTuple())
        Y, _ = Y_tuple
        @test size(Y) == (3, 2)
        @test Y ≈ X
    end

    @testset "$(cell_name(C)): outer call computes its own initial hidden state" begin
        rng = MersenneTwister(123)
        cell = build_cell(C, 2, 2;
            activation = identity,
            mix = 1.0,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_state = init_state3)

        ps = initialparameters(rng, cell)
        st = initialstates(rng, cell)

        x = Float32[7, 9]
        (y_tuple, st2) = cell(x, ps, st)
        y, _ = y_tuple

        @test y ≈ x
        @test haskey(st2, :rng)
    end
end

@testset "AbstractEchoStateNetworkCell contract" begin
    for C in (ESNCell, ES2NCell)
        test_echo_state_cell_contract(C)
    end
end
