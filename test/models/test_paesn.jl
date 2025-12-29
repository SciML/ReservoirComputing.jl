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
    B == 1 ? zeros(Float32, m) : zeros(Float32, m, B)
end

function _with_identity_readout(ps::NamedTuple; out_dims::Integer, in_dims::Integer)
    ro_ps = haskey(ps.readout, :bias) ?
            (weight = _I32(out_dims, in_dims), bias = _Z32(out_dims)) :
            (weight = _I32(out_dims, in_dims),)
    return merge(ps, (readout = ro_ps,))
end

@testset "ParameterAwareESNCell tests" begin
    @testset "constructor & show" begin
        cell = ParameterAwareESNCell(3 => 5, tanh;
            leak_coefficient = 0.5,
            parameter_coupling = 0.3,
            parameter_offset = 1.0,
            use_bias = False())
        io = IOBuffer()
        show(io, cell)
        shown = String(take!(io))

        @test occursin("ParameterAwareESNCell(3 => 5", shown)
        @test occursin("leak_coefficient=0.5", shown)
        @test occursin("parameter_coupling=0.3", shown)
        @test occursin("parameter_offset=1.0", shown)
        @test occursin("use_bias=false", shown)
    end

    @testset "initialparameters shapes" begin
        rng = MersenneTwister(1)

        cell_nobias = ParameterAwareESNCell(3 => 4, tanh;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_I,
            init_parameter = _W_I)

        ps = initialparameters(rng, cell_nobias)
        @test haskey(ps, :input_matrix)
        @test haskey(ps, :reservoir_matrix)
        @test haskey(ps, :parameter_matrix)
        @test size(ps.input_matrix) == (4, 3)
        @test size(ps.reservoir_matrix) == (4, 4)
        @test size(ps.parameter_matrix) == (4, 1)
        @test !haskey(ps, :bias)

        cell_bias = ParameterAwareESNCell(3 => 4, tanh;
            use_bias = True(),
            init_input = _W_I,
            init_reservoir = _W_I,
            init_parameter = _W_I,
            init_bias = _O32)

        ps_b = initialparameters(rng, cell_bias)
        @test haskey(ps_b, :bias)
        @test length(ps_b.bias) == 4
    end

    @testset "forward with identity activation" begin
        cell = ParameterAwareESNCell(3 => 3, identity;
            leak_coefficient = 1.0,
            parameter_coupling = 1.0,
            parameter_offset = 0.0,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_parameter = _W_ZZ,
            init_state = _Z32)

        rng = MersenneTwister(0)
        ps = initialparameters(rng, cell)

        # With identity activation, zero reservoir, zero param: output = input
        x = Float32[1, 2, 3]
        param = Float32[0.5]
        h0 = zeros(Float32, 3)

        (y_tuple, st2) = cell(((x, param), (h0,)), ps, NamedTuple())
        y, (hcarry,) = y_tuple
        @test y ≈ x
        @test hcarry ≈ y
    end

    @testset "parameter contribution" begin
        # Test that parameter actually affects the output
        cell = ParameterAwareESNCell(3 => 3, identity;
            leak_coefficient = 1.0,
            parameter_coupling = 1.0,
            parameter_offset = 0.0,
            use_bias = False(),
            init_input = _W_ZZ,  # zero input weights
            init_reservoir = _W_ZZ,  # zero reservoir
            init_parameter = (rng, m, n) -> ones(Float32, m, n),  # ones for parameter
            init_state = _Z32)

        rng = MersenneTwister(0)
        ps = initialparameters(rng, cell)

        x = Float32[0, 0, 0]  # zero input
        param = Float32[2.0]  # non-zero parameter
        h0 = zeros(Float32, 3)

        (y_tuple, _) = cell(((x, param), (h0,)), ps, NamedTuple())
        y, _ = y_tuple

        # With identity activation: y = W_b * param = ones * 2.0 = [2, 2, 2]
        @test y ≈ Float32[2, 2, 2]
    end

    @testset "parameter offset" begin
        cell = ParameterAwareESNCell(3 => 3, identity;
            leak_coefficient = 1.0,
            parameter_coupling = 1.0,
            parameter_offset = 1.0,  # offset = 1.0
            use_bias = False(),
            init_input = _W_ZZ,
            init_reservoir = _W_ZZ,
            init_parameter = (rng, m, n) -> ones(Float32, m, n),
            init_state = _Z32)

        rng = MersenneTwister(0)
        ps = initialparameters(rng, cell)

        x = Float32[0, 0, 0]
        param = Float32[2.0]  # param - offset = 2.0 - 1.0 = 1.0
        h0 = zeros(Float32, 3)

        (y_tuple, _) = cell(((x, param), (h0,)), ps, NamedTuple())
        y, _ = y_tuple

        # y = W_b * (param - offset) = ones * 1.0 = [1, 1, 1]
        @test y ≈ Float32[1, 1, 1]
    end
end

@testset "ParameterAwareStatefulLayer tests" begin
    @testset "constructor & show" begin
        cell = ParameterAwareESNCell(3 => 5, tanh)
        sl = ParameterAwareStatefulLayer(cell)
        io = IOBuffer()
        show(io, sl)
        shown = String(take!(io))
        @test occursin("ParameterAwareStatefulLayer", shown)
        @test occursin("ParameterAwareESNCell", shown)
    end

    @testset "state management" begin
        cell = ParameterAwareESNCell(3 => 3, identity;
            leak_coefficient = 1.0,
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_parameter = _W_ZZ,
            init_state = init_state3)

        sl = ParameterAwareStatefulLayer(cell)
        rng = MersenneTwister(42)
        ps = initialparameters(rng, cell)
        st = initialstates(rng, sl)

        @test haskey(st, :cell)
        @test haskey(st, :carry)
        @test st.carry === nothing

        # Forward pass should update carry
        x = Float32[1, 2, 3]
        param = Float32[0.0]
        y, st2 = sl((x, param), ps, st)

        @test st2.carry !== nothing
        @test y ≈ x
    end
end

@testset "ParameterAwareESN model tests" begin
    @testset "constructor & parameter/state shapes" begin
        rng = MersenneTwister(42)
        in_dims, res_dims, out_dims = 3, 5, 4

        model = ParameterAwareESN(in_dims, res_dims, out_dims, identity;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_parameter = _W_I,
            init_bias = _O32,
            init_state = init_state3,
            leak_coefficient = 1.0)

        ps, st = setup(rng, model)

        @test haskey(ps, :reservoir)
        @test haskey(ps.reservoir, :input_matrix)
        @test haskey(ps.reservoir, :reservoir_matrix)
        @test haskey(ps.reservoir, :parameter_matrix)
        @test !haskey(ps.reservoir, :bias)
        @test size(ps.reservoir.input_matrix) == (res_dims, in_dims)
        @test size(ps.reservoir.reservoir_matrix) == (res_dims, res_dims)
        @test size(ps.reservoir.parameter_matrix) == (res_dims, 1)

        @test haskey(ps, :readout)
        @test haskey(ps.readout, :weight)
        @test size(ps.readout.weight) == (out_dims, res_dims)

        @test haskey(st, :reservoir)
        @test haskey(st, :states_modifiers)
        @test haskey(st, :readout)
    end

    @testset "forward pass with identity pipeline" begin
        rng = MersenneTwister(0)
        D = 3

        model = ParameterAwareESN(D, D, D, identity;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_parameter = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            leak_coefficient = 1.0)

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[1, 2, 3]
        param = Float32[0.0]

        y, st2 = model((x, param), ps, st)

        # Output is 2D with batch dimension (D, 1) for single sample
        @test size(y, 1) == D
        @test vec(y) ≈ x
        @test haskey(st2, :reservoir)
        @test haskey(st2, :states_modifiers)
        @test haskey(st2, :readout)
    end

    @testset "state_modifiers are applied" begin
        rng = MersenneTwister(2)
        D = 3

        model = ParameterAwareESN(D, D, D, identity;
            state_modifiers = (x -> 2.0f0 .* x,),
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_parameter = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            leak_coefficient = 1.0)

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x = Float32[1, 2, 3]
        param = Float32[0.0]
        y, _ = model((x, param), ps, st)
        @test y ≈ 2.0f0 .* x
    end

    @testset "collectstates" begin
        rng = MersenneTwister(3)
        D = 3
        T = 5

        model = ParameterAwareESN(D, D, D, identity;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_parameter = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            leak_coefficient = 1.0)

        ps, st = setup(rng, model)

        state_data = randn(Float32, D, T)
        param_data = randn(Float32, 1, T)

        states, st2 = collectstates(model, (state_data, param_data), ps, st)

        @test size(states) == (D, T)
        # With identity activation and zero reservoir/param weights,
        # states should equal input
        @test states ≈ state_data
    end

    @testset "train! and predict" begin
        rng = MersenneTwister(4)
        in_dims, res_dims, out_dims = 3, 10, 3
        T = 20

        model = ParameterAwareESN(in_dims, res_dims, out_dims, tanh;
            use_bias = False(),
            init_state = init_state3,
            leak_coefficient = 0.9)

        ps, st = setup(rng, model)

        # Create simple training data
        state_data = randn(Float32, in_dims, T)
        param_data = randn(Float32, 1, T)
        # Target: just use the input shifted by param contribution
        target_data = state_data

        ps2,
        st2 = train!(model, (state_data, param_data), target_data, ps, st,
            StandardRidge(1e-6); washout = 2)

        # Test prediction
        Y, st3 = predict(model, (state_data, param_data), ps2, st2)
        @test size(Y) == (out_dims, T)
    end

    @testset "predict auto-regressive" begin
        rng = MersenneTwister(5)
        D = 3
        steps = 10

        model = ParameterAwareESN(D, D, D, identity;
            use_bias = False(),
            init_input = _W_I,
            init_reservoir = _W_ZZ,
            init_parameter = _W_ZZ,
            init_bias = _O32,
            init_state = init_state3,
            leak_coefficient = 1.0)

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x0 = Float32[1, 2, 3]
        p0 = Float32(0.0)

        # Constant parameter schedule
        Y,
        st2 = predict(model, steps, ps, st;
            initialdata = x0,
            initialparam = p0,
            param_schedule = 0.0)

        @test size(Y) == (D, steps)
        # With identity and zero param, output should be same as input
        for t in 1:steps
            @test Y[:, t] ≈ x0
        end
    end

    @testset "predict with varying parameter" begin
        rng = MersenneTwister(6)
        D = 3
        steps = 5

        model = ParameterAwareESN(D, D, D, identity;
            use_bias = False(),
            init_input = _W_ZZ,  # zero input
            init_reservoir = _W_ZZ,
            init_parameter = (rng, m, n) -> ones(Float32, m, n),  # ones for param
            init_bias = _O32,
            init_state = init_state3,
            leak_coefficient = 1.0,
            parameter_coupling = 1.0,
            parameter_offset = 0.0)

        ps, st = setup(rng, model)
        ps = _with_identity_readout(ps; out_dims = D, in_dims = D)

        x0 = Float32[0, 0, 0]
        schedule = t -> Float32(t)  # parameter = step number

        Y,
        st2 = predict(model, steps, ps, st;
            initialdata = x0,
            initialparam = 0.0,
            param_schedule = schedule)

        @test size(Y) == (D, steps)
        # Output should be [t, t, t] for each step t
        for t in 1:steps
            @test Y[:, t] ≈ Float32[t, t, t]
        end
    end

    @testset "show" begin
        model = ParameterAwareESN(3, 10, 2, tanh;
            parameter_coupling = 0.5,
            parameter_offset = 1.0)
        io = IOBuffer()
        show(io, model)
        shown = String(take!(io))
        @test occursin("ParameterAwareESN", shown)
        @test occursin("reservoir", shown)
        @test occursin("readout", shown)
    end
end
