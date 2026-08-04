include("../shared/generic_testsetup.jl")
using .GenericTestSetup

module InitializerOutputExtension
    import ReservoirComputing

    struct WrappedInitializerOutput{A}
        data::A
    end

    function ReservoirComputing.return_init_as(::Val{true}, output::Vector{Float64})
        return WrappedInitializerOutput(output)
    end
end

begin
    using Random
    using Static
    using ReservoirComputing
    using LuxCore: initialparameters, initialstates

    @testset "initializer output extension contract" begin
        output = [1.0, 2.0]
        wrapped = ReservoirComputing.return_init_as(Val(true), output)
        @test wrapped isa InitializerOutputExtension.WrappedInitializerOutput
        @test wrapped.data == output
        @test ReservoirComputing.return_init_as(Val(false), output) === output
    end

    function has_anykey(nt::NamedTuple, names)
        return any(Base.Fix1(haskey, nt), names)
    end

    function recurrent_cell_cases(::Type{T}, use_bias) where {T}
        matrix = dense_init(T)
        vector = vector_init(T)
        return (
            ESNCell(
                3 => 5;
                use_bias,
                init_input = matrix,
                init_reservoir = matrix,
                init_bias = vector,
                init_state = vector,
                leak_coefficient = T(0.7),
            ),
            ResESNCell(
                3 => 5;
                use_bias,
                init_input = matrix,
                init_reservoir = matrix,
                init_orthogonal = matrix,
                init_bias = vector,
                init_state = vector,
                alpha = T(0.8),
                beta = T(0.6),
            ),
            ES2NCell(
                3 => 5;
                use_bias,
                init_input = matrix,
                init_reservoir = matrix,
                init_orthogonal = matrix,
                init_bias = vector,
                init_state = vector,
                proximity = T(0.4),
            ),
            EuSNCell(
                3 => 5;
                use_bias,
                init_input = matrix,
                init_reservoir = matrix,
                init_bias = vector,
                init_state = vector,
                leak_coefficient = T(0.6),
                diffusion = T(0.2),
            ),
            ContinuousESNCell(
                3 => 5;
                tspan = (T(0), T(1)),
                use_bias,
                init_input = matrix,
                init_reservoir = matrix,
                init_bias = vector,
                init_state = vector,
            ),
        )
    end

    rng = MersenneTwister(11)
    for T in (Float32, Float64), use_bias in (False(), True())
        for cell in recurrent_cell_cases(T, use_bias)
            ps = initialparameters(rng, cell)
            @test eltype(ps.input_matrix) === T
            @test size(ps.input_matrix) == (5, 3)
            @test size(ps.reservoir_matrix) == (5, 5)
            @test has_anykey(ps, (:bias,)) == ReservoirComputing.known(use_bias)
            if haskey(ps, :bias)
                @test eltype(ps.bias) === T
                @test size(ps.bias) == (5,)
            end
        end

        memory_cells = (
            MemoryESNCell(
                (3, 4) => 5;
                use_bias,
                init_input = dense_init(T),
                init_reservoir = dense_init(T),
                init_memory = dense_init(T),
                init_bias = vector_init(T),
                init_state = vector_init(T),
                leak_coefficient = T(0.5),
            ),
            MemoryResESNCell(
                (3, 4) => 5;
                use_bias,
                init_input = dense_init(T),
                init_reservoir = dense_init(T),
                init_memory = dense_init(T),
                init_orthogonal = dense_init(T),
                init_bias = vector_init(T),
                init_state = vector_init(T),
                alpha = T(0.9),
                beta = T(0.3),
            ),
        )
        for cell in memory_cells
            ps = initialparameters(rng, cell)
            @test eltype(ps.input_matrix) === T
            @test size(ps.input_matrix) == (5, 3)
            @test size(ps.memory_matrix) == (5, 4)
            @test haskey(ps, :bias) == ReservoirComputing.known(use_bias)
        end

        ei_cells = (
            EIESNCell(
                3 => 5;
                use_bias,
                init_input = dense_init(T),
                init_reservoir = dense_init(T),
                init_bias = vector_init(T),
                init_state = vector_init(T),
                exc_recurrence_scale = T(0.8),
                inh_recurrence_scale = T(0.4),
            ),
            AdditiveEIESNCell(
                3 => 5;
                use_bias,
                init_input = dense_init(T),
                init_reservoir = dense_init(T),
                init_bias = vector_init(T),
                init_state = vector_init(T),
                input_activation = tanh,
                exc_recurrence_scale = T(0.8),
                inh_recurrence_scale = T(0.4),
            ),
        )
        for cell in ei_cells
            ps = initialparameters(rng, cell)
            @test eltype(ps.input_matrix) === T
            @test has_anykey(ps, (:bias_ex, :bias_inh, :bias_in)) ==
                ReservoirComputing.known(use_bias)
        end
    end

    for T in (Float32, Float64),
            use_bias in (False(), True()),
            include_collect in (False(), True())

        ro = LinearReadout(
            3 => 2;
            use_bias,
            include_collect,
            init_weight = dense_init(T),
            init_bias = vector_init(T),
        )
        ps = initialparameters(rng, ro)
        @test eltype(ps.weight) === T
        @test size(ps.weight) == (2, 3)
        @test haskey(ps, :bias) == ReservoirComputing.known(use_bias)

        x, x_view, x_batch = typed_inputs(T, 3)
        st = NamedTuple()
        @test size(first(ro(x, ps, st))) == (2,)
        @test size(first(ro(x_view, ps, st))) == (2,)
        @test size(first(ro(x_batch, ps, st))) == (2, 2)
    end

    for T in (Float32, Float64), include_input in (False(), True())
        square(x) = x .^ 2
        layer = NonlinearFeaturesLayer(square; include_input)
        x, x_view, x_batch = typed_inputs(T, 3)
        ps = initialparameters(rng, layer)
        st = initialstates(rng, layer)
        @test eltype(first(layer(x, ps, st))) === T
        @test eltype(first(layer(x_view, ps, st))) === T
    end
end

begin
    using LinearAlgebra
    using LuxCore
    using Random
    using ReservoirComputing
    using Static: True
    using Test

    struct ExternalESNCell <: ReservoirComputing.AbstractEchoStateNetworkCell
        in_dims::Int
        out_dims::Int
        init_input
        init_reservoir
        init_bias
        init_state
        use_bias
    end

    function (cell::ExternalESNCell)(
            input_and_carry::Tuple{<:AbstractArray, Tuple{<:AbstractArray}}, ps,
            st::NamedTuple
        )
        input, (carry,) = input_and_carry
        output = ps.input_matrix * input + ps.reservoir_matrix * carry + ps.bias
        return (output, (output,)), st
    end

    struct ExternalCollect <: ReservoirComputing.AbstractReservoirCollectionLayer end
    (layer::ExternalCollect)(input, ps, st::NamedTuple) = (input, st)
    LuxCore.initialparameters(::AbstractRNG, ::ExternalCollect) = NamedTuple()
    LuxCore.initialstates(::AbstractRNG, ::ExternalCollect) = NamedTuple()

    struct ExternalReadout <: ReservoirComputing.AbstractReservoirTrainableLayer end
    (layer::ExternalReadout)(input, ps, st::NamedTuple) = (2 .* input, st)
    LuxCore.initialparameters(::AbstractRNG, ::ExternalReadout) = NamedTuple()
    LuxCore.initialstates(::AbstractRNG, ::ExternalReadout) = NamedTuple()

    struct ExternalReservoirComputer <:
        ReservoirComputing.AbstractReservoirComputer{
            (:reservoir, :state_modifiers, :readout),
        }
        reservoir
        state_modifiers
        readout
    end

    struct ExternalEchoStateNetwork <:
        ReservoirComputing.AbstractEchoStateNetwork{
            (:reservoir, :state_modifiers, :readout),
        }
        reservoir
        state_modifiers
        readout
    end

    struct ExternalEncoding <: ReservoirComputing.AbstractInputEncoding end
    struct ExternalEncodingData <: ReservoirComputing.AbstractEncodingData end
    struct ExternalSolver <: ReservoirComputing.AbstractReservoirComputingSolver end

    @testset "Developer interface contracts" begin
        rng = MersenneTwister(44)
        matrix_init = (rng, m, n) -> Matrix{Float32}(I, m, n)
        vector_init = (rng, m) -> zeros(Float32, m)
        state_init = (rng, m, batch_size) -> zeros(Float32, m, batch_size)
        cell = ExternalESNCell(
            2, 2, matrix_init, matrix_init, vector_init, state_init, True()
        )

        cell_ps = LuxCore.initialparameters(rng, cell)
        cell_st = LuxCore.initialstates(rng, cell)
        @test keys(cell_ps) == (:input_matrix, :reservoir_matrix, :bias)
        @test size(cell_ps.input_matrix) == (2, 2)
        @test size(cell_ps.reservoir_matrix) == (2, 2)

        cell_output, cell_next_st = LuxCore.apply(cell, Float32[1, 2], cell_ps, cell_st)
        @test size(first(cell_output)) == (2, 1)
        @test vec(first(cell_output)) == Float32[1, 2]
        @test size(first(last(cell_output))) == (2, 1)
        @test vec(first(last(cell_output))) == Float32[1, 2]
        @test haskey(cell_next_st, :rng)

        reservoir = StatefulLayer(cell)
        components = (reservoir, (ExternalCollect(),), ExternalReadout())
        for Model in (ExternalReservoirComputer, ExternalEchoStateNetwork)
            model = Model(components...)
            model_ps, model_st = LuxCore.setup(rng, model)
            model_output, model_next_st = LuxCore.apply(model, Float32[3, 4], model_ps, model_st)
            @test size(model_output) == (2, 1)
            @test vec(model_output) == Float32[6, 8]
            @test keys(model_next_st) == (:reservoir, :state_modifiers, :readout)

            states, collected_st = collectstates(
                model, Float32[3 4; 4 5], model_ps, model_st
            )
            @test states == Float32[3 7; 4 9]
            @test keys(collected_st) == (:reservoir, :state_modifiers, :readout)
        end

        @test ExternalEncoding() isa ReservoirComputing.AbstractInputEncoding
        @test ExternalEncodingData() isa ReservoirComputing.AbstractEncodingData
        @test ExternalSolver() isa ReservoirComputing.AbstractReservoirComputingSolver
        @test_throws ArgumentError train(
            RidgeRegression(), ones(Float32, 2, 2), ones(Float32, 1, 2);
            solver = ExternalSolver()
        )
    end
end

begin
    using Random
    using Static
    using Test
    using ReservoirComputing

    rng = MersenneTwister(17)
    for T in (Float32, Float64), use_bias in (False(), True())
        models = (
            ESN(
                3,
                5,
                2;
                common_model_kwargs(T, use_bias)...,
                leak_coefficient = T(0.7),
                state_modifiers = (),
            ),
            ResESN(3, 5, 2; res_model_kwargs(T, use_bias)..., state_modifiers = ()),
            ES2N(3, 5, 2; es2n_model_kwargs(T, use_bias)..., state_modifiers = ()),
            EuSN(
                3,
                5,
                2;
                common_model_kwargs(T, use_bias)...,
                leak_coefficient = T(0.6),
                diffusion = T(0.2),
                state_modifiers = (),
            ),
        )
        for model in models
            run_model_smoke(Test, rng, model, T)
        end
    end
end

begin
    using Random
    using Static
    using Test
    using ReservoirComputing

    rng = MersenneTwister(17)
    for T in (Float32, Float64), use_bias in (False(), True())
        models = (
            AdditiveEIESN(
                3,
                5,
                2;
                common_model_kwargs(T, use_bias)...,
                input_activation = tanh,
                state_modifiers = (),
            ),
            EIESN(3, 5, 2; common_model_kwargs(T, use_bias)..., state_modifiers = ()),
        )
        for model in models
            run_model_smoke(Test, rng, model, T)
        end
    end
end

begin
    using Random
    using Static
    using Test
    using ReservoirComputing

    rng = MersenneTwister(17)
    for T in (Float32, Float64), use_bias in (False(), True())
        models = (
            InputDelayESN(
                3,
                5,
                2;
                common_model_kwargs(T, use_bias)...,
                num_delays = 1,
                stride = 1,
                state_modifiers = (),
            ),
            StateDelayESN(
                3,
                5,
                2;
                common_model_kwargs(T, use_bias)...,
                num_delays = 1,
                stride = 1,
                state_modifiers = (),
            ),
            DelayESN(
                3,
                5,
                2;
                common_model_kwargs(T, use_bias)...,
                num_input_delays = 1,
                num_state_delays = 1,
                input_stride = 1,
                state_stride = 1,
                state_modifiers = (),
            ),
        )
        for model in models
            run_model_smoke(Test, rng, model, T; batch = false)
        end
    end
end

begin
    using Random
    using Static
    using Test
    using ReservoirComputing

    rng = MersenneTwister(17)
    for T in (Float32, Float64), use_bias in (False(), True())
        models = (
            RMNESN(
                3,
                4,
                5,
                2;
                common_model_kwargs(T, use_bias)...,
                init_memory = dense_init(T),
                init_memory_input = dense_init(T),
                init_memory_reservoir = dense_init(T),
                init_memory_bias = vector_init(T),
                init_memory_state = dense_init(T),
                use_memory_bias = use_bias,
                state_modifiers = (),
            ),
            RMNResESN(
                3,
                4,
                5,
                2;
                res_model_kwargs(T, use_bias)...,
                init_memory = dense_init(T),
                init_memory_input = dense_init(T),
                init_memory_reservoir = dense_init(T),
                init_memory_bias = vector_init(T),
                init_memory_state = dense_init(T),
                use_memory_bias = use_bias,
                state_modifiers = (),
            ),
        )
        for model in models
            run_model_smoke(Test, rng, model, T)
        end
    end
end

begin
    using Random
    using Static
    using Test
    using ReservoirComputing

    rng = MersenneTwister(17)
    for T in (Float32, Float64), include_input in (False(), True())
        square(x) = x .^ 2
        ro_dims = ReservoirComputing.known(include_input) ? 12 : 6
        model = NGRC(
            3,
            2;
            num_delays = 1,
            stride = 1,
            features = (square,),
            include_input,
            init_delay = dense_init(T),
            ro_dims,
        )
        run_model_smoke(Test, rng, model, T; batch = false)
    end
end
