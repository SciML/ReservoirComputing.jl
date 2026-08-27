# Context-aware `Extend` state-modifier regression tests.
begin
    using LuxCore: apply, setup
    using Random: MersenneTwister
    using ReservoirComputing
    using Test

    const _EXTEND_MODIFIER = Extend(Collect())
    const _EXTEND_INPUT = Float32[0.25, -0.5, 0.75]
    const _EXTEND_RESERVOIR_INIT = rand_sparse(; sparsity = 1.0)

    __zero_knowledge(x) = zeros(eltype(x), 1)

    function __check_extended_model(model; feature_slice = identity)
        ps, st = setup(MersenneTwister(311), model)
        features, _ = ReservoirComputing.__partial_apply(
            model, _EXTEND_INPUT, ps, st
        )
        selected_features = vec(feature_slice(features))
        @test selected_features[1:length(_EXTEND_INPUT)] == _EXTEND_INPUT
        output, _ = apply(model, _EXTEND_INPUT, ps, st)
        @test size(output, 1) == 1
        return nothing
    end

    @testset "Extend receives the model input in state_modifiers" begin
        models = (
            ESN(3, 5, 1; init_reservoir = _EXTEND_RESERVOIR_INIT,
                state_modifiers = (_EXTEND_MODIFIER,)),
            ResESN(3, 5, 1; init_reservoir = _EXTEND_RESERVOIR_INIT,
                state_modifiers = (_EXTEND_MODIFIER,)),
            ES2N(3, 5, 1; init_reservoir = _EXTEND_RESERVOIR_INIT,
                state_modifiers = (_EXTEND_MODIFIER,)),
            EuSN(3, 5, 1; init_reservoir = _EXTEND_RESERVOIR_INIT,
                state_modifiers = (_EXTEND_MODIFIER,)),
            EIESN(3, 5, 1; init_reservoir = _EXTEND_RESERVOIR_INIT,
                state_modifiers = (_EXTEND_MODIFIER,)),
            AdditiveEIESN(3, 5, 1; init_reservoir = _EXTEND_RESERVOIR_INIT,
                state_modifiers = (_EXTEND_MODIFIER,)),
            LIFESN(3, 5, 1; init_reservoir = _EXTEND_RESERVOIR_INIT,
                state_modifiers = (_EXTEND_MODIFIER,)),
            RMNESN(3, 4, 5, 1; init_reservoir = _EXTEND_RESERVOIR_INIT,
                state_modifiers = (_EXTEND_MODIFIER,)),
            RMNResESN(3, 4, 5, 1; init_reservoir = _EXTEND_RESERVOIR_INIT,
                state_modifiers = (_EXTEND_MODIFIER,)),
            InputDelayESN(3, 5, 1; num_delays = 1,
                init_reservoir = _EXTEND_RESERVOIR_INIT,
                state_modifiers = (_EXTEND_MODIFIER,)),
            StateDelayESN(3, 5, 1; num_delays = 1,
                init_reservoir = _EXTEND_RESERVOIR_INIT,
                state_modifiers = (_EXTEND_MODIFIER,)),
            DelayESN(3, 5, 1; num_input_delays = 1, num_state_delays = 1,
                init_reservoir = _EXTEND_RESERVOIR_INIT,
                state_modifiers = (_EXTEND_MODIFIER,)),
            DeepESN(3, 5, 1; depth = 1,
                init_reservoir = _EXTEND_RESERVOIR_INIT,
                state_modifiers = _EXTEND_MODIFIER),
            DeepESN(3, 5, 1; depth = 2,
                init_reservoir = _EXTEND_RESERVOIR_INIT,
                state_modifiers = _EXTEND_MODIFIER),
            NGRC(3, 1; num_delays = 1,
                state_modifiers = (_EXTEND_MODIFIER,)),
        )

        for model in models
            __check_extended_model(model)
        end

        hybrid = HybridESN(
            __zero_knowledge, 1, 3, 5, 1;
            init_reservoir = _EXTEND_RESERVOIR_INIT,
            state_modifiers = (_EXTEND_MODIFIER,)
        )
        __check_extended_model(hybrid; feature_slice = Base.Fix2(getindex, 2:9))
    end

    @testset "readout_in_dims supports custom dimension-changing modifiers" begin
        model = ESN(
            3, 5, 1;
            init_reservoir = _EXTEND_RESERVOIR_INIT,
            state_modifiers = (Pad(),),
            readout_in_dims = 6
        )
        ps, st = setup(MersenneTwister(311), model)
        output, _ = apply(model, _EXTEND_INPUT, ps, st)
        @test size(output) == (1, 1)

        @test_throws ArgumentError ESN(3, 5, 1; readout_in_dims = 0)
    end

    @testset "collectstates preserves each external input" begin
        model = ESN(
            3, 5, 1;
            init_reservoir = _EXTEND_RESERVOIR_INIT,
            state_modifiers = (_EXTEND_MODIFIER,)
        )
        ps, st = setup(MersenneTwister(311), model)
        data = Float32[1 2 3; 4 5 6; 7 8 9]
        states, _ = collectstates(model, data, ps, st)
        @test states[1:3, :] == data
    end

    @testset "training and prediction consume extended features" begin
        model = ESN(
            2, 4, 2;
            init_reservoir = _EXTEND_RESERVOIR_INIT,
            state_modifiers = (_EXTEND_MODIFIER,)
        )
        ps, st = setup(MersenneTwister(311), model)
        data = reshape(Float32.(1:16), 2, 8) ./ 10
        (ps_trained, st_trained), states = train(
            model, data, data, ps, st;
            objective = RidgeRegression(Float32, 1.0f-6),
            return_states = true
        )
        @test states[1:2, :] == data
        @test size(ps_trained.readout.weight) == (2, 6)

        teacher_forced, _ = predict(model, data, ps_trained, st_trained)
        autoregressive, _ = predict(
            model, 3, ps_trained, st_trained; initialdata = data[:, 1]
        )
        @test size(teacher_forced) == (2, 8)
        @test size(autoregressive) == (2, 3)
        @test all(isfinite, teacher_forced)
        @test all(isfinite, autoregressive)
    end
end
