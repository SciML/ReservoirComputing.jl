@testitem "array wrappers and vector type contracts" tags=[:array_types] setup=[GenericTestSetup] begin
    using LinearAlgebra
    using Random
    using SparseArrays
    using StaticArrays
    using ReservoirComputing

    rng = MersenneTwister(101)

    @testset "LinearReadout accepts dense, view, static, and sparse inputs" begin
        for T in (Float32, Float64), use_bias in (false, true)
            ro = LinearReadout(
                3 => 2;
                use_bias,
                init_weight = dense_init(T),
                init_bias = vector_init(T; value = 0.25),
            )
            ps = initialparameters(rng, ro)
            st = NamedTuple()

            x = T[1, 2, 3]
            x_view = view(hcat(x, 2 .* x), :, 1)
            x_static = SVector{3,T}(x)
            x_sparse = sparsevec([1, 3], T[1, 3], 3)
            x_batch = hcat(x, 2 .* x)
            x_batch_view = view(hcat(x_batch, 3 .* x), :, 1:2)
            x_sparse_batch = sparse(x_batch)

            for input in (x, x_view, x_static, x_sparse)
                y, st2 = ro(input, ps, st)
                @test size(y) == (2,)
                @test y ≈ ro(Array(input), ps, st)[1]
                @test st2 === st
            end

            for input in (x_batch, x_batch_view, x_sparse_batch)
                y, st2 = ro(input, ps, st)
                @test size(y) == (2, 2)
                @test y ≈ ro(Array(input), ps, st)[1]
                @test st2 === st
            end
        end
    end

    @testset "DelayLayer preserves dense and view input semantics" begin
        for T in (Float32, Float64)
            dl = DelayLayer(3; num_delays = 2, stride = 1, init_delay = dense_init(T; value = 0))
            ps = initialparameters(rng, dl)

            x = T[1, 2, 3]
            x_view = view(hcat(x, 2 .* x), :, 1)
            y, st = dl(x, ps, initialstates(rng, dl))
            y_view, _ = dl(x_view, ps, initialstates(rng, dl))

            @test y ≈ y_view
            @test eltype(y) === T
            @test size(st.history) == (3, 2)
        end
    end

    @testset "collectstates accepts dense, view, and sparse sequence matrices" begin
        for T in (Float32, Float64)
            rc = ReservoirChain(identity, Collect(), LinearReadout(3 => 2; include_collect = false))
            ps, st = setup(rng, rc)
            data = reshape(T.(1:18), 3, 6) ./ T(10)
            data_view = view(hcat(data, data), :, 1:6)
            data_sparse = sparse(data)

            states, _ = collectstates(rc, data, ps, st)
            states_view, _ = collectstates(rc, data_view, ps, st)
            states_sparse, _ = collectstates(rc, data_sparse, ps, st)

            @test states ≈ data
            @test states_view ≈ data
            @test states_sparse ≈ data
            @test eltype(states) === T
            @test eltype(states_view) === T
            @test eltype(states_sparse) === T
        end
    end
end

@testitem "train predict workflow type contracts" tags=[:workflows] setup=[GenericTestSetup] begin
    using LinearAlgebra
    using Random
    using Static
    using ReservoirComputing

    function linear_targets(data)
        y1 = data[1, :] .+ 2 .* data[2, :]
        y2 = data[3, :] .- data[1, :]
        return vcat(
            reshape(y1, 1, length(y1)),
            reshape(y2, 1, length(y2)),
        )
    end

    function workflow_chain(::Type{T}) where {T}
        init_weight(rng, out, in) = zeros(T, out, in)
        return ReservoirChain(
            identity,
            Collect(),
            LinearReadout(3 => 2; include_collect = false, init_weight),
        )
    end

    @testset "ReservoirChain train! updates readout and predict matches targets" begin
        for T in (Float32, Float64)
            rng = MersenneTwister(202)
            rc = workflow_chain(T)
            ps, st = setup(rng, rc)
            data = reshape(T.(1:24), 3, 8) ./ T(10)
            data_view = view(hcat(data, data), :, 1:8)
            target = linear_targets(data)

            raw_states, _ = collectstates(rc, data_view, ps, st)
            @test raw_states ≈ data
            @test eltype(raw_states) === T

            (ps_trained, st_trained), states = train!(
                rc,
                data_view,
                target,
                ps,
                st,
                StandardRidge(T, T(0));
                return_states = true,
            )
            @test states ≈ data
            @test eltype(states) === T
            @test size(ps_trained.layer_3.weight) == (2, 3)
            @test eltype(ps_trained.layer_3.weight) === T

            teacher_forced, st_pred = predict(rc, data_view, ps_trained, st_trained)
            @test teacher_forced ≈ target
            @test eltype(teacher_forced) === T
            @test propertynames(st_pred) == propertynames(st_trained)
        end
    end

    @testset "ESN workflow trains and predicts across eltypes" begin
        for T in (Float32, Float64), use_bias in (False(), True())
            rng = MersenneTwister(303)
            model = ESN(
                2,
                4,
                2,
                identity;
                use_bias,
                init_input = dense_init(T; value = 0.2),
                init_reservoir = dense_init(T; value = 0),
                init_bias = vector_init(T; value = 0.1),
                init_state = dense_init(T; value = 0),
                leak_coefficient = T(1),
                state_modifiers = (),
            )
            ps, st = setup(rng, model)
            data = reshape(T.(1:16), 2, 8) ./ T(10)
            target = copy(data)

            (ps_trained, st_trained), states = train!(
                model,
                data,
                target,
                ps,
                st,
                StandardRidge(T, T(1.0e-6));
                return_states = true,
            )
            @test size(states) == (4, 8)
            @test eltype(states) === T
            @test size(ps_trained.readout.weight) == (2, 4)
            @test eltype(ps_trained.readout.weight) === T

            teacher_forced, _ = predict(model, data, ps_trained, st_trained)
            autoregressive, _ = predict(model, 3, ps_trained, st_trained; initialdata = data[:, 1])

            @test size(teacher_forced) == size(target)
            @test size(autoregressive) == (2, 3)
            @test eltype(teacher_forced) === T
            @test eltype(autoregressive) === T
            @test all(isfinite, teacher_forced)
            @test all(isfinite, autoregressive)
        end
    end
end
