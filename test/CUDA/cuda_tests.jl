begin
    using LinearAlgebra
    using Random
    using Test
    using ReservoirComputing

    function cuda_tests_requested()
        raw = lowercase(strip(get(ENV, "RESERVOIRCOMPUTING_TEST_CUDA", "")))
        return raw in ("1", "true", "yes", "on")
    end

    cuda_available = Base.find_package("CUDA") !== nothing
    cuda_required = cuda_tests_requested()

    if !cuda_available
        if cuda_required
            @test cuda_available
        else
            @test_skip "CUDA is not available in this test environment"
        end
    else
        @eval using CUDA

        if !CUDA.functional()
            if cuda_required
                @test CUDA.functional()
            else
                @test_skip "CUDA is available but not functional"
            end
        else
            CUDA.allowscalar(false)

            to_device(x) = x isa AbstractArray ? CUDA.cu(x) : x
            to_device(nt::NamedTuple) = NamedTuple{keys(nt)}(map(to_device, values(nt)))
            to_device(t::Tuple) = map(to_device, t)

            dense_init(::Type{T}; value = 0.1) where {T} =
                (rng, dims...) -> fill(T(value), dims...)
            vector_init(::Type{T}; value = 0.0) where {T} =
                (rng, dim::Integer) -> fill(T(value), dim)

            @testset "LinearReadout accepts CuArray vectors and batches" begin
                rng = MersenneTwister(701)
                for use_bias in (false, true)
                    ro = LinearReadout(
                        3 => 2,
                        identity;
                        use_bias,
                        init_weight = dense_init(Float32; value = 0.5),
                        init_bias = vector_init(Float32; value = 0.25),
                    )
                    ps_cpu = initialparameters(rng, ro)
                    ps_gpu = to_device(ps_cpu)
                    st = NamedTuple()

                    x_cpu = Float32[1, 2, 3]
                    x_gpu = CUDA.cu(x_cpu)
                    y_gpu, st_gpu = ro(x_gpu, ps_gpu, st)
                    y_cpu, _ = ro(x_cpu, ps_cpu, st)
                    @test y_gpu isa CUDA.CuArray
                    @test Array(y_gpu) ≈ y_cpu
                    @test st_gpu === st

                    x_batch_cpu = hcat(x_cpu, 2 .* x_cpu)
                    x_batch_gpu = CUDA.cu(x_batch_cpu)
                    y_batch_gpu, st_batch_gpu = ro(x_batch_gpu, ps_gpu, st)
                    y_batch_cpu, _ = ro(x_batch_cpu, ps_cpu, st)
                    @test y_batch_gpu isa CUDA.CuArray
                    @test size(y_batch_gpu) == (2, 2)
                    @test Array(y_batch_gpu) ≈ y_batch_cpu
                    @test st_batch_gpu === st
                end
            end

            @testset "ESNCell explicit-state forward accepts CuArray inputs" begin
                rng = MersenneTwister(702)
                cell = ESNCell(
                    3 => 3,
                    identity;
                    use_bias = true,
                    init_input = dense_init(Float32; value = 1),
                    init_reservoir = dense_init(Float32; value = 0),
                    init_bias = vector_init(Float32; value = 0.25),
                    init_state = vector_init(Float32),
                    leak_coefficient = 1.0f0,
                )
                ps_cpu = initialparameters(rng, cell)
                ps_gpu = to_device(ps_cpu)
                x_cpu = Float32[1, 2, 3]
                h_cpu = zeros(Float32, 3)
                x_gpu = CUDA.cu(x_cpu)
                h_gpu = CUDA.cu(h_cpu)

                (y_gpu_tuple, st_gpu) = cell((x_gpu, (h_gpu,)), ps_gpu, NamedTuple())
                y_gpu, (h_next_gpu,) = y_gpu_tuple
                (y_cpu_tuple, _) = cell((x_cpu, (h_cpu,)), ps_cpu, NamedTuple())
                y_cpu, _ = y_cpu_tuple

                @test y_gpu isa CUDA.CuArray
                @test h_next_gpu isa CUDA.CuArray
                @test Array(y_gpu) ≈ y_cpu
                @test Array(h_next_gpu) ≈ y_cpu
                @test st_gpu === NamedTuple()
            end

            @testset "ReservoirChain collectstates and predict preserve CuArray storage" begin
                rng = MersenneTwister(703)
                rc = ReservoirChain(
                    identity,
                    Collect(),
                    LinearReadout(
                        3 => 2,
                        identity;
                        include_collect = false,
                        init_weight = dense_init(Float32; value = 0.5),
                    ),
                )
                ps_cpu, st = setup(rng, rc)
                ps_gpu = to_device(ps_cpu)
                data_cpu = reshape(Float32.(1:18), 3, 6) ./ 10
                data_gpu = CUDA.cu(data_cpu)

                states_gpu, st_states = collectstates(rc, data_gpu, ps_gpu, st)
                states_cpu, _ = collectstates(rc, data_cpu, ps_cpu, st)
                @test states_gpu isa CUDA.CuArray
                @test Array(states_gpu) ≈ states_cpu
                @test propertynames(st_states) == propertynames(st)

                pred_gpu, st_pred = predict(rc, data_gpu, ps_gpu, st)
                pred_cpu, _ = predict(rc, data_cpu, ps_cpu, st)
                @test pred_gpu isa CUDA.CuArray
                @test Array(pred_gpu) ≈ pred_cpu
                @test propertynames(st_pred) == propertynames(st)
            end
        end
    end
end
