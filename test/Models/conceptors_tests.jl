begin
    using Test
    using LinearAlgebra
    using Random
    using Statistics
    using ReservoirComputing

    # A small reproducible reservoir state cloud for algebra tests.
    function sample_correlation(rng, state_dimension, sample_count)
        states = randn(rng, state_dimension, sample_count)
        return correlation_matrix(states)
    end

    @testset "Conceptors" begin
        rng = MersenneTwister(2024)

        @testset "conceptor matrix" begin
            correlation = sample_correlation(rng, 20, 200)
            conceptor = conceptor_matrix(correlation, 10.0)
            @test size(conceptor) == (20, 20)
            @test conceptor ≈ conceptor'
            eigenvalues = eigvals(Symmetric(conceptor))
            @test all(>=(-1.0e-10), eigenvalues)
            @test all(<=(1.0 + 1.0e-10), eigenvalues)
            @test conceptor_matrix(correlation, 1.0e8) ≈ I(20) atol = 1.0e-3
            @test norm(conceptor_matrix(correlation, 1.0e-6)) < 1.0e-6
            @test_throws ArgumentError conceptor_matrix(correlation, -1.0)
            @test_throws ArgumentError conceptor_matrix(correlation, Inf)
            @test_throws ArgumentError correlation_matrix(zeros(3, 0))

            for element_type in (Float32, Float64, BigFloat)
                typed_states = element_type.(randn(rng, 6, 30))
                typed_correlation = correlation_matrix(typed_states)
                typed_conceptor = conceptor_matrix(typed_correlation, 3)
                @test eltype(typed_correlation) == element_type
                @test eltype(typed_conceptor) == element_type
                @test typed_conceptor ≈ typed_conceptor'
                @test eltype(aperture_adapt(typed_conceptor, 2)) == element_type
            end
        end

        @testset "aperture adaptation" begin
            correlation = sample_correlation(rng, 15, 150)
            conceptor = conceptor_matrix(correlation, 5.0)
            @test aperture_adapt(conceptor, 1.0) ≈ conceptor
            # Repeated adaptations multiply their aperture factors.
            @test aperture_adapt(aperture_adapt(conceptor, 2.0), 3.0) ≈
                aperture_adapt(conceptor, 6.0)
            quotas = [
                quota(aperture_adapt(conceptor, factor)) for
                    factor in (0.25, 1.0, 4.0, 16.0)
            ]
            @test issorted(quotas)
            projector = aperture_adapt(conceptor, Inf)
            @test projector * projector ≈ projector atol = 1.0e-8
            @test norm(aperture_adapt(conceptor, 0.0)) < 1.0e-8
            @test conceptor_matrix(correlation, 7.0) ≈
                aperture_adapt(conceptor_matrix(correlation, 1.0), 7.0)
        end

        @testset "Boolean algebra" begin
            first_conceptor = conceptor_matrix(sample_correlation(rng, 12, 120), 8.0)
            second_conceptor = conceptor_matrix(sample_correlation(rng, 12, 120), 8.0)
            third_conceptor = conceptor_matrix(sample_correlation(rng, 12, 120), 8.0)

            @test conceptor_not(conceptor_not(first_conceptor)) ≈ first_conceptor
            @test conceptor_not(first_conceptor) ≈ I(12) - first_conceptor
            # De Morgan
            @test conceptor_or(first_conceptor, second_conceptor) ≈
                conceptor_not(
                conceptor_and(
                    conceptor_not(first_conceptor), conceptor_not(second_conceptor)
                )
            )
            @test conceptor_and(first_conceptor, second_conceptor) ≈
                conceptor_not(
                conceptor_or(
                    conceptor_not(first_conceptor), conceptor_not(second_conceptor)
                )
            )
            # commutativity
            @test conceptor_and(first_conceptor, second_conceptor) ≈
                conceptor_and(second_conceptor, first_conceptor)
            @test conceptor_or(first_conceptor, second_conceptor) ≈
                conceptor_or(second_conceptor, first_conceptor)
            # AND shrinks, OR grows (quota ordering)
            @test quota(conceptor_and(first_conceptor, second_conceptor)) <=
                quota(first_conceptor) + 1.0e-8
            @test quota(first_conceptor) <=
                quota(conceptor_or(first_conceptor, second_conceptor)) + 1.0e-8
            # associativity (holds for conceptors, unlike idempotence)
            @test conceptor_and(
                conceptor_and(first_conceptor, second_conceptor), third_conceptor
            ) ≈ conceptor_and(
                first_conceptor, conceptor_and(second_conceptor, third_conceptor)
            ) atol = 1.0e-6
            @test conceptor_or(
                conceptor_or(first_conceptor, second_conceptor), third_conceptor
            ) ≈ conceptor_or(
                first_conceptor, conceptor_or(second_conceptor, third_conceptor)
            ) atol = 1.0e-6
            # neutral / absorbing elements
            @test conceptor_and(first_conceptor, Matrix(1.0I, 12, 12)) ≈
                first_conceptor atol = 1.0e-6
            @test conceptor_or(first_conceptor, zeros(12, 12)) ≈
                first_conceptor atol = 1.0e-6

            # The generalized definition must also work for singular conceptors.
            first_singular_conceptor = Diagonal([0.8, 0.0, 0.4]) |> Matrix
            second_singular_conceptor = Diagonal([0.5, 0.7, 0.0]) |> Matrix
            @test conceptor_and(first_singular_conceptor, second_singular_conceptor) ≈
                Diagonal([1 / (1 / 0.8 + 1 / 0.5 - 1), 0, 0])
        end

        @testset "wrapper / library" begin
            rng2 = MersenneTwister(7)
            esn = ESN(1, 30, 1; use_bias = true)
            concept = Conceptor(esn)
            ps = initialparameters(rng2, concept)
            st = initialstates(rng2, concept)
            @test haskey(ps, :model)
            @test isempty(st.conceptors)
            conceptor = conceptor_matrix(sample_correlation(rng2, 30, 100), 10.0)
            new_st = store_conceptor(st, :first_pattern, conceptor, 10.0)
            # the input state is untouched; the returned state holds the conceptor
            @test !has_conceptor(st, :first_pattern)
            st = new_st
            @test has_conceptor(st, :first_pattern)
            @test get_conceptor(st, :first_pattern) ≈ conceptor
            st = set_active_conceptor(st, :first_pattern)
            @test active_conceptor(st) ≈ conceptor
            @test_throws KeyError set_active_conceptor(st, :missing)
        end

        @testset "loadpatterns / generate roundtrip" begin
            rng3 = MersenneTwister(99)
            bias_init(random_generator, dims...) =
                0.2f0 .* randn(random_generator, Float32, dims...)
            input_init(random_generator, dims...) =
                1.5f0 .* randn(random_generator, Float32, dims...)
            res_init(random_generator, dims...) =
                rand_sparse(random_generator, dims...; radius = 1.5f0, sparsity = 0.1f0)
            esn = ESN(
                1, 80, 1; use_bias = true, init_bias = bias_init,
                init_input = input_init, init_reservoir = res_init
            )
            concept = Conceptor(esn)
            ps = initialparameters(rng3, concept)
            st = initialstates(rng3, concept)

            sample_indices = 1:1000
            period = 8.8342522
            signal = reshape(Float32.(sin.(2pi .* sample_indices ./ period)), 1, :)
            @test_throws ArgumentError loadpatterns(
                rng3, concept, [:sine => signal], ps, st; washout = 0
            )
            ps, st = loadpatterns(
                rng3, concept, [:sine => signal], ps, st; aperture = 1000.0, washout = 400
            )
            @test has_conceptor(st, :sine)

            outputs, states = generate(
                concept, ps, st; conceptor = :sine, steps = 200, washout = 300, rng = rng3
            )
            @test size(outputs) == (1, 200)
            @test size(states) == (80, 200)
            output = vec(outputs)
            best = minimum(
                sqrt(
                        mean(
                            abs2,
                            output .- [
                                sign * sin(2pi * index / period + phase) for
                                index in eachindex(output)
                            ],
                        )
                    ) / std(output) for phase in 0:0.02:2pi, sign in (-1.0, 1.0)
            )
            @test best < 0.05
            @test count(>(0.5), conceptor_singular_values(get_conceptor(st, :sine))) < 80

            # aperture selection on the loaded reservoir
            att = attenuation(
                concept, ps, st; conceptor = :sine, steps = 200, washout = 100, rng = rng3
            )
            @test 0 <= att < 1
            states_driven, _ = collectstates(concept, signal, ps, st)
            driven_correlation = correlation_matrix(states_driven[:, 401:end])
            candidates = [1.0, 100.0, 1.0e4]
            best_aperture, attenuations = optimal_aperture(
                concept, driven_correlation, candidates, ps, st;
                steps = 100, washout = 50, rng = rng3
            )
            @test best_aperture in candidates
            @test length(attenuations) == 3
            @test all(>=(0), attenuations)
            @test attenuations[findfirst(==(best_aperture), candidates)] ==
                minimum(attenuations)
        end

        @testset "generate follows the wrapped cell's fields" begin
            rng5 = MersenneTwister(11)
            activation = x -> tanh(0.7 * x)
            leak = 0.6f0
            esn = ESN(1, 12, 1, activation; use_bias = true, leak_coefficient = leak)
            concept = Conceptor(esn)
            ps = initialparameters(rng5, concept)
            st = initialstates(rng5, concept)
            conceptor = conceptor_matrix(sample_correlation(rng5, 12, 60), 5.0)
            st = store_conceptor(st, :probe, conceptor, 5.0)
            init_state = 0.3f0 .* randn(rng5, Float32, 12)

            outputs, states = generate(
                concept, ps, st;
                conceptor = :probe, steps = 4, washout = 0, init_state = init_state
            )

            # manual recursion with the cell's own activation, leak, and bias
            recurrent = ps.model.reservoir.reservoir_matrix
            bias = ps.model.reservoir.bias
            readout_weights = ps.model.readout.weight
            conceptor32 = Float32.(conceptor)
            state = copy(init_state)
            for step in 1:4
                candidate = activation.(recurrent * state .+ bias)
                state = conceptor32 * ((1 - leak) .* state .+ leak .* candidate)
                @test states[:, step] ≈ state
                @test outputs[:, step] ≈ readout_weights * state
            end
        end

        @testset "model validation" begin
            rng6 = MersenneTwister(3)
            esn_mod = ESN(1, 10, 1; use_bias = true, state_modifiers = (NLAT1(),))
            concept_mod = Conceptor(esn_mod)
            ps_mod = initialparameters(rng6, concept_mod)
            st_mod = initialstates(rng6, concept_mod)
            signal_mod = reshape(Float32.(sin.(0.5 .* (1:50))), 1, :)
            @test_throws ArgumentError loadpatterns(
                rng6, concept_mod, [:s => signal_mod], ps_mod, st_mod; washout = 10
            )
            @test_throws ArgumentError generate(
                concept_mod, ps_mod, st_mod; conceptor = zeros(10, 10), steps = 5
            )
        end

        @testset "morph_conceptor" begin
            rng4 = MersenneTwister(5)
            esn = ESN(1, 25, 1; use_bias = true)
            concept = Conceptor(esn)
            st = initialstates(rng4, concept)
            first_conceptor = conceptor_matrix(sample_correlation(rng4, 25, 100), 10.0)
            second_conceptor = conceptor_matrix(sample_correlation(rng4, 25, 100), 10.0)
            st = store_conceptor(st, :first_pattern, first_conceptor, 10.0)
            st = store_conceptor(st, :second_pattern, second_conceptor, 10.0)
            @test morph_conceptor(st, (; first_pattern = 1.0, second_pattern = 0.0)) ≈
                first_conceptor
            @test morph_conceptor(st, (; first_pattern = 0.3, second_pattern = 0.7)) ≈
                0.3 .* first_conceptor .+ 0.7 .* second_conceptor
            # equivalent inputs: vector of pairs and dictionary
            expected = 0.3 .* first_conceptor .+ 0.7 .* second_conceptor
            @test morph_conceptor(st, [:first_pattern => 0.3, :second_pattern => 0.7]) ≈
                expected
            @test morph_conceptor(
                st, Dict(:first_pattern => 0.3, :second_pattern => 0.7)
            ) ≈ expected
            @test_throws KeyError morph_conceptor(st, (; missing_name = 1.0))
            @test_throws ArgumentError morph_conceptor(st, Pair{Symbol, Float64}[])
        end
    end
end
