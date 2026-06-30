using Test
using LinearAlgebra
using Random
using Statistics
using ReservoirComputing

# A small reproducible reservoir state cloud for algebra tests.
function sample_correlation(rng, N, L)
    X = randn(rng, N, L)
    return correlation_matrix(X)
end

@testset "Conceptors" begin
    rng = Xoshiro(2024)

    @testset "conceptor matrix" begin
        R = sample_correlation(rng, 20, 200)
        C = conceptor_matrix(R, 10.0)
        @test size(C) == (20, 20)
        @test C ≈ C'                                   # symmetric
        ev = eigvals(Symmetric(C))
        @test all(>=(-1e-10), ev)                      # PSD
        @test all(<=(1.0 + 1e-10), ev)                 # singular values ≤ 1
        @test conceptor_matrix(R, 1e8) ≈ I(20) atol = 1e-3
        @test norm(conceptor_matrix(R, 1e-6)) < 1e-6
        @test_throws ArgumentError conceptor_matrix(R, -1.0)
    end

    @testset "aperture adaptation" begin
        R = sample_correlation(rng, 15, 150)
        C = conceptor_matrix(R, 5.0)
        @test aperture_adapt(C, 1.0) ≈ C               # γ = 1 is identity
        # Prop 5: φ(φ(C, γ), β) = φ(C, γβ)
        @test aperture_adapt(aperture_adapt(C, 2.0), 3.0) ≈ aperture_adapt(C, 6.0)
        qs = [quota(aperture_adapt(C, g)) for g in (0.25, 1.0, 4.0, 16.0)]
        @test issorted(qs)
        P = aperture_adapt(C, Inf)
        @test P * P ≈ P atol = 1e-8                    # hardens to a projector
        @test norm(aperture_adapt(C, 0.0)) < 1e-8
        @test conceptor_matrix(R, 7.0) ≈ aperture_adapt(conceptor_matrix(R, 1.0), 7.0)
    end

    @testset "Boolean algebra" begin
        C = conceptor_matrix(sample_correlation(rng, 12, 120), 8.0)
        B = conceptor_matrix(sample_correlation(rng, 12, 120), 8.0)
        A = conceptor_matrix(sample_correlation(rng, 12, 120), 8.0)

        @test conceptor_not(conceptor_not(C)) ≈ C      # double negation
        @test conceptor_not(C) ≈ I(12) - C
        # De Morgan
        @test conceptor_or(C, B) ≈ conceptor_not(conceptor_and(conceptor_not(C), conceptor_not(B)))
        @test conceptor_and(C, B) ≈ conceptor_not(conceptor_or(conceptor_not(C), conceptor_not(B)))
        # commutativity
        @test conceptor_and(C, B) ≈ conceptor_and(B, C)
        @test conceptor_or(C, B) ≈ conceptor_or(B, C)
        # AND shrinks, OR grows (quota ordering)
        @test quota(conceptor_and(C, B)) <= quota(C) + 1e-8
        @test quota(C) <= quota(conceptor_or(C, B)) + 1e-8
        # associativity (holds for conceptors, unlike idempotence)
        @test conceptor_and(conceptor_and(C, B), A) ≈ conceptor_and(C, conceptor_and(B, A)) atol = 1e-6
        @test conceptor_or(conceptor_or(C, B), A) ≈ conceptor_or(C, conceptor_or(B, A)) atol = 1e-6
        # neutral / absorbing elements
        @test conceptor_and(C, Matrix(1.0I, 12, 12)) ≈ C atol = 1e-6
        @test conceptor_or(C, zeros(12, 12)) ≈ C atol = 1e-6
        # unexported infix sugar agrees
        and_op, or_op, not_op = ReservoirComputing.:∧, ReservoirComputing.:∨, ReservoirComputing.:¬
        @test and_op(C, B) ≈ conceptor_and(C, B)
        @test or_op(C, B) ≈ conceptor_or(C, B)
        @test not_op(C) ≈ conceptor_not(C)
    end

    @testset "wrapper / library" begin
        rng2 = Xoshiro(7)
        esn = ESN(1, 30, 1; use_bias = true)
        concept = Conceptor(esn)
        ps = initialparameters(rng2, concept)
        st = initialstates(rng2, concept)
        @test haskey(ps, :model)
        @test isempty(st.conceptors)
        C = conceptor_matrix(sample_correlation(rng2, 30, 100), 10.0)
        store_conceptor!(st, :a, C, 10.0)
        @test has_conceptor(st, :a)
        @test get_conceptor(st, :a) ≈ C
        st = set_active_conceptor(st, :a)
        @test active_conceptor(st) ≈ C
        @test_throws KeyError set_active_conceptor(st, :missing)
    end

    @testset "load! / generate roundtrip" begin
        rng3 = Xoshiro(99)
        bias_init(r, dims...) = 0.2f0 .* randn(r, Float32, dims...)
        input_init(r, dims...) = 1.5f0 .* randn(r, Float32, dims...)
        res_init(r, dims...) = rand_sparse(r, dims...; radius = 1.5f0, sparsity = 0.1f0)
        esn = ESN(1, 80, 1; use_bias = true, init_bias = bias_init,
            init_input = input_init, init_reservoir = res_init)
        concept = Conceptor(esn)
        ps = initialparameters(rng3, concept)
        st = initialstates(rng3, concept)

        n = 1:1000
        period = 8.8342522
        sig = reshape(Float32.(sin.(2pi .* n ./ period)), 1, :)
        ps, st = load!(rng3, concept, [:sine => sig], ps, st; aperture = 1000.0, washout = 400)
        @test has_conceptor(st, :sine)

        Y, X = generate(concept, ps, st; conceptor = :sine, steps = 200, washout = 300, rng = rng3)
        @test size(Y) == (1, 200)
        @test size(X) == (80, 200)
        y = vec(Y)
        best = minimum(
            sqrt(mean(abs2, y .- [s * sin(2pi * k / period + ph) for k in eachindex(y)])) / std(y)
            for ph in 0:0.02:2pi, s in (-1.0, 1.0)
        )
        @test best < 0.05
        @test count(>(0.5), conceptor_singular_values(get_conceptor(st, :sine))) < 80
    end

    @testset "morph_conceptor" begin
        rng4 = Xoshiro(5)
        esn = ESN(1, 25, 1; use_bias = true)
        concept = Conceptor(esn)
        st = initialstates(rng4, concept)
        Ca = conceptor_matrix(sample_correlation(rng4, 25, 100), 10.0)
        Cb = conceptor_matrix(sample_correlation(rng4, 25, 100), 10.0)
        store_conceptor!(st, :a, Ca, 10.0)
        store_conceptor!(st, :b, Cb, 10.0)
        @test morph_conceptor(st, (; a = 1.0, b = 0.0)) ≈ Ca
        @test morph_conceptor(st, (; a = 0.3, b = 0.7)) ≈ 0.3 .* Ca .+ 0.7 .* Cb
        @test_throws KeyError morph_conceptor(st, (; missing_name = 1.0))
    end
end
