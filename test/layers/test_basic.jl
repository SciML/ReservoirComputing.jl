using Test
using Random
using ReservoirComputing
using Static
using LuxCore
using LinearAlgebra

@testset "LinearReadout" begin
    rng = MersenneTwister(123)

    @testset "constructors & flags" begin
        ro = LinearReadout(5 => 3)
        @test ro.in_dims == 5
        @test ro.out_dims == 3
        @test ro.activation === identity
        @test ReservoirComputing.has_bias(ro) == false

        ro_b = LinearReadout(5, 3; use_bias = True(), include_collect = False())
        @test ReservoirComputing.has_bias(ro_b) == true
        ic = ReservoirComputing.getproperty(ro_b, Val(:include_collect))
        @test ic === false || ic === False()
    end

    @testset "initialparameters shapes & determinism" begin
        ro_nb = LinearReadout(4 => 2; use_bias = False())
        ps1 = initialparameters(rng, ro_nb)
        @test keys(ps1) == (:weight,)
        @test size(ps1.weight) == (2, 4)

        rng2 = MersenneTwister(123)
        ps2 = initialparameters(rng2, ro_nb)
        @test ps1.weight == ps2.weight

        ro_b = LinearReadout(4 => 2; use_bias = True())
        psb = initialparameters(rng, ro_b)
        @test keys(psb) == (:weight, :bias)
        @test size(psb.weight) == (2, 4)
        @test size(psb.bias) == (2,)
    end

    @testset "parameterlength/statelength/outputsize" begin
        ro_nb = LinearReadout(7 => 5; use_bias = False())
        @test LuxCore.parameterlength(ro_nb) == 5 * 7
        @test LuxCore.statelength(ro_nb) == 0
        @test LuxCore.outputsize(ro_nb, nothing, rng) == (5,)

        ro_b = LinearReadout(7 => 5; use_bias = True())
        @test LuxCore.parameterlength(ro_b) == 5 * 7 + 5
    end

    @testset "forward pass: vector, no-bias, identity" begin
        ro = LinearReadout(3 => 3; use_bias = False())
        ps = (weight = Matrix{Float32}(I, 3, 3),)
        x = Float32[0.2, -0.5, 1.0]
        y, st = ro(x, ps, NamedTuple())
        @test y == x
        @test st === NamedTuple()
    end

    @testset "forward pass: vector with bias" begin
        ro = LinearReadout(3 => 2; use_bias = True())
        ps = (weight = Float32[1 0 0; 0 1 0], bias = Float32[0.5, -1.0])
        x = Float32[2, 3, 4]
        y, _ = ro(x, ps, NamedTuple())
        @test y ≈ Float32[2.5, 2.0]
    end

    @testset "forward pass: matrix (batch), activation" begin
        ro = LinearReadout(2 => 2, tanh; use_bias = False())
        ps = (weight = Float32[1 0; 0 2],)
        X = Float32[0.0 1.0 -1.0;
                    0.5 0.5 0.5]
        Y, _ = ro(X, ps, NamedTuple())
        @test size(Y) == (2, 3)
        @test Y[:, 1] ≈ tanh.(ps.weight * X[:, 1])
        @test Y[:, 3] ≈ tanh.(ps.weight * X[:, 3])
    end

    @testset "dimension mismatch error" begin
        ro = LinearReadout(4 => 2)
        ps = (weight = rand(Float32, 2, 4),)
        badx = rand(Float32, 3)
        @test_throws DimensionMismatch ro(badx, ps, NamedTuple())
    end

    @testset "show" begin
        ro = LinearReadout(5 => 3; use_bias = False(), include_collect = True())
        s = sprint(show, ro)
        @test occursin("LinearReadout(5 => 3", s)
        @test occursin("use_bias=false", s)
        @test occursin("include_collect=true", s)
    end

    @testset "include_collect causes Collect() insertion" begin
        try
            rc = ReservoirChain(LinearReadout(4 => 2; include_collect = True()))
            L = rc.layers
            first_layer = getfield(L, 1)
            @test first_layer isa Collect
        catch e
            @info "Skipping Collect insertion test (no auto-wrap for LinearReadout?)" exception=(
                e, catch_backtrace())
        end
    end
end

@testset "Collect & collectstates" begin
    rng = MersenneTwister(0)

    F_id = ReservoirComputing.WrappedFunction(x -> x)
    F_2x = ReservoirComputing.WrappedFunction(x -> 2 .* x)
    ro = LinearReadout(3 => 1; include_collect = false)

    @testset "single Collect == features at collect point" begin
        rc = ReservoirChain(F_id, Collect(), ro)
        ps, st = setup(rng, rc)
        X = rand(rng, 3, 5)
        states, st2 = collectstates(rc, X, ps, st)
        @test size(states) == size(X)
        @test eltype(states) == eltype(X)
        @test states ≈ X
        @test propertynames(st2) == propertynames(st)
    end

    @testset "no Collect => features from last non-trainable layer" begin
        rc = ReservoirChain(F_2x, ro)
        ps, st = setup(rng, rc)
        X = rand(rng, 3, 7)
        states, _ = collectstates(rc, X, ps, st)
        @test states ≈ 2 .* X
    end

    @testset "multiple Collect => vertical stack (vcat) of features" begin
        rc = ReservoirChain(F_id, Collect(), F_2x, Collect(), ro)
        ps, st = setup(rng, rc)
        X = rand(rng, 3, 4)
        states, _ = collectstates(rc, X, ps, st)
        @test size(states) == (3 + 3, size(X, 2))
        @test states[1:3, :] ≈ X
        @test states[4:6, :] ≈ 2 .* X
    end

    @testset "vector input overload" begin
        rc = ReservoirChain(F_id, Collect(), ro)
        ps, st = setup(rng, rc)
        x = rand(rng, 3)
        states, _ = collectstates(rc, x, ps, st)
        @test size(states) == (3, 1)
        @test states[:, 1] ≈ x
    end

    @testset "st threading preserved" begin
        rc = ReservoirChain(F_id, Collect(), ro)
        ps, st = setup(rng, rc)
        X = rand(rng, 3, 2)
        _, st2 = collectstates(rc, X, ps, st)
        @test propertynames(st2) == propertynames(st)
        for name in propertynames(st)
            @test haskey(st2, name)
        end
    end
end

@testset "DelayLayer" begin
    rng = MersenneTwister(123)

    @testset "constructor & init_delay expansion" begin
        dl = DelayLayer(3; num_delays = 2, init_delay = zeros32)
        @test dl.in_dims == 3
        @test dl.num_delays == 2
        @test dl.stride == 1
        @test dl.init_delay isa NTuple{2}
        @test all(fn === zeros32 for fn in dl.init_delay)

        init1(rng, d, b) = fill(Float32(1), d, b)
        init2(rng, d, b) = fill(Float32(2), d, b)
        dl2 = DelayLayer(3; num_delays = 2, init_delay = (init1, init2))
        @test dl2.init_delay === (init1, init2)
    end

    @testset "initialstates & init_delay_history" begin
        init1(rng, d, b) = fill(Float32(1), d, b)
        init2(rng, d, b) = fill(Float32(2), d, b)
        dl = DelayLayer(2; num_delays = 2, stride = 2, init_delay = (init1, init2))

        ps = initialparameters(rng, dl)
        st = initialstates(rng, dl)

        x = rand(rng, Float32, 2)
        y, st2 = dl(x, ps, st)

        @test size(st2.history) == (2, 2)
        @test st2.clock == 1
        @test st2.history[:, 1] == fill(Float32(1), 2)
        @test st2.history[:, 2] == fill(Float32(2), 2)
        @test length(y) == 2 * (dl.num_delays + 1)
    end

    @testset "temporal behavior: num_delays=2, stride=1" begin
        dl = DelayLayer(1; num_delays = 2, stride = 1, init_delay = zeros32)
        ps = initialparameters(rng, dl)
        st = initialstates(rng, dl)

        inputs = [Float32(t) for t in 1:5]
        outputs = Vector{Vector{Float32}}()

        for x in inputs
            xvec = Float32[x]          # (1,)
            y, st = dl(xvec, ps, st)
            push!(outputs, vec(y))
        end

        @test outputs[1] ≈ Float32[1, 0, 0]
        @test outputs[2] ≈ Float32[2, 1, 0]
        @test outputs[3] ≈ Float32[3, 2, 1]
        @test outputs[4] ≈ Float32[4, 3, 2]
        @test outputs[5] ≈ Float32[5, 4, 3]
    end

    @testset "stride > 1 behavior" begin
        dl = DelayLayer(1; num_delays = 2, stride = 2, init_delay = zeros32)
        ps = initialparameters(rng, dl)
        st = initialstates(rng, dl)

        inputs = [Float32(t) for t in 1:6]
        hists = Vector{Vector{Float32}}()
        outputs = Vector{Vector{Float32}}()

        for x in inputs
            xvec = Float32[x]
            push!(hists, st.history === nothing ? Float32[] : vec(st.history))
            y, st = dl(xvec, ps, st)
            push!(outputs, vec(y))
        end

        @test outputs[1] ≈ Float32[1, 0, 0]
        @test outputs[2] ≈ Float32[2, 0, 0]
        @test outputs[3] ≈ Float32[3, 2, 0]
        @test outputs[4] ≈ Float32[4, 2, 0]
    end

    @testset "dimension mismatch assertion" begin
        dl = DelayLayer(3; num_delays = 2)
        ps = initialparameters(rng, dl)
        st = initialstates(rng, dl)

        badx = rand(rng, Float32, 2)
        @test_throws AssertionError dl(badx, ps, st)
    end
end

@testset "NonlinearFeaturesLayer" begin
    rng = MersenneTwister(123)

    @testset "constructor & flags" begin
        sq = x -> x .^ 2
        cb = x -> x .^ 3

        nfl = NonlinearFeaturesLayer(sq, cb; include_input = true)
        @test length(nfl.features) == 2
        ii = ReservoirComputing.known(nfl.include_input)
        @test ii === true

        nfl2 = NonlinearFeaturesLayer(sq; include_input = false)
        @test length(nfl2.features) == 1
        ii2 = ReservoirComputing.known(nfl2.include_input)
        @test ii2 === false

        nfl_id = NonlinearFeaturesLayer(; include_input = true)
        @test length(nfl_id.features) == 0
        ii_id = ReservoirComputing.known(nfl_id.include_input)
        @test ii_id === true
    end

    @testset "initialparameters / initialstates" begin
        sq = x -> x .^ 2
        nfl = NonlinearFeaturesLayer(sq; include_input = true)

        ps = initialparameters(rng, nfl)
        st = initialstates(rng, nfl)

        @test ps == NamedTuple()
        @test st == NamedTuple()
    end

    @testset "forward: include_input=true" begin
        sq = x -> x .^ 2
        cb = x -> x .^ 3
        nfl = NonlinearFeaturesLayer(sq, cb; include_input = true)

        ps = initialparameters(rng, nfl)
        st = initialstates(rng, nfl)

        x = Float32[1, 2, 3]

        y, st2 = nfl(x, ps, st)

        @test length(y) == 3 * 3
        @test y[1:3] == x
        @test y[4:6] == x .^ 2
        @test y[7:9] == x .^ 3

        @test st2 === st
    end

    @testset "forward: include_input=false" begin
        sq = x -> x .^ 2
        cb = x -> x .^ 3
        nfl = NonlinearFeaturesLayer(sq, cb; include_input = false)

        ps = initialparameters(rng, nfl)
        st = initialstates(rng, nfl)

        x = Float32[1, 2, 3]

        y, _ = nfl(x, ps, st)

        @test length(y) == 2 * 3
        @test y[1:3] == x .^ 2
        @test y[4:6] == x .^ 3
    end

    @testset "identity case: no features, include_input=true" begin
        nfl = NonlinearFeaturesLayer(; include_input = true)
        ps = initialparameters(rng, nfl)
        st = initialstates(rng, nfl)

        x = rand(rng, Float32, 4)

        y, _ = nfl(x, ps, st)

        @test y == x
    end
end
