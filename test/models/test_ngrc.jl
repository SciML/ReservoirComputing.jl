using Random
using ReservoirComputing
using LuxCore
using Static

@testset "NGRC" begin
    rng = MersenneTwister(1234)

    const_feature = x -> Float32[1.0]
    square_feature = x -> x .^ 2

    @testset "constructor & composition" begin
        ngrc = NGRC(
            3, 2; num_delays = 1, stride = 2,
            features = (const_feature, square_feature), include_input = True(),
            init_delay = zeros32, readout_activation = tanh
        )

        @test ngrc isa NGRC
        @test ngrc.reservoir isa DelayLayer
        @test ngrc.readout isa LinearReadout

        dl = ngrc.reservoir
        @test dl.in_dims == 3
        @test dl.num_delays == 1
        @test dl.stride == 2

        @test !isempty(ngrc.states_modifiers)
        first_mod = getfield(ngrc.states_modifiers, 1)
        @test first_mod isa NonlinearFeaturesLayer
    end

    @testset "initialparameters & initialstates" begin
        ngrc = NGRC(
            3, 2; num_delays = 1, features = (square_feature,),
            include_input = True()
        )

        ps = initialparameters(rng, ngrc)
        st = initialstates(rng, ngrc)

        @test hasproperty(ps, :reservoir)
        @test hasproperty(ps, :states_modifiers)
        @test hasproperty(ps, :readout)

        @test hasproperty(st, :reservoir)
        @test hasproperty(st, :states_modifiers)
        @test hasproperty(st, :readout)

        @test ps.readout.weight isa AbstractArray
    end

    @testset "forward pass: vector input" begin
        ngrc = NGRC(
            3, 2; num_delays = 1, features = (square_feature,),
            include_input = True()
        )

        ps, st = setup(rng, ngrc)

        x = rand(Float32, 3)
        y, st2 = ngrc(x, ps, st)

        @test size(y) == (2,)
        @test propertynames(st2) == propertynames(st)
    end

    @testset "forward pass: matrix input via collectstates" begin
        ngrc = NGRC(
            3, 2; num_delays = 1, features = (square_feature,),
            include_input = True()
        )

        ps, st = setup(rng, ngrc)

        X = rand(Float32, 3, 10)
        states, st2 = collectstates(ngrc, X, ps, st)

        @test size(states, 2) == size(X, 2)
        @test size(states, 1) == ngrc.readout.in_dims

        @test propertynames(st2) == propertynames(st)
    end

    @testset "simple 1D linear system learning" begin
        # x_{t+1} = a * x_t with a = 0.8
        a = 0.8f0
        T = 200
        x = zeros(Float32, T)
        x[1] = 1.0f0
        for t in 1:(T - 1)
            x[t + 1] = a * x[t]
        end

        X_in = reshape(x[1:(end - 1)], 1, :)
        Y_out = reshape(x[2:end], 1, :)

        ngrc = NGRC(
            1, 1; num_delays = 0, stride = 1, features = (),
            include_input = True(), ro_dims = 1
        )

        ps, st = setup(rng, ngrc)

        ps_tr, st_tr = train!(
            ngrc, X_in, Y_out, ps, st;
            train_method = StandardRidge(1.0e-6)
        )

        @test hasproperty(ps_tr, :readout)
        w = ps_tr.readout.weight
        @test size(w) == (1, 1)
        @test isapprox(w[1, 1], a; atol = 0.05)
    end
end
