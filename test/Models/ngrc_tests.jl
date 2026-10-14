# Next-generation reservoir model contracts.
begin
    using Random
    using ReservoirComputing
    using LuxCore
    import LuxCore: initialparameters, initialstates, setup
    using WeightInitializers: zeros32
    using Static

    @testset "NGRC" begin
        rng = MersenneTwister(1234)

        const_feature = x -> Float32[1.0]
        square_feature = x -> x .^ 2

        @testset "constructor & composition" begin
            ngrc = NGRC(
                3,
                2;
                num_delays = 1,
                stride = 2,
                features = (const_feature, square_feature),
                include_input = True(),
                init_delay = zeros32,
                readout_activation = tanh,
            )

            @test ngrc isa NGRC
            @test ngrc.reservoir isa DelayLayer
            @test ngrc.readout isa LinearReadout

            dl = ngrc.reservoir
            @test dl.in_dims == 3
            @test dl.num_delays == 1
            @test dl.stride == 2

            @test !isempty(ngrc.state_modifiers)
            first_mod = getfield(ngrc.state_modifiers, 1)
            @test first_mod isa NonlinearFeaturesLayer
        end

        @testset "initialparameters & initialstates" begin
            ngrc = NGRC(
                3,
                2;
                num_delays = 1,
                features = (square_feature,),
                include_input = True(),
            )

            ps = initialparameters(rng, ngrc)
            st = initialstates(rng, ngrc)

            @test hasproperty(ps, :reservoir)
            @test hasproperty(ps, :state_modifiers)
            @test hasproperty(ps, :readout)

            @test hasproperty(st, :reservoir)
            @test hasproperty(st, :state_modifiers)
            @test hasproperty(st, :readout)

            @test ps.readout.weight isa AbstractArray
        end

        @testset "forward pass: vector input" begin
            ngrc = NGRC(
                3,
                2;
                num_delays = 1,
                features = (square_feature,),
                include_input = True(),
            )

            ps, st = setup(rng, ngrc)

            x = rand(Float32, 3)
            y, st2 = ngrc(x, ps, st)

            @test size(y) == (2,)
            @test propertynames(st2) == propertynames(st)
        end

        @testset "forward pass: matrix input via collectstates" begin
            ngrc = NGRC(
                3,
                2;
                num_delays = 1,
                features = (square_feature,),
                include_input = True(),
            )

            ps, st = setup(rng, ngrc)

            X = rand(Float32, 3, 10)
            states, st2 = collectstates(ngrc, X, ps, st)

            @test size(states, 2) == size(X, 2)
            @test size(states, 1) == ngrc.readout.in_dims

            @test propertynames(st2) == propertynames(st)
        end

        @testset "resetcarry! restores the initial delay state" begin
            ngrc = NGRC(
                2, 1; num_delays = 2, features = (square_feature,), include_input = True(),
            )
            ps, st = setup(MersenneTwister(2), ngrc)
            X = rand(MersenneTwister(3), Float32, 2, 30)
            Y = rand(MersenneTwister(4), Float32, 1, 30)
            ps_tr, st_tr = train(
                ngrc, X, Y, ps, st; objective = RidgeRegression(1.0e-4),
            )
            Xt = rand(MersenneTwister(5), Float32, 2, 10)
            _, st_new = setup(MersenneTwister(2), ngrc)
            st_fresh = merge(st_tr, (reservoir = st_new.reservoir,))
            y_fresh, _ = predict(ngrc, Xt, ps_tr, st_fresh)

            y_carried, _ = predict(ngrc, Xt, ps_tr, deepcopy(st_tr))
            @test y_carried != y_fresh

            st_reset = resetcarry!(rng, ngrc, deepcopy(st_tr))
            @test st_reset.reservoir.history === nothing
            @test st_reset.reservoir.clock == 0
            @test st_reset.readout == st_tr.readout
            y_reset, _ = predict(ngrc, Xt, ps_tr, st_reset)
            @test y_reset == y_fresh

            st_zero = resetcarry!(rng, ngrc, deepcopy(st_tr); init_carry = zeros32)
            y_zero, _ = predict(ngrc, Xt, ps_tr, st_zero)
            @test y_zero == y_fresh

            ones_init = (rng, dims...) -> ones(Float32, dims...)
            st_ones = resetcarry!(rng, ngrc, deepcopy(st_tr); init_carry = ones_init)
            @test st_ones.reservoir.history == ones(Float32, 2, 2)
            @test st_ones.reservoir.clock == 0
            _, st_ones_ps = resetcarry!(rng, ngrc, ps_tr, st_tr; init_carry = ones_init)
            @test st_ones_ps.reservoir.history == ones(Float32, 2, 2)
            y_ones, _ = predict(ngrc, Xt, ps_tr, st_ones)
            @test size(y_ones) == size(y_fresh)
        end

        @testset "resetcarry! resets DelayLayers among the state modifiers" begin
            ngrc = NGRC(
                2, 1; num_delays = 2, include_input = True(),
                state_modifiers = (DelayLayer(6; num_delays = 2),),
            )
            ps, st0 = setup(MersenneTwister(2), ngrc)
            X1 = rand(MersenneTwister(3), Float32, 2, 10)
            X2 = rand(MersenneTwister(4), Float32, 2, 10)
            run = st -> first(predict(ngrc, X2, ps, deepcopy(st)))
            ref = run(st0)
            _, st1 = predict(ngrc, X1, ps, deepcopy(st0))
            @test run(st1) != ref

            st_reset = resetcarry!(rng, ngrc, deepcopy(st1))
            @test st_reset.state_modifiers[end].history === nothing
            @test st_reset.state_modifiers[end].clock == 0
            @test run(st_reset) == ref

            st_zero = resetcarry!(rng, ngrc, deepcopy(st1); init_carry = zeros32)
            @test run(st_zero) == ref

            ones_init = (rng, dims...) -> ones(Float32, dims...)
            st_ones = resetcarry!(rng, ngrc, deepcopy(st1); init_carry = ones_init)
            @test st_ones.state_modifiers[end].history == ones(Float32, 6, 2)
            @test st_ones.state_modifiers[end].clock == 0
            @test st_ones.readout == st1.readout
            @test run(st_ones) != ref
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
                1,
                1;
                num_delays = 0,
                stride = 1,
                features = (),
                include_input = True(),
                ro_dims = 1,
            )

            ps, st = setup(rng, ngrc)

            ps_tr, st_tr = train(
                ngrc, X_in, Y_out, ps, st;
                objective = RidgeRegression(1.0e-6),
            )

            @test hasproperty(ps_tr, :readout)
            w = ps_tr.readout.weight
            @test size(w) == (1, 1)
            @test isapprox(w[1, 1], a; atol = 0.05)
        end
    end


    chebT(n, x) =
        n == 0 ? one(x) :
        n == 1 ? x :
        n == 2 ? 2x^2 - 1 : n == 3 ? 4x^3 - 3x : error("extend reference if needed")

    @testset "chebyshev_monomials basic functionality" begin
        x = [2.0, 3.0]
        feats = chebyshev_monomials(x; degrees = 1:1)
        @test feats == [2.0, 3.0]  # T1(x) = x
        feats2 = chebyshev_monomials(x; degrees = 2:2)
        T2_x1 = 2x[1]^2 - 1
        T2_x2 = 2x[2]^2 - 1
        @test feats2 == [T2_x1, T2_x1 * T2_x2, T2_x2]
        feats12 = chebyshev_monomials(x; degrees = 1:2)
        @test feats12 == vcat(feats, feats2)
    end


    @testset "degree skipping and ordering" begin
        x = [1.5, -0.5, 0.25]
        feats = chebyshev_monomials(x; degrees = [0, 1])
        @test length(feats) == 3
        featsA = chebyshev_monomials(x; degrees = 1:2)
        featsB = chebyshev_monomials(x; degrees = 1:2)
        @test featsA == featsB
    end


    @testset "higher-degree correctness" begin
        x = [0.3, -0.7]
        feats = chebyshev_monomials(x; degrees = 3:3)
        T3_x1 = 4x[1]^3 - 3x[1]
        T3_x2 = 4x[2]^3 - 3x[2]
        @test feats ≈ [T3_x1, T3_x1 * T3_x2, T3_x2]
    end

    @testset "distinct-variable subset contract" begin
        x = [0.2, 0.4, 0.6]
        transformed = chebT.(2, x)
        expected = [
            transformed[1],
            transformed[1] * transformed[2],
            transformed[1] * transformed[3],
            transformed[2],
            transformed[2] * transformed[3],
            transformed[3],
        ]
        @test chebyshev_monomials(x; degrees = 2:2) ≈ expected
        @test chebyshev_monomials(x; degrees = (degree for degree in 1:2)) ≈
            vcat(x, expected)
    end

    @testset "type stability" begin
        x = Float32[1, 2]
        feats = chebyshev_monomials(x; degrees = 1:2)
        @test eltype(feats) == Float32
    end


    @testset "single-variable behavior" begin
        x = [2.0]
        feats = chebyshev_monomials(x; degrees = 1:3)
        @test feats == [
            2.0,                    # T1
            2 * 2.0^2 - 1,            # T2
            4 * 2.0^3 - 3 * 2.0,         # T3
        ]
    end

end
