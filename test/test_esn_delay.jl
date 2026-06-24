using Test
using Random
using ReservoirComputing
using Static
using LinearAlgebra

@testset "InputDelayESN" begin
    rng = MersenneTwister(123)

    @testset "constructor wiring & dimensions" begin
        in_dims = 3
        res_dims = 5
        out_dims = 2
        num_delays = 2

        idesn = InputDelayESN(
            in_dims, res_dims, out_dims;
            num_delays = num_delays, stride = 1
        )

        @test idesn isa InputDelayESN
        @test idesn.input_delay isa DelayLayer
        @test Int(idesn.input_delay.in_dims) == in_dims
        @test idesn.reservoir isa StatefulLayer
        @test idesn.reservoir.cell isa ESNCell
        @test Int(idesn.reservoir.cell.in_dims) == in_dims * (num_delays + 1)
        @test Int(idesn.reservoir.cell.out_dims) == res_dims
        @test idesn.states_modifiers isa Tuple
        @test isempty(idesn.states_modifiers)
        @test idesn.readout isa LinearReadout
        @test Int(idesn.readout.in_dims) == res_dims
    end

    @testset "with state modifiers" begin
        in_dims = 3
        res_dims = 5
        out_dims = 2

        idesn = InputDelayESN(
            in_dims, res_dims, out_dims;
            states_modifiers = (NLAT1,)
        )

        @test !isempty(idesn.states_modifiers)
        @test idesn.states_modifiers[1].func === NLAT1
    end

    @testset "num_delays changes reservoir input dim" begin
        in_dims = 2
        res_dims = 6
        out_dims = 1

        idesn1 = InputDelayESN(in_dims, res_dims, out_dims; num_delays = 0)
        idesn2 = InputDelayESN(in_dims, res_dims, out_dims; num_delays = 3)

        @test Int(idesn1.reservoir.cell.in_dims) == in_dims * (0 + 1)
        @test Int(idesn2.reservoir.cell.in_dims) == in_dims * (3 + 1)
    end
end

@testset "StateDelayESN" begin
    rng = MersenneTwister(123)

    @testset "constructor wiring & dimensions" begin
        in_dims = 3
        res_dims = 5
        out_dims = 2
        num_delays = 2

        desn = StateDelayESN(
            in_dims, res_dims, out_dims;
            num_delays = num_delays, stride = 1
        )

        @test desn isa StateDelayESN

        reservoir = desn.reservoir
        @test reservoir isa StatefulLayer

        mods = desn.states_modifiers
        @test mods isa Tuple
        @test !isempty(mods)
        @test first(mods) isa DelayLayer

        dl = first(mods)
        @test Int(dl.in_dims) == res_dims
        @test Int(dl.num_delays) == num_delays
        @test Int(dl.stride) == 1

        ro = desn.readout
        @test ro isa LinearReadout
        @test Int(ro.in_dims) == res_dims * (num_delays + 1)
        @test Int(ro.out_dims) == out_dims
    end

    @testset "num_delays changes readout input dim" begin
        in_dims = 2
        res_dims = 6
        out_dims = 1

        desn1 = StateDelayESN(in_dims, res_dims, out_dims; num_delays = 0)
        desn2 = StateDelayESN(in_dims, res_dims, out_dims; num_delays = 3)

        ro1 = desn1.readout
        ro2 = desn2.readout

        @test Int(ro1.in_dims) == res_dims * (0 + 1)
        @test Int(ro2.in_dims) == res_dims * (3 + 1)
    end
end

@testset "DelayESN" begin
    rng = MersenneTwister(123)

    @testset "constructor wiring & dimensions" begin
        in_dims = 3
        res_dims = 5
        out_dims = 2
        num_input_delays = 2
        num_state_delays = 3

        fesn = DelayESN(
            in_dims, res_dims, out_dims;
            num_input_delays = num_input_delays,
            num_state_delays = num_state_delays,
            input_stride = 1,
            state_stride = 1
        )

        @test fesn isa DelayESN

        @test fesn.input_delay isa DelayLayer
        @test Int(fesn.input_delay.in_dims) == in_dims
        @test Int(fesn.input_delay.num_delays) == num_input_delays

        @test fesn.reservoir isa StatefulLayer
        @test fesn.reservoir.cell isa ESNCell
        @test Int(fesn.reservoir.cell.in_dims) == in_dims * (num_input_delays + 1)
        @test Int(fesn.reservoir.cell.out_dims) == res_dims

        @test fesn.states_modifiers isa Tuple
        @test !isempty(fesn.states_modifiers)

        state_delay_layer = first(fesn.states_modifiers)
        @test state_delay_layer isa DelayLayer
        @test Int(state_delay_layer.in_dims) == res_dims
        @test Int(state_delay_layer.num_delays) == num_state_delays

        @test fesn.readout isa LinearReadout
        @test Int(fesn.readout.in_dims) == res_dims * (num_state_delays + 1)
        @test Int(fesn.readout.out_dims) == out_dims
    end

    @testset "with state modifiers" begin
        in_dims = 3
        res_dims = 5
        out_dims = 2

        fesn = DelayESN(
            in_dims, res_dims, out_dims;
            states_modifiers = (NLAT1,)
        )

        @test length(fesn.states_modifiers) == 2
        @test fesn.states_modifiers[1] isa DelayLayer
        @test fesn.states_modifiers[2].func === NLAT1
    end

    @testset "delays change dimensions independently" begin
        in_dims = 2
        res_dims = 6
        out_dims = 1

        fesn1 = DelayESN(
            in_dims, res_dims, out_dims;
            num_input_delays = 0, num_state_delays = 0
        )

        fesn2 = DelayESN(
            in_dims, res_dims, out_dims;
            num_input_delays = 3, num_state_delays = 2
        )

        @test Int(fesn1.reservoir.cell.in_dims) == in_dims * (0 + 1)
        @test Int(fesn2.reservoir.cell.in_dims) == in_dims * (3 + 1)

        @test Int(fesn1.readout.in_dims) == res_dims * (0 + 1)
        @test Int(fesn2.readout.in_dims) == res_dims * (2 + 1)
    end
end

@testset "InputDelayESN forward pass & training" begin
    rng = MersenneTwister(123)
    in_dims, res_dims, out_dims, num_delays = 3, 40, 3, 2
    n_steps = 60
    data = rand(rng, Float32, in_dims, n_steps)
    target = rand(rng, Float32, out_dims, n_steps)

    idesn = InputDelayESN(in_dims, res_dims, out_dims; num_delays = num_delays)
    ps, st = setup(rng, idesn)

    @testset "params/state expose input_delay" begin
        @test haskey(ps, :input_delay)
        @test haskey(st, :input_delay)
        @test size(ps.reservoir.input_matrix, 2) == in_dims * (num_delays + 1)
    end

    @testset "collectstates" begin
        states, newst = collectstates(idesn, data, ps, st)
        @test size(states) == (res_dims, n_steps)
        @test all(isfinite, states)
        @test newst.input_delay.clock == n_steps
    end

    @testset "train! + predict" begin
        (ps_t, st_t), states = train!(
            idesn, data, target, ps, st, StandardRidge(1.0e-6); return_states = true
        )
        @test size(states) == (res_dims, n_steps)

        out_tf, _ = predict(idesn, data, ps_t, st_t)
        @test size(out_tf) == (out_dims, n_steps)
        @test all(isfinite, out_tf)

        steps = 8
        out_ar, _ = predict(idesn, steps, ps_t, st_t; initialdata = data[:, end])
        @test size(out_ar) == (out_dims, steps)
        @test all(isfinite, out_ar)
    end

    @testset "deterministic for a fixed seed" begin
        ps1, st1 = setup(MersenneTwister(99), idesn)
        ps2, st2 = setup(MersenneTwister(99), idesn)
        s1, _ = collectstates(idesn, data, ps1, st1)
        s2, _ = collectstates(idesn, data, ps2, st2)
        @test s1 == s2
    end

    @testset "stride > 1 smoke" begin
        strided = InputDelayESN(
            in_dims, res_dims, out_dims; num_delays = num_delays, stride = 2
        )
        ps_s, st_s = setup(MersenneTwister(321), strided)

        states_s, st_states = collectstates(strided, data, ps_s, st_s)
        @test size(states_s) == (res_dims, n_steps)
        @test all(isfinite, states_s)
        @test st_states.input_delay.clock == n_steps

        ps_t, st_t = train!(strided, data, target, ps_s, st_s, StandardRidge(1.0e-6))
        out_tf, st_pred = predict(strided, data, ps_t, st_t)
        @test size(out_tf) == (out_dims, n_steps)
        @test all(isfinite, out_tf)
        @test st_pred.input_delay.clock == st_t.input_delay.clock + n_steps
    end
end

@testset "DelayESN forward pass & training" begin
    rng = MersenneTwister(123)
    in_dims, res_dims, out_dims = 3, 40, 2
    num_input_delays, num_state_delays = 2, 1
    n_steps = 60
    data = rand(rng, Float32, in_dims, n_steps)
    target = rand(rng, Float32, out_dims, n_steps)

    fesn = DelayESN(
        in_dims, res_dims, out_dims;
        num_input_delays = num_input_delays, num_state_delays = num_state_delays
    )
    ps, st = setup(rng, fesn)

    @testset "params/state expose input_delay" begin
        @test haskey(ps, :input_delay)
        @test haskey(st, :input_delay)
        @test size(ps.reservoir.input_matrix, 2) == in_dims * (num_input_delays + 1)
    end

    @testset "collectstates" begin
        states, newst = collectstates(fesn, data, ps, st)
        @test size(states) == (res_dims * (num_state_delays + 1), n_steps)
        @test all(isfinite, states)
        @test newst.input_delay.clock == n_steps
    end

    @testset "train! + predict" begin
        ps_t, st_t = train!(fesn, data, target, ps, st, StandardRidge(1.0e-6))

        out_tf, _ = predict(fesn, data, ps_t, st_t)
        @test size(out_tf) == (out_dims, n_steps)
        @test all(isfinite, out_tf)
    end

    @testset "stride > 1 smoke" begin
        strided = DelayESN(
            in_dims, res_dims, out_dims;
            num_input_delays = num_input_delays,
            input_stride = 2,
            num_state_delays = num_state_delays,
            state_stride = 2
        )
        ps_s, st_s = setup(MersenneTwister(321), strided)

        states_s, st_states = collectstates(strided, data, ps_s, st_s)
        @test size(states_s) == (res_dims * (num_state_delays + 1), n_steps)
        @test all(isfinite, states_s)
        @test st_states.input_delay.clock == n_steps
        @test st_states.states_modifiers[1].clock == n_steps

        ps_t, st_t = train!(strided, data, target, ps_s, st_s, StandardRidge(1.0e-6))
        out_tf, st_pred = predict(strided, data, ps_t, st_t)
        @test size(out_tf) == (out_dims, n_steps)
        @test all(isfinite, out_tf)
        @test st_pred.input_delay.clock == st_t.input_delay.clock + n_steps
        @test st_pred.states_modifiers[1].clock ==
            st_t.states_modifiers[1].clock + n_steps
    end
end

@testset "StateDelayESN forward pass & training" begin
    rng = MersenneTwister(123)
    in_dims, res_dims, out_dims, num_delays = 3, 40, 2, 2
    n_steps = 60
    data = rand(rng, Float32, in_dims, n_steps)
    target = rand(rng, Float32, out_dims, n_steps)

    sdesn = StateDelayESN(in_dims, res_dims, out_dims; num_delays = num_delays)
    ps, st = setup(rng, sdesn)

    states, _ = collectstates(sdesn, data, ps, st)
    @test size(states) == (res_dims * (num_delays + 1), n_steps)

    ps_t, st_t = train!(sdesn, data, target, ps, st, StandardRidge(1.0e-6))
    out_tf, _ = predict(sdesn, data, ps_t, st_t)
    @test size(out_tf) == (out_dims, n_steps)
    @test all(isfinite, out_tf)
end
