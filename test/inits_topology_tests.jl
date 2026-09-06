using Test
using LinearAlgebra
using Random
using ReservoirComputing

@topology topology_selfloop_cycle begin
    self_loop!
    simple_cycle!
end
@topology topology_forward begin
    forward = delay_line!(shift = 2)
end
@topology topology_delayline_backward begin
    delay_line!
    backward_connection!
end
@topology topology_perm begin
    weight = self_loop!
    permute_matrix!
end
@topology topology_cycle_jumps begin
    simple_cycle!
    add_jumps!
end
@topology topology_true_doublecycle begin
    simple_cycle!
    reverse_simple_cycle!
end
@topology topology_selfloop_forward begin
    self_loop!
    forward = delay_line!(shift = 2)
end
@topology topology_heavy_cycle begin
    simple_cycle!(weight = 0.5)
end
@topology topology_delay begin
    delay_line!
end
@topology topology_cycle begin
    simple_cycle!
end
@topology topology_selfloop_delayline_backward begin
    self_loop!
    delay_line!
    backward_connection!(shift = 2)
end

@testset "@topology" begin
    @test size(topology_selfloop_cycle(16, 16)) == (16, 16)
    @test eltype(topology_selfloop_cycle(Float64, 8, 8)) == Float64
    @test eltype(topology_selfloop_cycle(Xoshiro(1))(Float32, 8, 8)) == Float32

    @test topology_selfloop_cycle(5, 5) == selfloop_cycle(5, 5)
    @test topology_selfloop_cycle(
        5, 5; cycle_weight = -0.2, selfloop_weight = 0.5
    ) == selfloop_cycle(5, 5; cycle_weight = -0.2, selfloop_weight = 0.5)
    @test topology_selfloop_cycle(
        MersenneTwister(123), 5, 5; cycle_kwargs = (; signs = RandomSigns())
    ) == selfloop_cycle(
        MersenneTwister(123), 5, 5; cycle_kwargs = (; signs = RandomSigns())
    )
    @test topology_forward(5, 5) == forward_connection(5, 5)
    @test topology_delayline_backward(5, 5; delay_shift = 3, fb_shift = 2) ==
        delayline_backward(5, 5; delay_shift = 3, fb_shift = 2)
    @test topology_heavy_cycle(4, 4) == simple_cycle(4, 4; cycle_weight = 0.5)
    @test topology_cycle_jumps(5, 5) == cycle_jumps(5, 5)
    @test topology_cycle_jumps(5, 5; jump_size = 2) ==
        cycle_jumps(5, 5; jump_size = 2)
    @test topology_true_doublecycle(5, 5) == true_doublecycle(5, 5)
    @test topology_selfloop_forward(5, 5) == selfloop_forwardconnection(5, 5)
    @test topology_selfloop_forward(
        5, 5; forward_weight = 0.5f0, selfloop_weight = 0.99f0
    ) == selfloop_forwardconnection(
        5, 5; forward_weight = 0.5f0, selfloop_weight = 0.99f0
    )
    @test topology_selfloop_forward(
        5, 5; delay_kwargs = (; signs = IrrationalDigitSigns())
    ) == selfloop_forwardconnection(
        5, 5; delay_kwargs = (; signs = IrrationalDigitSigns())
    )
    @test topology_perm(MersenneTwister(123), 5, 5) ==
        permutation_init(MersenneTwister(123), 5, 5)
    @test topology_perm(MersenneTwister(123), 5, 5; weight = 0.99f0) ==
        permutation_init(MersenneTwister(123), 5, 5; weight = 0.99f0)
    @test topology_delay(5, 5) == delay_line(5, 5)
    @test topology_delay(5, 5; delay_shift = 3, signs = IrrationalDigitSigns()) ==
        delay_line(5, 5; delay_shift = 3, signs = IrrationalDigitSigns())
    @test topology_cycle(5, 5; cycle_weight = 0.99) ==
        simple_cycle(5, 5; cycle_weight = 0.99)
    @test topology_selfloop_delayline_backward(5, 5) ==
        selfloop_delayline_backward(5, 5)
    cw = Float32[0.2, 0.4, 0.6, 0.8, 1.0]
    sw = -Float32[0.1, 0.3, 0.5, 0.7, 0.9]
    @test topology_selfloop_cycle(5, 5; cycle_weight = cw, selfloop_weight = sw) ==
        selfloop_cycle(5, 5; cycle_weight = cw, selfloop_weight = sw)
    @test isapprox(
        maximum(abs.(eigvals(topology_selfloop_cycle(8, 8; radius = 1.0)))),
        1.0; atol = 1.0e-5
    )
    @test_throws DimensionMismatch topology_selfloop_cycle(3, 4)

    ps, = setup(Xoshiro(0), ESNCell(2 => 8; init_reservoir = topology_selfloop_cycle))
    @test ps.reservoir_matrix == topology_selfloop_cycle(8, 8)
    ps_esn, = setup(
        Xoshiro(0), ESN(2, 8, 1; init_reservoir = topology_selfloop_cycle)
    )
    @test ps_esn.reservoir.reservoir_matrix == topology_selfloop_cycle(8, 8)

    @test_throws ArgumentError @macroexpand @topology empty_topology begin
    end
    @test_throws ArgumentError @macroexpand @topology (10, 10) begin
        self_loop!
    end
    @test_throws ArgumentError @macroexpand @topology bad_block begin
        not_a_block!
    end
    @test_throws ArgumentError @macroexpand @topology duplicate_prefix begin
        delay_line!
        delay_line!
    end
end
