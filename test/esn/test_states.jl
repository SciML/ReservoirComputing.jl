using ReservoirComputing

test_types = [Float64, Float32, Float16]
states = [1, 2, 3, 4, 5, 6, 7, 8, 9]
in_data = fill(1, 3)

states_types = [StandardStates, ExtendedStates, PaddedStates, PaddedExtendedStates]

# testing extension and padding
for tt in test_types
    st_states = StandardStates()(NLADefault(), tt.(states), tt.(in_data))
    @test length(st_states) == length(states)
    @test typeof(st_states) == typeof(tt.(states))

    st_states = ExtendedStates()(NLADefault(), tt.(states), tt.(in_data))
    @test length(st_states) == length(states) + length(in_data)
    @test typeof(st_states) == typeof(tt.(states))

    st_states = PaddedStates()(NLADefault(), tt.(states), tt.(in_data))
    @test length(st_states) == length(states) + 1
    @test typeof(st_states[1]) == typeof(tt.(states)[1])

    st_states = PaddedExtendedStates()(NLADefault(), tt.(states), tt.(in_data))
    @test length(st_states) == length(states) + length(in_data) + 1
    @test typeof(st_states[1]) == typeof(tt.(states)[1])
end



## testing non linear algos
nla1_states = [1, 2, 9, 4, 25, 6, 49, 8, 81]
nla2_states = [1, 2, 2, 4, 12, 6, 30, 8, 9]
nla3_states = [1, 2, 8, 4, 24, 6, 48, 8, 9]



for tt in test_types
    # test default
    nla_states = ReservoirComputing.nla(NLADefault(), tt.(states))
    @test nla_states == tt.(states)
    # test NLAT1
    nla_states = ReservoirComputing.nla(NLAT1(), tt.(states))
    @test nla_states == tt.(nla1_states)
    # test nlat2
    nla_states = ReservoirComputing.nla(NLAT2(), tt.(states))
    @test nla_states == tt.(nla2_states)
    # test nlat3
    nla_states = ReservoirComputing.nla(NLAT3(), tt.(states))
    @test nla_states == tt.(nla3_states)
end
