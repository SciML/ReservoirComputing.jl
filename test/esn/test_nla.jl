using ReservoirComputing

states = [1, 2, 3, 4, 5, 6, 7, 8, 9]
nla1_states = [1, 2, 9, 4, 25, 6, 49, 8, 81]
nla2_states = [1, 2, 2, 4, 12, 6, 30, 8, 9]
nla3_states = [1, 2, 8, 4, 24, 6, 48, 8, 9]


test_types = [Float64, Float32, Float16]

for tt in test_types
    # test default
    nla_states = ReservoirComputing.nla(NLADefault(), tt.(states))
    @test nla_states == tt.(states)
    # test NLAT1
    nla_states = ReservoirComputing.nla(NLAT1(), tt.(states))
    @test nla_states = tt.(nla1_states)
    # test nlat2
    nla_states = ReservoirComputing.nla(NLAT2(), tt.(states))
    @test nla_states = tt.(nla2_states)
    # test nlat3
    nla_states = ReservoirComputing.nla(NLAT3(), tt.(states))
    @test nla_states = tt.(nla3_states)
end
