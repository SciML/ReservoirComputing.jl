using ReservoirComputing

test_array = [1, 2, 3, 4, 5, 6, 7, 8, 9]
extension = [0, 0, 0]
padding = 10.0

nlas = [(NLADefault(), test_array),
    (NLAT1(), [1, 2, 9, 4, 25, 6, 49, 8, 81]),
    (NLAT2(), [1, 2, 2, 4, 12, 6, 30, 8, 9]),
    (NLAT3(), [1, 2, 8, 4, 24, 6, 48, 8, 9])]

pes = [(StandardStates(), test_array),
    (PaddedStates(padding = padding),
        reshape(vcat(padding, test_array), length(test_array) + 1, 1)),
    (PaddedExtendedStates(padding = padding),
        reshape(vcat(padding, extension, test_array),
            length(test_array) + length(extension) + 1,
            1)),
    (ExtendedStates(), vcat(extension, test_array))]

function test_nla(algo, expected_output)
    nla_array = ReservoirComputing.nla(algo, test_array)
    @test nla_array == expected_output
end

function test_states_type(state_type, expected_output)
    states_output = state_type(NLADefault(), test_array, extension)
    @test states_output == expected_output
end

@testset "Nonlinear Algorithms Testing" for (algo, expected_output) in nlas
    test_nla(algo, expected_output)
end

@testset "States Testing" for (state_type, expected_output) in pes
    test_states_type(state_type, expected_output)
end
