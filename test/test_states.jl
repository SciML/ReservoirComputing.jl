using ReservoirComputing

test_array = [1, 2, 3, 4, 5, 6, 7, 8, 9]
extension = [0, 0, 0]
padding = 10.0
test_types = [Float64, Float32, Float16]

nlas = [(NLADefault(), test_array),
    (NLAT1(), [1, 2, 9, 4, 25, 6, 49, 8, 81]),
    (NLAT2(), [1, 2, 2, 4, 12, 6, 30, 8, 9]),
    (NLAT3(), [1, 2, 8, 4, 24, 6, 48, 8, 9]),
    (PartialSquare(0.6), [1, 4, 9, 16, 25, 6, 7, 8, 9]),
    (ExtendedSquare(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 4, 9, 16, 25, 36, 49, 64, 81])]

pes = [(StandardStates(), test_array),
    (PaddedStates(; padding = padding),
        vcat(test_array, padding)),
    (PaddedExtendedStates(; padding = padding),
        vcat(test_array, padding, extension)),
    (ExtendedStates(), vcat(test_array, extension))]

@testset "States Testing" for T in test_types
    @testset "Nonlinear Algorithms Testing: $algo $T" for (algo, expected_output) in nlas
        nla_array = ReservoirComputing.nla(algo, T.(test_array))
        @test nla_array == expected_output
        @test eltype(nla_array) == T
    end
    @testset "States Testing: $state_type $T" for (state_type, expected_output) in pes
        states_output = state_type(NLADefault(), T.(test_array), T.(extension))
        @test states_output == expected_output
        @test eltype(states_output) == T
    end
end
