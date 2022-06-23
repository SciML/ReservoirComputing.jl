using ReservoirComputing

#padding
test_array = [1, 2, 3, 4, 5, 6, 7, 8, 9]
standard_array = zeros(length(test_array), 1)
extension = [0, 0, 0]
padded_array = zeros(length(test_array) + 1, 1)
extended_array = zeros(length(test_array) + length(extension), 1)
padded_extended_array = zeros(length(test_array) + length(extension) + 1, 1)
padding = 10.0

#testing non linear algos
nla_array = ReservoirComputing.nla(NLADefault(), test_array)
@test nla_array == test_array

nla_array = ReservoirComputing.nla(NLAT1(), test_array)
@test nla_array == [1, 2, 9, 4, 25, 6, 49, 8, 81]

nla_array = ReservoirComputing.nla(NLAT2(), test_array)
@test nla_array == [1, 2, 2, 4, 12, 6, 30, 8, 9]

nla_array = ReservoirComputing.nla(NLAT3(), test_array)
@test nla_array == [1, 2, 8, 4, 24, 6, 48, 8, 9]

#testing padding and extension
states_type = StandardStates()
standard_array = states_type(NLADefault(), test_array, extension)
@test standard_array == test_array

states_type = PaddedStates(padding = padding)
padded_array = states_type(NLADefault(), test_array, extension)
@test padded_array == reshape(vcat(padding, test_array), length(test_array) + 1, 1)

states_type = PaddedStates(padding)
padded_array = states_type(NLADefault(), test_array, extension)
@test padded_array == reshape(vcat(padding, test_array), length(test_array) + 1, 1)

states_type = PaddedExtendedStates(padding = padding)
padded_extended_array = states_type(NLADefault(), test_array, extension)
@test padded_extended_array == reshape(vcat(padding, extension, test_array),
              length(test_array) + length(extension) + 1, 1)

states_type = PaddedExtendedStates(padding)
padded_extended_array = states_type(NLADefault(), test_array, extension)
@test padded_extended_array == reshape(vcat(padding, extension, test_array),
              length(test_array) + length(extension) + 1, 1)

states_type = ExtendedStates()
extended_array = states_type(NLADefault(), test_array, extension)
@test extended_array == vcat(extension, test_array)
#reshape(vcat(extension, test_array), length(test_array)+length(extension), 1)
