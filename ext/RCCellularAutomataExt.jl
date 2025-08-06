module RCCellularAutomataExt
using ReservoirComputing: RECA, RandomMapping, RandomMaps
import ReservoirComputing: train, next_state_prediction!, AbstractOutputLayer, NLADefault,
                           StandardStates, obtain_prediction
using CellularAutomata
using Random: randperm

function RECA(train_data,
        automata;
        generations = 8,
        input_encoding = RandomMapping(),
        nla_type = NLADefault(),
        states_type = StandardStates())
    in_size = size(train_data, 1)
    #res_size = obtain_res_size(input_encoding, generations)
    state_encoding = create_encoding(input_encoding, train_data, generations)
    states = reca_create_states(state_encoding, automata, train_data)

    return RECA(train_data, automata, state_encoding, nla_type, states, states_type)
end

#training dispatch
function train(reca::RECA, target_data, training_method = StandardRidge; kwargs...)
    states_new = reca.states_type(reca.nla_type, reca.states, reca.train_data)
    return train(training_method, Float32.(states_new), Float32.(target_data); kwargs...)
end

#predict dispatch
function (reca::RECA)(prediction,
        output_layer::AbstractOutputLayer,
        initial_conditions = output_layer.last_value,
        last_state = zeros(reca.input_encoding.ca_size))
    return obtain_prediction(reca, prediction, last_state, output_layer;
        initial_conditions = initial_conditions)
end

function next_state_prediction!(reca::RECA, x, out, i, args...)
    rm = reca.input_encoding
    x = encoding(rm, out, x)
    ca = CellularAutomaton(reca.automata, x, rm.generations + 1)
    ca_states = ca.evolution[2:end, :]
    x_new = reshape(transpose(ca_states), rm.states_size)
    x = ca.evolution[end, :]
    return x, x_new
end

function RandomMapping(; permutations = 8, expansion_size = 40)
    RandomMapping(permutations, expansion_size)
end

function RandomMapping(permutations; expansion_size = 40)
    RandomMapping(permutations, expansion_size)
end

function create_encoding(rm::RandomMapping, input_data, generations)
    maps = init_maps(size(input_data, 1), rm.permutations, rm.expansion_size)
    states_size = generations * rm.expansion_size * rm.permutations
    ca_size = rm.expansion_size * rm.permutations
    return RandomMaps(rm.permutations, rm.expansion_size, generations, maps, states_size,
        ca_size)
end

function reca_create_states(rm::RandomMaps, automata, input_data)
    train_time = size(input_data, 2)
    states = zeros(rm.states_size, train_time)
    init_ca = zeros(rm.expansion_size * rm.permutations)

    for i in 1:train_time
        init_ca = encoding(rm, input_data[:, i], init_ca)
        ca = CellularAutomaton(automata, init_ca, rm.generations + 1)
        ca_states = ca.evolution[2:end, :]
        states[:, i] = reshape(transpose(ca_states), rm.states_size)
        init_ca = ca.evolution[end, :]
    end

    return states
end

function encoding(rm::RandomMaps, input_vector, tot_encoded_vector)
    input_size = size(input_vector, 1)
    #single_encoded_size = Int(size(tot_encoded_vector, 1)/permutations)
    new_tot_enc_vec = copy(tot_encoded_vector)

    for i in 1:(rm.permutations)
        new_tot_enc_vec[((i - 1) * rm.expansion_size + 1):(i * rm.expansion_size)] = single_encoding(
            input_vector,
            new_tot_enc_vec[((i - 1) * rm.expansion_size + 1):(i * rm.expansion_size)],
            rm.maps[i,
            :])
    end

    return new_tot_enc_vec
end

#function obtain_res_size(rm::RandomMapping, generations)
#    generations*rm.expansion_size*rm.permutations
#end

function single_encoding(input_vector, encoded_vector, map)
    new_enc_vec = copy(encoded_vector)

    for i in 1:size(input_vector, 1)
        new_enc_vec[map[i]] = input_vector[i]
    end

    return new_enc_vec
end

function init_maps(input_size, permutations, mapped_vector_size)
    maps = Array{Int}(undef, permutations, input_size)
    #tot_size = input_size*permutations

    for i in 1:permutations
        maps[i, :] = mapping(input_size, mapped_vector_size)
    end

    return maps
end

function mapping(input_size, mapped_vector_size)
    #sample(1:mapped_vector_size, input_size; replace=false)
    return randperm(mapped_vector_size)[1:input_size]
end

end #module
