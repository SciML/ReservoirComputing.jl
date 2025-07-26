abstract type AbstractReca <: AbstractReservoirComputer end

struct RECA{S, R, E, T, Q} <: AbstractReca
    #res_size::I
    train_data::S
    automata::R
    input_encoding::E
    nla_type::ReservoirComputing.NonLinearAlgorithm
    states::T
    states_type::Q
end

"""
    RECA(train_data,
        automata;
        generations = 8,
        input_encoding=RandomMapping(),
        nla_type = NLADefault(),
        states_type = StandardStates())

Builds a Resercoir Computing model with cellular automata [Yilmaz2014](@cite)
[Nichele2017](@cite).
"""
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
function train(reca::AbstractReca, target_data, training_method = StandardRidge; kwargs...)
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
