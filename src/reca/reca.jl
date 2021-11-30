abstract type AbstractReca <: AbstractReservoirComputer end

struct RECA{S,R,E,T,Q} <: AbstractReca
    #res_size::I
    train_data::S
    automata::R
    input_encoding::E
    nla_type::ReservoirComputing.NonLinearAlgorithm
    states::T
    states_type::Q
end

"""
    RECA()

[1] Yilmaz, Ozgur. “Reservoir computing using cellular automata.” arXiv preprint arXiv:1410.0162 (2014).
[2] Nichele, Stefano, and Andreas Molund. “Deep reservoir computing using cellular automata.” arXiv preprint arXiv:1703.02806 (2017).
"""
function RECA(train_data,
    automata;
    generations = 8,
    input_encoding=RandomMapping(),
    nla_type = NLADefault(),
    states_type = StandardStates())
    
    in_size = size(train_data, 1)
    #res_size = obtain_res_size(input_encoding, generations)
    encoding = create_encoding(input_encoding, train_data, generations)
    states = reca_create_states(encoding, automata, train_data)
    
    RECA(train_data, automata, encoding, nla_type, states, states_type)
end

#training dispatch
function train(reca::AbstractReca, target_data, training_method=StandardRidge(0.0))

    states_new = reca.states_type(reca.nla_type, reca.states, reca.train_data)

    _train(states_new, target_data, training_method)
end

#predict dispatch
function (reca::RECA)(prediction,
    output_layer::AbstractOutputLayer,
    initial_conditions=output_layer.last_value,
    last_state=reca.states[:, end])

    obtain_prediction(reca, prediction, last_state, output_layer; initial_conditions=initial_conditions)
end

function next_state_prediction!(reca::RECA, x, out, i, args...)

    rm = reca.input_encoding

    x = encoding(rm, out, x)
    ca = CellularAutomaton(reca.automata, x, rm.generations+1)
    ca_states = ca.evolution[2:end ,:]
    x_new = reshape(transpose(ca_states), rm.generations*rm.expansion_size*rm.permutations)
    x = ca.evolution[end, :]
    x, x_new
end

