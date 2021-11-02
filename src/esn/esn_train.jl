function train(esn::AbstractEchoStateNetwork, target_data, training_method=StandardRidge(0.0))

    esn.variation isa Hybrid ? states = vcat(esn.states, esn.variation.model_data[:, 2:end]) : states=esn.states
    esn.extended_states ? states=vcat(states, hcat(zeros(in_size), train_data[:,1:end])) : states = states
    states_new = nla(esn.nla_type, states)

    output_layer = _train(states_new, target_data, training_method)
    output_layer
end