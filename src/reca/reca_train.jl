function train(reca::AbstractReca, target_data, training_method=StandardRidge(0.0))

    states_new = nla(reca.nla_type, reca.states)

    output_layer = _train(states_new, target_data, training_method)
    output_layer
end