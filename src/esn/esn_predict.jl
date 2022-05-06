function obtain_esn_prediction(esn, prediction::Generative,
    x,
    output_layer,
    args...;
    initial_conditions=output_layer.last_value)

    prediction_len = prediction.prediction_len
    output = output_storing(output_layer.training_method, output_layer.out_size, prediction_len, typeof(esn.states))
    out = initial_conditions

    out_pad = allocate_outpad(esn.variation, esn.states_type, out)
    tmp_array = allocate_tmp(esn.reservoir_driver, typeof(esn.states), esn.res_size)
    x_new = esn.states_type(esn.nla_type, x, out_pad)

    for i=1:prediction_len
        x, x_new = next_state_prediction!(esn, x, x_new, out, out_pad, i, tmp_array, args...)
        out_tmp = get_prediction(output_layer.training_method, output_layer, x_new)
        out = store_results!(output_layer.training_method, out_tmp, output, i)
    end
    output
end

function obtain_esn_prediction(esn, prediction::Predictive,
    x,
    output_layer,
    args...;
    initial_conditions=output_layer.last_value)

    prediction_len = prediction.prediction_len
    output = output_storing(output_layer.training_method, output_layer.out_size, prediction_len, typeof(esn.states))
    out = initial_conditions

    out_pad = allocate_outpad(esn.variation, esn.states_type, out)
    tmp_array = allocate_tmp(esn.reservoir_driver, typeof(esn.states), esn.res_size)
    x_new = esn.states_type(esn.nla_type, x, out_pad)

    for i=1:prediction_len
        x, x_new = next_state_prediction!(esn, x, x_new, prediction.prediction_data[:,i], out_pad, i, tmp_array, args...)
        out_tmp = get_prediction(output_layer.training_method, output_layer, x_new)
        out = store_results!(output_layer.training_method, out_tmp, output, i)
    end
    output
end

#prediction dispatch on esn 
function next_state_prediction!(esn::ESN, x, x_new, out, out_pad, i, tmp_array, args...)
    _variation_prediction!(esn.variation, esn, x, x_new, out, out_pad, i, tmp_array, args...)
end

#dispatch the prediction on the esn variation
function _variation_prediction!(variation, esn, x, x_new, out, out_pad, i, tmp_array, args...)
    out_pad = pad_state!(esn.states_type, out_pad, out)
    xv = @view x[1:esn.res_size]
    x = next_state!(x, esn.reservoir_driver, xv, out_pad,
        esn.reservoir_matrix, esn.input_matrix, esn.bias_vector, tmp_array)
    x_new = esn.states_type(esn.nla_type, x, out_pad)
    x, x_new
end

function _variation_prediction!(variation::Hybrid, esn, x, x_new, out, out_pad, i, tmp_array, model_prediction_data)
    out_tmp = vcat(out, model_prediction_data[:,i])
    out_pad = pad_state!(esn.states_type, out_pad, out_tmp)
    x = next_state!(x, esn.reservoir_driver, x[1:esn.res_size], out_pad,
        esn.reservoir_matrix, esn.input_matrix, esn.bias_vector, tmp_array)
    x_tmp = vcat(x, model_prediction_data[:,i])
    x_new = esn.states_type(esn.nla_type, x_tmp, out_pad)
    x, x_new
end

function allocate_outpad(variation, states_type, out)
    out_pad = allocate_singlepadding(states_type, out)
end

function allocate_outpad(variation::Hybrid, states_type, out)
    pad_length = length(out) + size(variation.model_data[:,1], 1)
    out_tmp = Adapt.adapt(typeof(out), zeros(pad_length))
    out_pad = allocate_singlepadding(states_type, out_tmp)
end
    
allocate_singlepadding(::AbstractPaddedStates, out) = Adapt.adapt(typeof(out), zeros(size(out, 1)+1))
allocate_singlepadding(::StandardStates, out) = Adapt.adapt(typeof(out), zeros(size(out, 1)))
allocate_singlepadding(::ExtendedStates, out) = Adapt.adapt(typeof(out), zeros(size(out, 1)))


