function obtain_esn_prediction(esn,
        prediction::Generative,
        x,
        output_layer,
        args...;
        initial_conditions = output_layer.last_value,
        save_states = false)
    out_size = output_layer.out_size
    training_method = output_layer.training_method
    prediction_len = prediction.prediction_len

    output = output_storing(training_method, out_size, prediction_len, typeof(esn.states))
    out = initial_conditions
    states = similar(esn.states, size(esn.states, 1), prediction_len)

    out_pad = allocate_outpad(esn, esn.states_type, out)
    tmp_array = allocate_tmp(esn.reservoir_driver, typeof(esn.states), esn.res_size)
    x_new = esn.states_type(esn.nla_type, x, out_pad)

    for i in 1:prediction_len
        x,
        x_new = next_state_prediction!(esn, x, x_new, out, out_pad, i, tmp_array,
            args...)
        out_tmp = get_prediction(output_layer.training_method, output_layer, x_new)
        out = store_results!(output_layer.training_method, out_tmp, output, i)
        states[:, i] = x
    end

    return save_states ? (output, states) : output
end

function obtain_esn_prediction(esn,
        prediction::Predictive,
        x,
        output_layer,
        args...;
        initial_conditions = output_layer.last_value,
        save_states = false)
    out_size = output_layer.out_size
    training_method = output_layer.training_method
    prediction_len = prediction.prediction_len

    output = output_storing(training_method, out_size, prediction_len, typeof(esn.states))
    out = initial_conditions
    states = similar(esn.states, size(esn.states, 1), prediction_len)

    out_pad = allocate_outpad(esn, esn.states_type, out)
    tmp_array = allocate_tmp(esn.reservoir_driver, typeof(esn.states), esn.res_size)
    x_new = esn.states_type(esn.nla_type, x, out_pad)

    for i in 1:prediction_len
        x,
        x_new = next_state_prediction!(esn, x, x_new, prediction.prediction_data[:, i],
            out_pad, i, tmp_array, args...)
        out_tmp = get_prediction(training_method, output_layer, x_new)
        out = store_results!(training_method, out_tmp, output, i)
        states[:, i] = x
    end

    return save_states ? (output, states) : output
end

#prediction dispatch on esn 
function next_state_prediction!(
        esn::AbstractEchoStateNetwork, x, x_new, out, out_pad, i, tmp_array, args...)
    out_pad = pad_state!(esn.states_type, out_pad, out)
    xv = @view x[1:(esn.res_size)]
    x = next_state!(x, esn.reservoir_driver, x, out_pad,
        esn.reservoir_matrix, esn.input_matrix, esn.bias_vector, tmp_array)
    x_new = esn.states_type(esn.nla_type, x, out_pad)
    return x, x_new
end

#TODO fixme @MatrinuzziFra
function next_state_prediction!(hesn::HybridESN,
        x,
        x_new,
        out,
        out_pad,
        i,
        tmp_array,
        model_prediction_data)
    out_tmp = vcat(out, model_prediction_data[:, i])
    out_pad = pad_state!(hesn.states_type, out_pad, out_tmp)
    x = next_state!(x, hesn.reservoir_driver, x[1:(hesn.res_size)], out_pad,
        hesn.reservoir_matrix, hesn.input_matrix, hesn.bias_vector, tmp_array)
    x_tmp = vcat(x, model_prediction_data[:, i])
    x_new = hesn.states_type(hesn.nla_type, x_tmp, out_pad)
    return x, x_new
end

function allocate_outpad(ens::AbstractEchoStateNetwork, states_type, out)
    return allocate_singlepadding(states_type, out)
end

function allocate_outpad(hesn::HybridESN, states_type, out)
    pad_length = length(out) + size(hesn.model.model_data[:, 1], 1)
    out_tmp = adapt(typeof(out), zeros(pad_length))
    return allocate_singlepadding(states_type, out_tmp)
end

function allocate_singlepadding(::AbstractPaddedStates, out)
    return adapt(typeof(out), zeros(size(out, 1) + 1))
end
function allocate_singlepadding(::StandardStates, out)
    return adapt(typeof(out), zeros(size(out, 1)))
end
function allocate_singlepadding(::ExtendedStates, out)
    return adapt(typeof(out), zeros(size(out, 1)))
end
