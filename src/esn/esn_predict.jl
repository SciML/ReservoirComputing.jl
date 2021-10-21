#dispatch on different training methods
#linear
function obtain_autonomous_prediction(esn::ESN, 
                                      output_layer, 
                                      prediction_len, 
                                      training_method::AbstractLinearModel)

    output = zeros(output_layer.out_size, prediction_len)
    x = esn.states[:, end] 

    for i=1:prediction_len
        x_new = nla(esn.nla_type, x)
        out = (output_layer.output_matrix*x_new)
        output[:, i] = out
        esn.extended_states ? x = vcat(next_state(esn.reservoir_driver, x[1:esn.res_size], out, esn.reservoir_matrix, 
        esn.input_matrix), out) : x = next_state(esn.reservoir_driver, x, out, esn.reservoir_matrix, esn.input_matrix)
    end
    output
end

function obtain_direct_prediction(esn::ESN, 
                                  output_layer, 
                                  prediction_data, 
                                  training_method::AbstractLinearModel)

    prediction_len = size(prediction_data, 2)
    output = zeros(output_layer.out_size, prediction_len)
    x = esn.states[:, end] #x = zeros(size(esn.states,2))?

    for i=1:prediction_len
        esn.extended_states ? x = vcat(next_state(esn.reservoir_driver, x[1:esn.res_size], direct.prediction_data[:,i], 
        esn.reservoir_matrix, esn.input_matrix), prediction_data[:,i]) : x = next_state(esn.reservoir_driver, x, 
        prediction_data[:,i], esn.reservoir_matrix, esn.input_matrix)
        x_new = nla(esn.nla_type, x)
        out = (output_layer.output_matrix*x_new)
        output[:, i] = out    
    end
    output
end