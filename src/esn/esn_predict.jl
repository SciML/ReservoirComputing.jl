#dispatch on different training methods
#linear models
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

#gaussian processes
function obtain_autonomous_prediction(esn::ESN, 
    output_layer, 
    prediction_len, 
    training_method::AbstractGaussianProcess)
    
    output = zeros(output_layer.out_size, prediction_len)
    sigmas = zeros(output_layer.out_size, prediction_len)
    x = esn.states[:, end]

    for i=1:prediction_len
        x_new = hcat(nla(esn.nla_type, x)...)'
        out, sigma = [], []

        for i=1:size(output_layer.output_matrix, 1)
            append!(out, GaussianProcesses.predict_y(output_layer.output_matrix[i], x_new)[1][1])
            append!(sigma, GaussianProcesses.predict_y(output_layer.output_matrix[i], x_new)[2][1])
        end
            
        output[:, i] = out
        sigmas[:,i] = sigma
        esn.extended_states ? x = vcat(next_state(esn.reservoir_driver, x[1:esn.res_size], out, esn.reservoir_matrix, 
        esn.input_matrix), out) : x = next_state(esn.reservoir_driver, x, out, esn.reservoir_matrix, esn.input_matrix)
    end
    (output, sigmas)
end

function obtain_direct_prediction(esn::ESN, 
    output_layer, 
    prediction_data, 
    training_method::AbstractGaussianProcess)
    
    prediction_len = size(prediction_data, 2)
    output = zeros(output_layer.out_size, prediction_len)
    sigmas = zeros(output_layer.out_size, prediction_len)
    x = esn.states[:, end] #x = zeros(size(esn.states,2))?

    for i=1:prediction_len
        esn.extended_states ? x = vcat(next_state(esn.reservoir_driver, x[1:esn.res_size], direct.prediction_data[:,i], 
        esn.reservoir_matrix, esn.input_matrix), prediction_data[:,i]) : x = next_state(esn.reservoir_driver, x, 
        prediction_data[:,i], esn.reservoir_matrix, esn.input_matrix)
        x_new = hcat(nla(esn.nla_type, x)...)'
        out, sigma = [], []

        for i=1:size(output_layer.output_matrix, 1)
            append!(out, GaussianProcesses.predict_y(output_layer.output_matrix[i], x_new)[1][1])
            append!(sigma, GaussianProcesses.predict_y(output_layer.output_matrix[i], x_new)[2][1])
        end
            
        output[:, i] = out
        sigmas[:,i] = sigma
        
    end
    (output, sigmas)
end

#support vector regression
function obtain_autonomous_prediction(esn::ESN, 
    output_layer, 
    prediction_len, 
    training_method::LIBSVM.AbstractSVR)

    output = zeros(output_layer.out_size, prediction_len)
    x = esn.states[:, end]

    for i=1:prediction_len
        x_new = hcat(nla(esn.nla_type, x)...)
        out = []
        for i=1:size(output_layer.output_matrix, 1)
            push!(out, LIBSVM.predict(output_layer.output_matrix[i], x_new)[1])
        end
        output[:, i] = out
        esn.extended_states ? x = vcat(next_state(esn.reservoir_driver, x[1:esn.res_size], out, esn.reservoir_matrix, 
        esn.input_matrix), out) : x = next_state(esn.reservoir_driver, x, out, esn.reservoir_matrix, esn.input_matrix)
    end

    output
end

function obtain_direct_prediction(esn::ESN, 
    output_layer, 
    prediction_data, 
    training_method::LIBSVM.AbstractSVR)

    prediction_len = size(prediction_data, 2)
    output = zeros(output_layer.out_size, prediction_len)
    x = esn.states[:, end]

    for i=1:prediction_len
        esn.extended_states ? x = vcat(next_state(esn.reservoir_driver, x[1:esn.res_size], direct.prediction_data[:,i], 
        esn.reservoir_matrix, esn.input_matrix), prediction_data[:,i]) : x = next_state(esn.reservoir_driver, x, 
        prediction_data[:,i], esn.reservoir_matrix, esn.input_matrix)
        x_new = hcat(nla(esn.nla_type, x)...)
        out = []
        for i=1:size(output_layer.output_matrix, 1)
            push!(out, LIBSVM.predict(output_layer.output_matrix[i], x_new)[1])
        end
        output[:, i] = out
    end

    output
end