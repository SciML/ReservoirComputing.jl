#dispatch on different training methods
#default
#linear models
function obtain_autonomous_prediction(esn::ESN, 
    output_layer, 
    prediction_len, 
    training_method::AbstractLinearModel,
    variation::Default)

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
    training_method::AbstractLinearModel,
    variation::Default)

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

#hybrid
function obtain_autonomous_prediction(esn::ESN, 
    output_layer, 
    prediction_len, 
    training_method::AbstractLinearModel,
    variation::Hybrid)

    output = zeros(output_layer.out_size, prediction_len)
    x = esn.states[:, end]
    predict_tsteps = [variation.tspan[2]+variation.dt]
    [append!(predict_tsteps, predict_tsteps[end]+variation.dt) for i in 1:prediction_len]
    tspan_new = (variation.tspan[2]+variation.dt, predict_tsteps[end])
    u0 = variation.model_data[:, end]
    model_prediction_data = variation.prior_model(u0, tspan_new, predict_tsteps)[:, 2:end]

    for i=1:prediction_len
        x_new = vcat(x, model_prediction_data[:,i])
        x_new = nla(esn.nla_type, x_new)
        out = (output_layer.output_matrix*x_new)
        output[:, i] = out
        out = vcat(out, model_prediction_data[:,i])
        esn.extended_states ? x = vcat(next_state(esn.reservoir_driver, x[1:esn.res_size], out, esn.reservoir_matrix, 
        esn.input_matrix), out) : x = next_state(esn.reservoir_driver, x, out, esn.reservoir_matrix, esn.input_matrix)
    end
    output
end

function obtain_direct_prediction(esn::ESN, 
    output_layer, 
    prediction_data, 
    training_method::AbstractLinearModel,
    variation::Hybrid)

    prediction_len = size(prediction_data, 2)
    output = zeros(output_layer.out_size, prediction_len)
    x = esn.states[:, end] #x = zeros(size(esn.states,2))?
    predict_tsteps = [variation.tspan[2]+variation.dt]
    [append!(predict_tsteps, predict_tsteps[end]+variation.dt) for i in 1:predict_len]
    tspan_new = (variation.tspan[2]+variation.dt, predict_tsteps[end])
    u0 = variation.model_data[:, end]
    model_prediction_data = variation.prior_model(u0, tspan_new, predict_tsteps)[:, 2:end]

    for i=1:prediction_len
        esn.extended_states ? x = vcat(next_state(esn.reservoir_driver, x[1:esn.res_size], direct.prediction_data[:,i], 
        esn.reservoir_matrix, esn.input_matrix), prediction_data[:,i]) : x = next_state(esn.reservoir_driver, x, 
        prediction_data[:,i], esn.reservoir_matrix, esn.input_matrix)
        x_new = vcat(nla(esn.nla_type, x), model_prediction_data[:,i])
        out = (output_layer.output_matrix*x_new)
        output[:, i] = out
        out = vcat(out, model_prediction_data[:,i]) 
    end
    output
end




#default
#gaussian processes
function obtain_autonomous_prediction(esn::ESN, 
    output_layer, 
    prediction_len, 
    training_method::AbstractGaussianProcess,
    variation::Default)
    
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
    training_method::AbstractGaussianProcess,
    variation::Default)
    
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
    training_method::LIBSVM.AbstractSVR,
    variation::Default)

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
    training_method::LIBSVM.AbstractSVR,
    variation::Default)

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