function obtain_prediction(
    rc::AbstractReservoirComputer,
    prediction::Generative,
    x,
    output_layer,
    args...;
    initial_conditions=output_layer.last_value
)
    #x = last_state
    prediction_len = prediction.prediction_len
    train_method = output_layer.training_method
    out_size = output_layer.out_size
    output = output_storing(train_method, out_size, prediction_len, typeof(rc.states))
    out = initial_conditions

    for i=1:prediction_len
        x, x_new = next_state_prediction!(rc, x, out, i, args...)
        out_tmp = get_prediction(train_method, output_layer, x_new)
        out = store_results!(train_method, out_tmp, output, i)
    end
    
    return output
end

function obtain_prediction(
    rc::AbstractReservoirComputer,
    prediction::Predictive,
    x,
    output_layer,
    args...;
    kwargs...
)
    
    prediction_len = prediction.prediction_len
    train_method = output_layer.training_method
    out_size = output_layer.out_size
    output = output_storing(train_method, out_size, prediction_len, typeof(rc.states))

    for i in 1:prediction_len
        y = @view prediction.prediction_data[:,i]
        x, x_new = next_state_prediction!(rc, x, y, i, args...)
        out_tmp = get_prediction(train_method, output_layer, x_new)
        out = store_results!(output_layer.training_method, out_tmp, output, i)
    end

    return output
end

#linear models
function get_prediction(training_method::AbstractLinearModel, output_layer, x)
    return output_layer.output_matrix*x
end

#gaussian regression
function get_prediction(training_method::AbstractGaussianProcess, output_layer, x)
    out, sigma = zeros(output_layer.out_size), zeros(output_layer.out_size)

    for j in 1:output_layer.out_size
        x_new = reshape(x, length(x), 1)
        gr = GaussianProcesses.predict_y(output_layer.output_matrix[j], x_new)
        out[j] = gr[1][1]
        sigma[j] = gr[2][1]
    end

    return (out, sigma)
end

#support vector regression
function get_prediction(training_method::LIBSVM.AbstractSVR, output_layer, x)
    out = zeros(output_layer.out_size)

    for i in 1:output_layer.out_size
        x_new = reshape(x, 1, length(x))
        out[i] = LIBSVM.predict(output_layer.output_matrix[i], x_new)[1]
    end

    return out
end

#creation of matrices for storing gaussian results (outs and sigmas)
function output_storing(
    training_method::AbstractGaussianProcess,
    out_size,
    prediction_len,
    storing_type)
    out = Adapt.adapt(storing_type, zeros(out_size, prediction_len))
    sigma = Adapt.adapt(storing_type, zeros(out_size, prediction_len))
    return (out, sigma)
end

#single matrix for other training methods
function output_storing(training_method, out_size, prediction_len, storing_type)
    return Adapt.adapt(storing_type, zeros(out_size, prediction_len))
end

#storing results for gaussian training, getting also the sigmas
function store_results!(training_method::AbstractGaussianProcess, out, output, i)
    output[1][:, i] = out[1]
    output[2][:, i] = out[2]
    return out[1]
end

#general storing -> single matrix
function store_results!(training_method, out, output, i)
    output[:, i] = out
    return out
end
