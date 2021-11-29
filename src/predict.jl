function obtain_prediction(rc::AbstractReservoirComputer,
    prediction::Generative,
    last_state,
    output_layer,
    args...;
    initial_conditions=output_layer.last_value)

    x = last_state
    prediction_len = prediction.prediction_len
    output = output_storing(output_layer.training_method, output_layer.out_size, prediction_len)
    out = initial_conditions

    for i=1:prediction_len
        x, x_new = next_state_prediction!(rc, x, out, i, args...)
        out = get_prediction(output_layer.training_method, output_layer, x_new)
        store_results!(output_layer.training_method, out, output, i)
    end
    output
end

function obtain_prediction(rc::AbstractReservoirComputer,
    prediction::Predictive,
    last_state,
    output_layer,
    args...;
    kwargs...)

    x = last_state
    prediction_len = prediction.prediction_len
    output = output_storing(output_layer.training_method, output_layer.out_size, prediction_len)

    for i=1:prediction_len
        x, x_new = next_state_prediction!(rc, x, prediction.prediction_data[:,i], i, args...)
        out = get_prediction(output_layer.training_method, output_layer, x_new)
        store_results!(output_layer.training_method, out, output, i)
    end
    output
end

#linear models
function get_prediction(training_method::AbstractLinearModel, output_layer, x)
    output_layer.output_matrix*x
end

#gaussian regression
function get_prediction(training_method::AbstractGaussianProcess, output_layer, x)
    out, sigma = zeros(output_layer.out_size), zeros(output_layer.out_size)

    for j=1:output_layer.out_size
        gr = GaussianProcesses.predict_y(output_layer.output_matrix[j], x_new)
        out[j] = gr[1][1]
        sigma[j] = gr[2][1]
    end
    out, sigma
end

#support vector regression
function get_prediction(training_method::LIBSVM.AbstractSVR, output_layer, x)
    out = zeros(output_layer.out_size)

    for i=1:output_layer.out_size
        out[i] = LIBSVM.predict(output_layer.output_matrix[i], x_new)[1]
    end
    out
end

#creation of matrices for storing gaussian results (outs and sigmas)
function output_storing(training_method::AbstractGaussianProcess, out_size, prediction_len)
    zeros(out_size, prediction_len), zeros(out_size, prediction_len)
end

#single matrix for other training methods
function output_storing(training_method, out_size, prediction_len)
    zeros(out_size, prediction_len)
end

#storing results for gaussian training, getting also the sigmas
function store_results!(training_method::AbstractGaussianProcess, out, output, i)
    output[1][:, i] = out[1]
    output[2][:, i] = out[2]
end

#general storing -> single matrix
function store_results!(training_method, out, output, i)
    output[:, i] = out
end