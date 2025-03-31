abstract type AbstractOutputLayer end
abstract type AbstractPrediction end

#general output layer struct
struct OutputLayer{T, I, S, L} <: AbstractOutputLayer
    training_method::T
    output_matrix::I
    out_size::S
    last_value::L
end

function Base.show(io::IO, ol::OutputLayer)
    print(io, "OutputLayer successfully trained with output size: ", ol.out_size)
end

#prediction types
"""
    Generative(prediction_len)

A prediction strategy that enables models to generate autonomous multi-step
forecasts by recursively feeding their own outputs back as inputs for
subsequent prediction steps.

# Parameters

  - `prediction_len`: The number of future steps to predict.

# Description

The `Generative` prediction method allows a model to perform multi-step
forecasting by using its own previous predictions as inputs for future predictions.

At each step, the model takes the current input, generates a prediction,
and then incorporates that prediction into the input for the next step.
This recursive process continues until the specified
number of prediction steps (`prediction_len`) is reached.
"""
struct Generative{T} <: AbstractPrediction
    prediction_len::T
end

struct Predictive{I, T} <: AbstractPrediction
    prediction_data::I
    prediction_len::T
end

"""
    Predictive(prediction_data)

A prediction strategy for supervised learning tasks,
where a model predicts labels based on a provided set
of input features (`prediction_data`).

# Parameters

  - `prediction_data`: The input data used for prediction, `feature` x `sample`

# Description

The `Predictive` prediction method uses the provided input data
(`prediction_data`) to produce corresponding labels or outputs based
on the learned relationships in the model.
"""
function Predictive(prediction_data::AbstractArray)
    prediction_len = size(prediction_data, 2)
    return Predictive(prediction_data, prediction_len)
end

function obtain_prediction(rc::AbstractReservoirComputer, prediction::Generative,
        x, output_layer::AbstractOutputLayer, args...;
        initial_conditions = output_layer.last_value)
    #x = last_state
    prediction_len = prediction.prediction_len
    train_method = output_layer.training_method
    out_size = output_layer.out_size
    output = output_storing(train_method, out_size, prediction_len, typeof(rc.states))
    out = initial_conditions

    for i in 1:prediction_len
        x, x_new = next_state_prediction!(rc, x, out, i, args...)
        out_tmp = get_prediction(train_method, output_layer, x_new)
        out = store_results!(train_method, out_tmp, output, i)
    end

    return output
end

function obtain_prediction(rc::AbstractReservoirComputer, prediction::Predictive,
        x, output_layer::AbstractOutputLayer, args...; kwargs...)
    prediction_len = prediction.prediction_len
    train_method = output_layer.training_method
    out_size = output_layer.out_size
    output = output_storing(train_method, out_size, prediction_len, typeof(rc.states))

    for i in 1:prediction_len
        y = @view prediction.prediction_data[:, i]
        x, x_new = next_state_prediction!(rc, x, y, i, args...)
        out_tmp = get_prediction(train_method, output_layer, x_new)
        out = store_results!(output_layer.training_method, out_tmp, output, i)
    end

    return output
end

#linear models
function get_prediction(training_method, output_layer::AbstractOutputLayer, x)
    return output_layer.output_matrix * x
end

#single matrix for other training methods
function output_storing(training_method, out_size, prediction_len, storing_type)
    return adapt(storing_type, zeros(out_size, prediction_len))
end

#general storing -> single matrix
function store_results!(training_method, out, output, i)
    output[:, i] = out
    return out
end
