abstract type AbstractEchoStateNetwork <: AbstractReservoirComputer end
struct ESN{I, S, V, N, T, O, M, B, ST, W, IS} <: AbstractEchoStateNetwork
    res_size::I
    train_data::S
    variation::V
    nla_type::N
    input_matrix::T
    reservoir_driver::O
    reservoir_matrix::M
    bias_vector::B
    states_type::ST
    washout::W
    states::IS
end

"""
    Default()

Sets the type of the ESN as the standard model. No parameters are needed.
"""
struct Default <: AbstractVariation end
struct Hybrid{T, K, O, I, S, D} <: AbstractVariation
    prior_model::T
    u0::K
    tspan::O
    dt::I
    datasize::S
    model_data::D
end

"""
    Hybrid(prior_model, u0, tspan, datasize)

Given the model parameters, returns an ```Hybrid``` variation of the ESN. This entails
a different training and prediction. Construction based on [1].

[1] Jaideep Pathak et al. "Hybrid Forecasting of Chaotic Processes: Using Machine
Learning in Conjunction with a Knowledge-Based Model" (2018)
"""
function Hybrid(prior_model, u0, tspan, datasize)
    trange = collect(range(tspan[1], tspan[2], length = datasize))
    dt = trange[2] - trange[1]
    tsteps = push!(trange, dt + trange[end])
    tspan_new = (tspan[1], dt + tspan[2])
    model_data = prior_model(u0, tspan_new, tsteps)
    return Hybrid(prior_model, u0, tspan, dt, datasize, model_data)
end

"""
    ESN(train_data;
        variation = Default(),
        input_layer = DenseLayer(),
        reservoir = RandSparseReservoir(),
        bias = NullLayer(),
        reservoir_driver = RNN(),
        nla_type = NLADefault(),
        states_type = StandardStates())
    (esn::ESN)(prediction::AbstractPrediction,
        output_layer::AbstractOutputLayer;
        initial_conditions=output_layer.last_value,
        last_state=esn.states[:, end])

Constructor for the Echo State Network model. It requires the reservoir size as the input
and the data for the training. It returns a struct ready to be trained with the states
already harvested.

After the training, this struct can be used for the prediction following the second
function call. This will take as input a prediction type and the output layer from the
training. The ```initial_conditions``` and ```last_state``` parameters can be left as
they are, unless there is a specific reason to change them. All the components are
detailed in the API documentation. More examples are given in the general documentation.
"""
function ESN(train_data;
             variation = Default(),
             input_layer = DenseLayer(),
             reservoir = RandSparseReservoir(100),
             bias = NullLayer(),
             reservoir_driver = RNN(),
             nla_type = NLADefault(),
             states_type = StandardStates(),
             washout = 0,
             matrix_type = typeof(train_data))
    if variation isa Hybrid
        train_data = vcat(train_data, variation.model_data[:, 1:(end - 1)])
    end

    if states_type isa AbstractPaddedStates
        in_size = size(train_data, 1) + 1
        train_data = vcat(Adapt.adapt(matrix_type, ones(1, size(train_data, 2))),
                          train_data)
    else
        in_size = size(train_data, 1)
    end

    input_matrix, reservoir_matrix, bias_vector, res_size = obtain_layers(in_size,
                                                                          input_layer,
                                                                          reservoir, bias;
                                                                          matrix_type = matrix_type)

    inner_res_driver = reservoir_driver_params(reservoir_driver, res_size, in_size)
    states = create_states(inner_res_driver, train_data, washout, reservoir_matrix,
                           input_matrix, bias_vector)
    train_data = train_data[:, (washout + 1):end]

    ESN(sum(res_size), train_data, variation, nla_type, input_matrix,
        inner_res_driver, reservoir_matrix, bias_vector, states_type, washout,
        states)
end

#shallow esn construction
function obtain_layers(in_size,
                       input_layer,
                       reservoir,
                       bias;
                       matrix_type = Matrix{Float64})
    input_res_size = get_ressize(reservoir)
    input_matrix = create_layer(input_layer, input_res_size, in_size,
                                matrix_type = matrix_type)
    res_size = size(input_matrix, 1) #WeightedInput actually changes the res size
    reservoir_matrix = create_reservoir(reservoir, res_size, matrix_type = matrix_type)
    @assert size(reservoir_matrix, 1) == res_size
    bias_vector = create_layer(bias, res_size, 1, matrix_type = matrix_type)
    return input_matrix, reservoir_matrix, bias_vector, res_size
end

#deep esn construction
#there is a bug going on with WeightedLayer in this construction.
#it works for eny other though
function obtain_layers(in_size,
                       input_layer,
                       reservoir::Vector,
                       bias;
                       matrix_type = Matrix{Float64})
    esn_depth = length(reservoir)
    input_res_sizes = [get_ressize(reservoir[i]) for i in 1:esn_depth]
    in_sizes = zeros(Int, esn_depth)
    in_sizes[2:end] = input_res_sizes[1:(end - 1)]
    in_sizes[1] = in_size

    if input_layer isa Array
        input_matrix = [create_layer(input_layer[j], input_res_sizes[j], in_sizes[j],
                                     matrix_type = matrix_type) for j in 1:esn_depth]
    else
        _input_layer = fill(input_layer, esn_depth)
        input_matrix = [create_layer(_input_layer[k], input_res_sizes[k], in_sizes[k],
                                     matrix_type = matrix_type) for k in 1:esn_depth]
    end

    res_sizes = [get_ressize(input_matrix[j]) for j in 1:esn_depth]
    reservoir_matrix = [create_reservoir(reservoir[k], res_sizes[k],
                                         matrix_type = matrix_type) for k in 1:esn_depth]

    if bias isa Array
        bias_vector = [create_layer(bias[j], res_sizes[j], 1, matrix_type = matrix_type)
                       for j in 1:esn_depth]
    else
        _bias = fill(bias, esn_depth)
        bias_vector = [create_layer(_bias[k], res_sizes[k], 1, matrix_type = matrix_type)
                       for k in 1:esn_depth]
    end

    return input_matrix, reservoir_matrix, bias_vector, res_sizes
end

function (esn::ESN)(prediction::AbstractPrediction,
                    output_layer::AbstractOutputLayer;
                    last_state = esn.states[:, [end]],
                    kwargs...)
    variation = esn.variation
    pred_len = prediction.prediction_len

    if variation isa Hybrid
        model = variation.prior_model
        predict_tsteps = [variation.tspan[2] + variation.dt]
        [append!(predict_tsteps, predict_tsteps[end] + variation.dt) for i in 1:pred_len]
        tspan_new = (variation.tspan[2] + variation.dt, predict_tsteps[end])
        u0 = variation.model_data[:, end]
        model_pred_data = model(u0, tspan_new, predict_tsteps)[:, 2:end]
        return obtain_esn_prediction(esn, prediction, last_state, output_layer,
                                     model_pred_data;
                                     kwargs...)
    else
        return obtain_esn_prediction(esn, prediction, last_state, output_layer;
                                     kwargs...)
    end
end

#training dispatch on esn
"""
    train(esn::AbstractEchoStateNetwork, target_data, training_method=StandardRidge(0.0))

Training of the built ESN over the ```target_data```. The default training method is
RidgeRegression. The output is an ```OutputLayer``` object to be fed to the esn call
for the prediction.
"""
function train(esn::AbstractEchoStateNetwork,
               target_data,
               training_method = StandardRidge(0.0))
    variation = esn.variation

    if esn.variation isa Hybrid
        states = vcat(esn.states, esn.variation.model_data[:, 2:end])
    else
        states = esn.states
    end
    states_new = esn.states_type(esn.nla_type, states, esn.train_data[:, 1:end])

    return _train(states_new, target_data, training_method)
end

function pad_esnstate(variation::Hybrid, states_type, x_pad, x, model_prediction_data)
    x_tmp = vcat(x, model_prediction_data)
    x_pad = pad_state!(states_type, x_pad, x_tmp)
end

function pad_esnstate!(variation, states_type, x_pad, x, args...)
    x_pad = pad_state!(states_type, x_pad, x)
end
