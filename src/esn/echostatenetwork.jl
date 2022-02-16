abstract type AbstractEchoStateNetwork <: AbstractReservoirComputer end
struct ESN{I,S,V,N,T,O,M,B,ST,W,IS} <: AbstractEchoStateNetwork
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
struct Hybrid{T,K,O,I,S,D} <: AbstractVariation
    prior_model::T
    u0::K
    tspan::O
    dt::I
    datasize::S
    model_data::D
end

"""
    Hybrid(prior_model, u0, tspan, datasize)

Given the model parameters returns an ```Hybrid``` variation of the ESN. This entails a different training 
and prediction. Construction based on [1].

[1] Jaideep Pathak et al. "Hybrid Forecasting of Chaotic Processes: Using Machine Learning in Conjunction with a Knowledge-Based Model" (2018)
"""
function Hybrid(prior_model, u0, tspan, datasize)
    trange = collect(range(tspan[1], tspan[2], length = datasize))
    dt = trange[2]-trange[1]
    tsteps = push!(trange, dt + trange[end])
    tspan_new = (tspan[1], dt+tspan[2])
    model_data = prior_model(u0, tspan_new, tsteps)
    Hybrid(prior_model, u0, tspan, dt, datasize, model_data)
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

Constructor for the Echo State Network model. It requires the erserovir size as the input and the data for the training. 
It returns a struct ready to be trained with the states already harvested. 

After the training this struct can be used for the prediction following the second function call. This will take as input a 
prediction type and the output layer from the training. The ```initial_conditions``` and ```last_state``` parameters 
can be left as they are, unless there is a specific reason to change them. All the components are detailed in the 
API documentation. More examples are given in the general documentation.
"""
function ESN(train_data;
             variation = Default(),
             input_layer = DenseLayer(),
             reservoir = RandSparseReservoir(),
             bias = NullLayer(),
             reservoir_driver = RNN(),
             nla_type = NLADefault(),
             states_type = StandardStates(),
             washout = 0)

    variation isa Hybrid ? train_data = vcat(train_data, variation.model_data[:, 1:end-1]) : nothing
    input_res_size = get_ressize(reservoir)

    if states_type isa AbstractPaddedStates
        in_size = size(train_data, 1) + 1
        train_data = vcat(ones(1, size(train_data, 2)), train_data)
    else 
        in_size = size(train_data, 1)
    end

    input_matrix = create_layer(input_layer, input_res_size, in_size)
    res_size = size(input_matrix, 1) #WeightedInput actually changes the res size
    reservoir_matrix = create_reservoir(reservoir, res_size)
    @assert size(reservoir_matrix, 1) == res_size
    bias_vector = create_layer(bias, res_size, 1)

    inner_reservoir_driver = reservoir_driver_params(reservoir_driver, res_size, in_size)
    states = create_states(inner_reservoir_driver, train_data, washout, 
        reservoir_matrix, input_matrix, bias_vector)
    train_data = train_data[:,washout+1:end]

    ESN(res_size, train_data, variation, nla_type, input_matrix, inner_reservoir_driver, 
        reservoir_matrix, bias_vector, states_type, washout, states)
end


function (esn::ESN)(prediction::AbstractPrediction,
    output_layer::AbstractOutputLayer;
    initial_conditions=output_layer.last_value,
    last_state=esn.states[:, end])

    variation = esn.variation

    if variation isa Hybrid
        predict_tsteps = [variation.tspan[2]+variation.dt]
        [append!(predict_tsteps, predict_tsteps[end]+variation.dt) for i in 1:prediction.prediction_len]
        tspan_new = (variation.tspan[2]+variation.dt, predict_tsteps[end])
        u0 = variation.model_data[:, end]
        model_prediction_data = variation.prior_model(u0, tspan_new, predict_tsteps)[:, 2:end]
        return obtain_prediction(esn, prediction, last_state, output_layer, model_prediction_data; initial_conditions=initial_conditions)
    else 
        return obtain_prediction(esn, prediction, last_state, output_layer; initial_conditions=initial_conditions)
    end
end

#training dispatch on esn
"""
    train(esn::AbstractEchoStateNetwork, target_data, training_method=StandardRidge(0.0))

Training of the built ESN over the ```target_data```. The default training method is RidgeRegression. The output is 
an ```OutputLayer``` object to be fed at the esn call for the prediction.
"""
function train(esn::AbstractEchoStateNetwork, target_data, training_method=StandardRidge(0.0))

    esn.variation isa Hybrid ? states = vcat(esn.states, esn.variation.model_data[:, 2:end]) : states=esn.states
    states_new = esn.states_type(esn.nla_type, states, esn.train_data[:,1:end])

    _train(states_new, target_data, training_method)
end

#prediction dispatch on esn 
function next_state_prediction!(esn::ESN, x, out, i, args...)
    _variation_prediction!(esn.variation, esn, x, out, i, args...)
end

#dispatch the prediction on the esn variation
function _variation_prediction!(variation, esn, x, out, i, args...)
    out_pad = pad_state(esn.states_type, out)
    x = next_state(esn.reservoir_driver, x[1:esn.res_size], out_pad, esn.reservoir_matrix, esn.input_matrix, esn.bias_vector)
    x_new = esn.states_type(esn.nla_type, x, out_pad)
    x, x_new
end

function _variation_prediction!(variation::Hybrid, esn, x, out, i, model_prediction_data)
    out_tmp = vcat(out, model_prediction_data[:,i])
    out_pad = pad_state(esn.states_type, out_tmp)
    x = next_state(esn.reservoir_driver, x[1:esn.res_size], out_pad, esn.reservoir_matrix, esn.input_matrix, esn.bias_vector)
    x_tmp = vcat(x, model_prediction_data[:,i])
    x_new = esn.states_type(esn.nla_type, x_tmp, out_pad)
    x, x_new
end